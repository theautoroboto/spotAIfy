package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"spotaify-web/internal/auth"
	"spotaify-web/internal/db"
	"spotaify-web/internal/spotify"
)

// ── Run registry ─────────────────────────────────────────────────────────────

type run struct {
	lines  chan string
	done   chan struct{}
	cancel context.CancelFunc
}

var (
	runsMu sync.Mutex
	runs   = make(map[string]*run)
)

func newRun(cancel context.CancelFunc) (string, *run) {
	id := randomHex(16)
	r := &run{lines: make(chan string, 256), done: make(chan struct{}), cancel: cancel}
	runsMu.Lock()
	runs[id] = r
	runsMu.Unlock()
	return id, r
}

func getRun(id string) (*run, bool) {
	runsMu.Lock()
	defer runsMu.Unlock()
	r, ok := runs[id]
	return r, ok
}

func deleteRun(id string) {
	runsMu.Lock()
	delete(runs, id)
	runsMu.Unlock()
}

// ── Templates ────────────────────────────────────────────────────────────────

var tmpls map[string]*template.Template

func loadTemplates() {
	tmpls = make(map[string]*template.Template)
	for _, name := range []string{"index.html", "login.html"} {
		tmpls[name] = template.Must(
			template.ParseFiles("templates/base.html", "templates/"+name),
		)
	}
}

func render(w http.ResponseWriter, name string, data any) {
	t, ok := tmpls[name]
	if !ok {
		http.Error(w, "unknown template: "+name, http.StatusInternalServerError)
		return
	}
	if err := t.ExecuteTemplate(w, "base", data); err != nil {
		log.Printf("template %s: %v", name, err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func randomHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ── Handlers ─────────────────────────────────────────────────────────────────

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	username, _ := auth.GetSession(r)
	render(w, "index.html", map[string]any{
		"Username":       username,
		"SpotifyLinked":  db.HasToken(username),
	})
}

func handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		render(w, "login.html", map[string]any{"Next": r.URL.Query().Get("next")})
		return
	}
	username := strings.TrimSpace(r.FormValue("username"))
	password := r.FormValue("password")
	next := r.FormValue("next")
	if next == "" {
		next = "/"
	}
	if !auth.CheckPassword(username, password) {
		render(w, "login.html", map[string]any{"Error": "Invalid username or password.", "Next": next})
		return
	}
	auth.SetSession(w, username)
	http.Redirect(w, r, next, http.StatusFound)
}

func handleLogout(w http.ResponseWriter, r *http.Request) {
	auth.ClearSession(w)
	http.Redirect(w, r, "/login", http.StatusFound)
}

func handleSpotifyConnect(w http.ResponseWriter, r *http.Request) {
	username, _ := auth.GetSession(r)
	state := randomHex(16)
	// Embed username in state so callback knows who to associate the token with.
	// Format: "<username>:<nonce>"
	state = username + ":" + state
	http.Redirect(w, r, spotify.AuthURL(state), http.StatusFound)
}

func handleSpotifyCallback(w http.ResponseWriter, r *http.Request) {
	state := r.URL.Query().Get("state")
	code := r.URL.Query().Get("code")
	errParam := r.URL.Query().Get("error")

	if errParam != "" {
		http.Error(w, "Spotify auth denied: "+errParam, http.StatusBadRequest)
		return
	}

	// Extract username from state.
	idx := strings.Index(state, ":")
	if idx < 1 {
		http.Error(w, "Invalid state", http.StatusBadRequest)
		return
	}
	username := state[:idx]

	// Verify the session belongs to the same user (prevents CSRF).
	sessionUser, ok := auth.GetSession(r)
	if !ok || sessionUser != username {
		http.Error(w, "Session mismatch", http.StatusForbidden)
		return
	}

	if err := spotify.ExchangeAndStore(username, code); err != nil {
		http.Error(w, "Token exchange failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	http.Redirect(w, r, "/?connected=1", http.StatusFound)
}

// handleRun accepts the playlist form, spawns the Python agent subprocess, and
// returns a run ID that the browser uses to open the SSE stream.
func handleRun(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	username, _ := auth.GetSession(r)

	// Build CLI args from form fields.
	args := buildArgs(r)

	// Get a fresh Spotify token and write a spotipy-compatible cache file.
	accessToken, err := spotify.FreshToken(username)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	cachePath, err := writeTokenCache(username, accessToken)
	if err != nil {
		http.Error(w, "failed to write token cache", http.StatusInternalServerError)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	runID, run := newRun(cancel)

	go func() {
		defer func() {
			close(run.done)
			deleteRun(runID)
			os.Remove(cachePath)
			cancel()
		}()

		cmd := exec.CommandContext(ctx, "python", append([]string{"-u", "-m", "spotaify.agent"}, args...)...)
		cmd.Env = append(os.Environ(),
			"SPOTIFY_CACHE_PATH="+cachePath,
			"PYTHONUNBUFFERED=1",
		)
		cmd.Stderr = os.Stderr // surface Python errors in server logs

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			run.lines <- "[error] " + err.Error()
			return
		}
		if err := cmd.Start(); err != nil {
			run.lines <- "[error] " + err.Error()
			return
		}

		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			run.lines <- scanner.Text()
		}
		if err := cmd.Wait(); err != nil && ctx.Err() == nil {
			run.lines <- "[error] agent exited: " + err.Error()
		}
	}()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"runId": runID})
}

// handleStream is the SSE endpoint. HTMX connects here after /run returns a runId.
func handleStream(w http.ResponseWriter, r *http.Request) {
	runID := r.PathValue("id")
	run, ok := getRun(runID)
	if !ok {
		http.Error(w, "run not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	for {
		select {
		case line, open := <-run.lines:
			if !open {
				fmt.Fprintf(w, "event: done\ndata: \n\n")
				flusher.Flush()
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", template.HTMLEscapeString(line))
			flusher.Flush()
		case <-run.done:
			// Drain remaining lines.
			for {
				select {
				case line := <-run.lines:
					fmt.Fprintf(w, "data: %s\n\n", template.HTMLEscapeString(line))
					flusher.Flush()
				default:
					fmt.Fprintf(w, "event: done\ndata: \n\n")
					flusher.Flush()
					return
				}
			}
		case <-r.Context().Done():
			return
		}
	}
}

// ── History handlers ──────────────────────────────────────────────────────────

func handleHistoryGet(w http.ResponseWriter, r *http.Request) {
	username, _ := auth.GetSession(r)
	runs, err := db.GetRuns(username)
	if err != nil {
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}
	if runs == nil {
		runs = []db.Run{} // encode as [] not null
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(runs)
}

type saveRunReq struct {
	Label string   `json:"label"`
	Lines []string `json:"lines"`
}

func handleHistorySave(w http.ResponseWriter, r *http.Request) {
	username, _ := auth.GetSession(r)
	var req saveRunReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if err := db.InsertRun(username, req.Label, req.Lines); err != nil {
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func handleHistoryClear(w http.ResponseWriter, r *http.Request) {
	username, _ := auth.GetSession(r)
	if err := db.DeleteRuns(username); err != nil {
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func handleCancel(w http.ResponseWriter, r *http.Request) {
	runID := r.PathValue("id")
	run, ok := getRun(runID)
	if !ok {
		http.Error(w, "run not found", http.StatusNotFound)
		return
	}
	run.cancel()
	w.WriteHeader(http.StatusNoContent)
}

// ── Form → CLI args ───────────────────────────────────────────────────────────

func buildArgs(r *http.Request) []string {
	var args []string

	switch r.FormValue("mode") {
	case "dna":
		args = append(args, "--dna", r.FormValue("dna_track"))
		if a := r.FormValue("dna_artist"); a != "" {
			args = append(args, "--dna-artist", a)
		}
	case "connection":
		args = append(args, "--artist", r.FormValue("artist"), "--connections")
		if r.FormValue("depth") != "" {
			args = append(args, "--depth", r.FormValue("depth"))
		}
		if r.FormValue("new_only") == "on" {
			args = append(args, "--new-only")
		}
		if r.FormValue("match_sound") == "on" {
			args = append(args, "--match-sound")
		}
	case "rediscovery":
		args = append(args, "--rediscovery")
		if v := r.FormValue("stale_days"); v != "" {
			args = append(args, "--stale-days", v)
		}
		if v := r.FormValue("min_plays"); v != "" {
			args = append(args, "--min-plays", v)
		}
	default: // sonic
		if prompt := strings.TrimSpace(r.FormValue("prompt")); prompt != "" {
			args = append(args, prompt)
		}
	}

	if v := r.FormValue("count"); v != "" {
		args = append(args, "--count", v)
	}
	if v := r.FormValue("energy"); v != "" {
		args = append(args, "--energy", v)
	}
	if v := r.FormValue("valence"); v != "" {
		args = append(args, "--valence", v)
	}
	if v := r.FormValue("tempo"); v != "" {
		args = append(args, "--tempo", v)
	}
	return args
}

// writeTokenCache writes a spotipy-compatible JSON token cache and returns the path.
func writeTokenCache(username, accessToken string) (string, error) {
	tok, err := db.GetToken(username)
	if err != nil {
		return "", err
	}
	cache := map[string]any{
		"access_token":  accessToken,
		"token_type":    "Bearer",
		"refresh_token": tok.RefreshToken,
		"scope":         tok.Scope,
		"expires_at":    float64(tok.ExpiresAt.Unix()),
	}
	dir := os.TempDir()
	path := filepath.Join(dir, "spotaify_token_"+username+".json")
	f, err := os.CreateTemp(dir, "spotaify_token_"+username+"_*.json")
	if err != nil {
		return "", err
	}
	path = f.Name()
	if err := json.NewEncoder(f).Encode(cache); err != nil {
		f.Close()
		os.Remove(path)
		return "", err
	}
	f.Close()
	return path, nil
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	// Auth setup.
	secret := envOr("SESSION_SECRET", "dev-secret-change-me")
	if secret == "dev-secret-change-me" {
		log.Printf("WARNING: SESSION_SECRET is not set — using insecure default. Set it in your .env file.")
	}
	auth.SetSecret(secret)
	auth.Users = auth.ParseUsers(os.Getenv("USERS"))
	if len(auth.Users) == 0 {
		log.Fatal("USERS env var is required (e.g. alice:$2b$10$...)")
	}

	// Spotify config.
	spotify.Config.ClientID = os.Getenv("SPOTIFY_CLIENT_ID")
	spotify.Config.ClientSecret = os.Getenv("SPOTIFY_CLIENT_SECRET")
	spotify.Config.RedirectURI = envOr("SPOTIFY_WEB_REDIRECT_URI", "http://localhost:8000/spotify/callback")

	// Database.
	dbPath := envOr("DB_PATH", "/data/spotaify.db")
	if err := db.Init(dbPath); err != nil {
		log.Fatalf("db init: %v", err)
	}

	// Templates.
	loadTemplates()

	// Routes.
	mux := http.NewServeMux()

	// Public.
	mux.HandleFunc("GET /login", handleLogin)
	mux.HandleFunc("POST /login", handleLogin)
	mux.HandleFunc("GET /static/", func(w http.ResponseWriter, r *http.Request) {
		http.StripPrefix("/static/", http.FileServer(http.Dir("static"))).ServeHTTP(w, r)
	})

	// Protected.
	protected := http.NewServeMux()
	protected.HandleFunc("GET /", handleIndex)
	protected.HandleFunc("POST /logout", handleLogout)
	protected.HandleFunc("GET /spotify/connect", handleSpotifyConnect)
	protected.HandleFunc("GET /spotify/callback", handleSpotifyCallback)
	protected.HandleFunc("POST /run", handleRun)
	protected.HandleFunc("GET /stream/{id}", handleStream)
	protected.HandleFunc("DELETE /run/{id}", handleCancel)
	protected.HandleFunc("GET /history", handleHistoryGet)
	protected.HandleFunc("POST /history", handleHistorySave)
	protected.HandleFunc("DELETE /history", handleHistoryClear)

	mux.Handle("/", auth.RequireAuth(protected))

	port := envOr("PORT", "8000")
	log.Printf("spotAIfy web — listening on :%s", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}

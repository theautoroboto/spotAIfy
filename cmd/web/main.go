package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"spotaify-web/internal/ai"
	"spotaify-web/internal/auth"
	"spotaify-web/internal/db"
	"spotaify-web/internal/history"
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

var tmplFuncs = template.FuncMap{
	"add":        func(a, b int) int { return a + b },
	"pct":        func(f float64) int { return int(f * 100) },
	"paragraphs": func(s string) []string { return strings.Split(strings.TrimSpace(s), "\n\n") },
}

func loadTemplates() {
	tmpls = make(map[string]*template.Template)
	for _, name := range []string{"index.html", "login.html", "profile.html"} {
		tmpls[name] = template.Must(
			template.New("").Funcs(tmplFuncs).ParseFiles("templates/base.html", "templates/"+name),
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

// realIP returns the client IP, preferring X-Forwarded-For set by Caddy.
func realIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		if ip := strings.TrimSpace(strings.SplitN(xff, ",", 2)[0]); ip != "" {
			return ip
		}
	}
	host, _, _ := net.SplitHostPort(r.RemoteAddr)
	if host != "" {
		return host
	}
	return r.RemoteAddr
}

// safeNext validates the redirect target to prevent open-redirect attacks.
func safeNext(next string) string {
	if next == "" || !strings.HasPrefix(next, "/") || strings.HasPrefix(next, "//") {
		return "/"
	}
	return next
}

// requestLogger logs method, path, and response status for every request.
func requestLogger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rw := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r)
		log.Printf("%s %s %d [%s]", r.Method, r.URL.RequestURI(), rw.status, realIP(r))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (sw *statusWriter) WriteHeader(code int) {
	sw.status = code
	sw.ResponseWriter.WriteHeader(code)
}

func (sw *statusWriter) Flush() {
	if f, ok := sw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// securityHeaders adds defensive HTTP headers to every response.
func securityHeaders(next http.Handler) http.Handler {
	const csp = "default-src 'self'; " +
		"script-src 'self' 'unsafe-inline' https://unpkg.com; " +
		"style-src 'self' 'unsafe-inline'; " +
		"img-src 'self' data: https://*.scdn.co https://*.spotifycdn.com; " +
		"connect-src 'self'; " +
		"frame-ancestors 'none'"
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Frame-Options", "DENY")
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "strict-origin-when-cross-origin")
		h.Set("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
		h.Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		h.Set("Content-Security-Policy", csp)
		next.ServeHTTP(w, r)
	})
}

// ── Handlers ─────────────────────────────────────────────────────────────────

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	username, _ := auth.GetSession(r)

	// Load years from listening history for the Expand year dropdown.
	var expandYears []int
	histBase := envOr("HISTORY_DIR", "data/history")
	if hist, err := history.Load(filepath.Join(histBase, username)); err == nil {
		for _, y := range hist.YearlyTrend {
			expandYears = append(expandYears, y.Year)
		}
	}

	render(w, "index.html", map[string]any{
		"Username":      username,
		"SpotifyLinked": db.HasToken(username),
		"ExpandYears":   expandYears,
	})
}

func handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		render(w, "login.html", map[string]any{"Next": r.URL.Query().Get("next")})
		return
	}
	next := safeNext(r.FormValue("next"))

	if !auth.LoginAllowed(realIP(r)) {
		render(w, "login.html", map[string]any{
			"Error": "Too many login attempts. Please try again later.",
			"Next":  next,
		})
		return
	}

	username := strings.TrimSpace(r.FormValue("username"))
	password := r.FormValue("password")
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
		log.Printf("spotify callback: session mismatch — session=%q state_user=%q ok=%v", sessionUser, username, ok)
		http.Error(w, "Session mismatch — please log in and try connecting again.", http.StatusForbidden)
		return
	}

	if err := spotify.ExchangeAndStore(username, code); err != nil {
		log.Printf("spotify callback: exchange failed for %q: %v", username, err)
		http.Error(w, "Token exchange failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	log.Printf("spotify callback: connected %q successfully", username)
	http.Redirect(w, r, "/profile", http.StatusFound)
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
			"SPOTIFY_ACCESS_TOKEN="+accessToken,
			"PYTHONUNBUFFERED=1",
			"SPOTAIFY_USERNAME="+username,
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
			fmt.Fprintf(w, "data: %s\n\n", line)
			flusher.Flush()
		case <-run.done:
			// Drain remaining lines.
			for {
				select {
				case line := <-run.lines:
					fmt.Fprintf(w, "data: %s\n\n", line)
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

// ── Profile handler ───────────────────────────────────────────────────────────

func handleProfile(w http.ResponseWriter, r *http.Request) {
	username, _ := auth.GetSession(r)

	// Load local streaming history (primary data source). A missing directory
	// is not an error — the profile renders with whatever data is available.
	histDir := envOr("HISTORY_DIR", "data/history")
	hist, err := history.Load(filepath.Join(histDir, username))
	if err != nil {
		log.Printf("profile[%s]: history load error (continuing): %v", username, err)
		hist = &history.Stats{}
	}

	// Fetch Spotify API data for user info + metadata enrichment.
	accessToken, spotifyErr := spotify.FreshToken(username)
	var pd *spotify.ProfileData
	if spotifyErr == nil {
		pd, _ = spotify.FetchProfileData(accessToken)
	}

	// Enrich top tracks with album art + Spotify URLs + audio features.
	var audioFeatures *spotify.AudioFeaturesSummary
	var recommendations []spotify.Recommendation
	var persona *ai.Persona
	if spotifyErr != nil {
		log.Printf("profile[%s]: spotify token error: %v", username, spotifyErr)
	} else if len(hist.TopTracks) == 0 {
		log.Printf("profile[%s]: no top tracks in history, skipping enrichment", username)
	}
	if spotifyErr == nil && len(hist.TopTracks) > 0 {
		ids := make([]string, 0, len(hist.TopTracks))
		for _, t := range hist.TopTracks {
			ids = append(ids, t.TrackID)
		}
		trackMeta := spotify.FetchTracksMeta(accessToken, ids)
		audioFeatures = spotify.FetchAudioFeaturesSummary(accessToken, ids)

		// Collect unique artist IDs so we can fetch artist images in one batch.
		artistIDByTrackID := map[string]string{}
		uniqueArtistIDs := map[string]struct{}{}
		for _, t := range hist.TopTracks {
			if m, ok := trackMeta[t.TrackID]; ok {
				t.ImageURL = m.ImageURL
				t.SpotifyURL = m.SpotifyURL
				if m.ArtistID != "" {
					artistIDByTrackID[t.TrackID] = m.ArtistID
					uniqueArtistIDs[m.ArtistID] = struct{}{}
				}
			}
		}

		// Build artist name → artist ID map for enriching top artists.
		artistIDByName := map[string]string{}
		for _, t := range hist.TopTracks {
			if aid, ok := artistIDByTrackID[t.TrackID]; ok && t.ArtistName != "" {
				artistIDByName[t.ArtistName] = aid
			}
		}

		// Batch-fetch artist metadata.
		ids2 := make([]string, 0, len(uniqueArtistIDs))
		for id := range uniqueArtistIDs {
			ids2 = append(ids2, id)
		}
		artistMeta := spotify.FetchArtistsMeta(accessToken, ids2)

		// Enrich top artists.
		for _, a := range hist.TopArtists {
			if aid, ok := artistIDByName[a.ArtistName]; ok {
				if m, ok := artistMeta[aid]; ok {
					a.ImageURL = m.ImageURL
					a.SpotifyURL = m.SpotifyURL
					a.Genres = m.Genres
				}
			}
		}

		// ── Recommendations + Persona (parallel) ─────────────────────────────

		// Seed artist IDs: top 5 from listening history.
		var seedArtistIDs []string
		for _, a := range hist.TopArtists {
			if len(seedArtistIDs) >= 5 {
				break
			}
			if aid, ok := artistIDByName[a.ArtistName]; ok {
				seedArtistIDs = append(seedArtistIDs, aid)
			}
		}

		// Known artists set for filtering recommendations.
		knownArtists := make(map[string]struct{}, len(hist.TopArtists))
		for _, a := range hist.TopArtists {
			knownArtists[a.ArtistName] = struct{}{}
		}

		// Persona input: aggregate stats.
		var totalPlays, totalSkip, totalCompl int
		for _, t := range hist.TopTracks {
			totalPlays += t.PlayCount
			totalSkip += t.SkipCount
			totalCompl += t.CompletionCount
		}
		avgSkip, avgCompl := 0.0, 0.0
		if totalPlays > 0 {
			avgSkip = float64(totalSkip) / float64(totalPlays)
			avgCompl = float64(totalCompl) / float64(totalPlays)
		}
		peak := 0
		for h, v := range hist.HourlyPattern {
			if v > hist.HourlyPattern[peak] {
				peak = h
			}
		}
		var topArtistNames []string
		for _, a := range hist.TopArtists {
			topArtistNames = append(topArtistNames, a.ArtistName)
			if len(topArtistNames) >= 5 {
				break
			}
		}
		var topGenres []string
		if pd != nil {
			for _, g := range pd.Insights.TopGenres {
				topGenres = append(topGenres, g.Genre)
				if len(topGenres) >= 5 {
					break
				}
			}
		}
		if len(topGenres) == 0 {
			seen := map[string]bool{}
			for _, a := range hist.TopArtists {
				for _, g := range a.Genres {
					if !seen[g] {
						topGenres = append(topGenres, g)
						seen[g] = true
					}
					if len(topGenres) >= 5 {
						break
					}
				}
				if len(topGenres) >= 5 {
					break
				}
			}
		}

		var wg2 sync.WaitGroup
		wg2.Add(2)
		go func() {
			defer wg2.Done()
			recommendations = spotify.FetchRecommendations(accessToken, seedArtistIDs, audioFeatures, knownArtists)
		}()
		go func() {
			defer wg2.Done()
			// Use cached persona if available.
			if cached, err := db.GetPersonaJSON(username); err == nil {
				var p ai.Persona
				if json.Unmarshal([]byte(cached), &p) == nil {
					persona = &p
					return
				}
			}
			earliestYear := 0
			if !hist.EarliestPlay.IsZero() {
				earliestYear = hist.EarliestPlay.Year()
			}
			inp := ai.PersonaInput{
				TotalHours:        hist.TotalHours(),
				EarliestYear:      earliestYear,
				YearsActive:       len(hist.YearlyTrend),
				UniqueArtists:     hist.UniqueArtistCount,
				UniqueTracks:      hist.UniqueTrackCount,
				TopArtistNames:    topArtistNames,
				TopGenres:         topGenres,
				PeakHour:          peak,
				AvgSkipRate:       avgSkip,
				AvgCompletionRate: avgCompl,
			}
			if audioFeatures != nil {
				inp.Energy = audioFeatures.Energy
				inp.Danceability = audioFeatures.Danceability
				inp.Valence = audioFeatures.Valence
				inp.Acousticness = audioFeatures.Acousticness
				inp.Instrumentalness = audioFeatures.Instrumentalness
				inp.Liveness = audioFeatures.Liveness
				inp.Tempo = audioFeatures.Tempo
				inp.VibeLabel = audioFeatures.VibeLabel
				inp.VibeDesc = audioFeatures.VibeDesc
			}
			var perr error
			persona, perr = ai.GeneratePersona(inp)
			if perr != nil {
				log.Printf("persona: generation failed: %v", perr)
			} else if persona == nil {
				log.Printf("persona: API key missing or returned nil")
			} else {
				log.Printf("persona: generated archetype=%q", persona.Archetype)
				if b, err := json.Marshal(persona); err == nil {
					db.SavePersonaJSON(username, string(b))
				}
			}
		}()
		wg2.Wait()
	}

	// Derive per-user aura parameters from listening history when audio features unavailable.
	aura := computeAura(hist, audioFeatures, username)

	// Cache-busting timestamp for the background image URL. Using the file's
	// mod-time ensures the browser fetches a fresh image whenever the file changes.
	bgDir := envOr("BG_CACHE_DIR", "/data/backgrounds")
	var bgTimestamp int64
	if info, err := os.Stat(filepath.Join(bgDir, username+".jpg")); err == nil {
		bgTimestamp = info.ModTime().Unix()
	}

	render(w, "profile.html", map[string]any{
		"Title":           "Profile",
		"Username":        username,
		"SpotifyLinked":   db.HasToken(username),
		"History":         hist,
		"Spotify":         pd,
		"Audio":           audioFeatures,
		"Aura":            aura,
		"Recommendations": recommendations,
		"Persona":         persona,
		"BGTimestamp":     bgTimestamp,
	})
}

// ── Profile background ────────────────────────────────────────────────────────

// handleProfileBg serves the AI-generated album-art background for the logged-in
// user. On the first request it calls the HF Inference API (may take ~20 s), caches
// the result to /data/backgrounds/{username}.jpg, and then serves it. All subsequent
// requests are served instantly from the cache.
func handleProfileBg(w http.ResponseWriter, r *http.Request) {
	username, _ := auth.GetSession(r)

	bgDir := envOr("BG_CACHE_DIR", "/data/backgrounds")
	if err := os.MkdirAll(bgDir, 0o755); err != nil {
		http.Error(w, "cache dir error", http.StatusInternalServerError)
		return
	}
	cachePath := filepath.Join(bgDir, username+".jpg")

	// Serve cached image if present.
	if img, err := os.ReadFile(cachePath); err == nil {
		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("Cache-Control", "max-age=3600")
		_, _ = w.Write(img)
		return
	}

	// Build generation inputs from cached persona + history + aura.
	var inp ai.BackgroundInput

	if cached, err := db.GetPersonaJSON(username); err == nil {
		var p ai.Persona
		if json.Unmarshal([]byte(cached), &p) == nil {
			inp.Archetype = p.Archetype
			inp.Traits = p.Traits
			inp.Headline = p.Headline
		}
	}

	histBase := envOr("HISTORY_DIR", "data/history")
	hist, _ := history.Load(filepath.Join(histBase, username))

	var topGenres []string
	seen := map[string]bool{}
	for _, a := range hist.TopArtists {
		for _, g := range a.Genres {
			if !seen[g] {
				topGenres = append(topGenres, g)
				seen[g] = true
			}
			if len(topGenres) >= 5 {
				break
			}
		}
		if len(topGenres) >= 5 {
			break
		}
	}
	inp.TopGenres = topGenres

	for _, a := range hist.TopArtists {
		inp.TopArtists = append(inp.TopArtists, a.ArtistName)
		if len(inp.TopArtists) >= 5 {
			break
		}
	}

	aura := computeAura(hist, nil, username)
	inp.Hue = aura.Hue
	inp.Hue2 = aura.Hue2
	inp.Energy = aura.Energy
	inp.Valence = aura.Valence
	inp.Acoustic = aura.Acoustic

	imgBytes, err := ai.GenerateBackground(inp)
	if err != nil {
		log.Printf("background: generation failed for %s: %v", username, err)
		http.Error(w, "background generation failed", http.StatusServiceUnavailable)
		return
	}

	if err := os.WriteFile(cachePath, imgBytes, 0o644); err != nil {
		log.Printf("background: failed to cache for %s: %v", username, err)
	}

	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "max-age=86400, immutable")
	_, _ = io.Copy(w, bytes.NewReader(imgBytes))
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
	case "setlist":
		args = append(args, "--setlist")
		if a := r.FormValue("setlist_artist"); a != "" {
			args = append(args, "--setlist-artist", a)
		}
		if v := r.FormValue("setlist_pages"); v != "" {
			args = append(args, "--setlist-pages", v)
		}
	case "expand":
		args = append(args, "--expand")
		if yr := r.FormValue("expand_year"); yr != "" && yr != "all" {
			args = append(args, "--expand-year", yr)
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

// ── Aura ─────────────────────────────────────────────────────────────────────

type auraParams struct {
	Hue      int
	Hue2     int     // accent hue — guaranteed unique per username
	Sat      int
	Energy   float64
	Valence  float64
	Dance    float64
	Acoustic float64
	Tempo    float64
	Variety  int     // 0–3 visual layout variant — unique per username
}

// computeAura derives canvas background parameters from listening history,
// falling back to Spotify audio features when available. The username is used
// to seed a unique palette for users with no local export data.
func computeAura(hist *history.Stats, af *spotify.AudioFeaturesSummary, username string) auraParams {
	var a auraParams
	if af != nil {
		a = auraParams{
			Hue:      af.PrimaryHue,
			Sat:      af.Saturation,
			Energy:   af.Energy,
			Valence:  af.Valence,
			Dance:    af.Danceability,
			Acoustic: af.Acousticness,
			Tempo:    af.Tempo,
		}
	} else if hist.TotalMsPlayed == 0 {
		a = usernameAura(username)
	} else {
		// Peak listening hour → hue (night=250, morning=35, afternoon=130, evening=210)
		peak := 0
		for h := range hist.HourlyPattern {
			if hist.HourlyPattern[h] > hist.HourlyPattern[peak] {
				peak = h
			}
		}
		a.Hue = (260 - peak*8 + 360) % 360
		// Skip rate → energy (frequent skipper = restless/high-energy listener)
		a.Energy = clamp01(float64(hist.SkipPct)/100.0*1.4 + 0.15)
		// Completion rate → valence (finishers tend toward positive, upbeat music)
		a.Valence = clamp01(float64(hist.CompletionPct)/100.0*1.2 + 0.1)
		// Library breadth → saturation (wider taste = more vivid palette)
		diversity := clamp01(float64(hist.UniqueArtistCount) / 400.0)
		a.Sat = 35 + int(diversity*45)
		// Danceability proxy: high completion + midday peak = more rhythmic
		a.Dance = clamp01((a.Valence + (1 - float64(peak)/24)) / 2)
		// Acousticness: patient listeners (low skip) lean acoustic
		a.Acoustic = clamp01(1.0 - a.Energy*0.8)
		// Tempo: peak hour maps to BPM range (late night=slow, daytime=fast)
		a.Tempo = 70 + float64(peak)/24*80
	}
	// Inject accent hue and layout variant from username in every code path.
	// This guarantees visual distinctiveness even when two users have identical listening data.
	uh := usernameHash(username)
	a.Hue2 = (a.Hue + 90 + int(uh%181)) % 360 // 90–270° away from primary
	a.Variety = int((uh >> 16) % 4)
	return a
}

// usernameHash returns a stable FNV-1a hash of the username.
func usernameHash(username string) uint32 {
	const offset, prime = uint32(2166136261), uint32(16777619)
	h := offset
	for _, c := range username {
		h ^= uint32(c)
		h *= prime
	}
	return h
}

// usernameAura produces a deterministic, visually varied aura seeded by username.
// Ranges are intentionally wide so different usernames produce clearly distinct looks.
// Hue2 and Variety are added by computeAura after this returns.
func usernameAura(username string) auraParams {
	h := usernameHash(username)
	hue := int(h % 360)                          // 0–359°, full spectrum
	energy := 0.10 + float64((h>>8)%100)/111.0   // 0.10–1.00
	valence := 0.30 + float64((h>>16)%100)/143.0 // 0.30–1.00
	sat := 62 + int((h>>24)%28)                  // 62–89 — vivid throughout
	return auraParams{
		Hue:      hue,
		Sat:      sat,
		Energy:   energy,
		Valence:  valence,
		Dance:    (valence + energy) / 2,
		Acoustic: clamp01(1.0 - energy*0.8),
		Tempo:    80 + float64((h>>4)%100), // 80–180 BPM
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
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
	protected.HandleFunc("GET /profile", handleProfile)
	protected.HandleFunc("GET /profile/bg", handleProfileBg)
	protected.HandleFunc("GET /history", handleHistoryGet)
	protected.HandleFunc("POST /history", handleHistorySave)
	protected.HandleFunc("DELETE /history", handleHistoryClear)

	mux.Handle("/", auth.RequireAuth(protected))

	port := envOr("PORT", "8000")
	log.Printf("spotAIfy web — listening on :%s", port)
	if err := http.ListenAndServe(":"+port, requestLogger(securityHeaders(mux))); err != nil {
		log.Fatal(err)
	}
}

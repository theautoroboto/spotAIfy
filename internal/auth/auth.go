package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/bcrypt"
)

var (
	Users         map[string]string
	sessionSecret []byte
	dummyHash     []byte
)

func init() {
	// Pre-compute a dummy hash so unknown-username lookups take the same
	// time as valid ones, preventing username enumeration via timing.
	dummyHash, _ = bcrypt.GenerateFromPassword([]byte("dummy-timing-defense"), bcrypt.DefaultCost)
}

// ── Rate limiter ──────────────────────────────────────────────────────────────

const (
	maxLoginAttempts = 10
	loginWindow      = 15 * time.Minute
)

type ipBucket struct {
	mu      sync.Mutex
	count   int
	resetAt time.Time
}

var (
	rateMu  sync.Mutex
	buckets = make(map[string]*ipBucket)
)

// LoginAllowed returns true if the IP is within the rate limit and increments
// the counter. Call before checking credentials.
func LoginAllowed(ip string) bool {
	rateMu.Lock()
	b, ok := buckets[ip]
	if !ok {
		b = &ipBucket{}
		buckets[ip] = b
	}
	rateMu.Unlock()

	b.mu.Lock()
	defer b.mu.Unlock()
	now := time.Now()
	if now.After(b.resetAt) {
		b.count = 0
		b.resetAt = now.Add(loginWindow)
	}
	b.count++
	return b.count <= maxLoginAttempts
}

// ── Auth ──────────────────────────────────────────────────────────────────────

func ParseUsers(usersEnv string) map[string]string {
	out := make(map[string]string)
	for _, pair := range strings.Split(usersEnv, ",") {
		pair = strings.TrimSpace(pair)
		idx := strings.Index(pair, ":")
		if idx < 1 {
			continue
		}
		out[pair[:idx]] = pair[idx+1:]
	}
	return out
}

func SetSecret(secret string) { sessionSecret = []byte(secret) }

func CheckPassword(username, password string) bool {
	hash, ok := Users[username]
	if !ok {
		// Always run bcrypt to prevent username enumeration via timing.
		bcrypt.CompareHashAndPassword(dummyHash, []byte(password))
		return false
	}
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

func sign(value string) string {
	mac := hmac.New(sha256.New, sessionSecret)
	mac.Write([]byte(value))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func SetSession(w http.ResponseWriter, username string) {
	encoded := base64.RawURLEncoding.EncodeToString([]byte(username))
	value := encoded + "." + sign(encoded)
	http.SetCookie(w, &http.Cookie{
		Name:     "session",
		Value:    value,
		Path:     "/",
		HttpOnly: true,
		Secure:   true,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   86400 * 7,
	})
}

func GetSession(r *http.Request) (string, bool) {
	cookie, err := r.Cookie("session")
	if err != nil {
		return "", false
	}
	idx := strings.LastIndex(cookie.Value, ".")
	if idx < 0 {
		return "", false
	}
	encoded, sig := cookie.Value[:idx], cookie.Value[idx+1:]
	if !hmac.Equal([]byte(sig), []byte(sign(encoded))) {
		return "", false
	}
	raw, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return "", false
	}
	return string(raw), true
}

func ClearSession(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name:   "session",
		MaxAge: -1,
		Path:   "/",
		Secure: true,
	})
}

// RequireAuth wraps a handler and redirects to /login if not authenticated.
func RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := GetSession(r); !ok {
			http.Redirect(w, r, "/login?next="+r.URL.RequestURI(), http.StatusFound)
			return
		}
		next.ServeHTTP(w, r)
	})
}

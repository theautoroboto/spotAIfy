package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"strings"

	"golang.org/x/crypto/bcrypt"
)

var (
	Users         map[string]string // username → bcrypt hash
	sessionSecret []byte
)

// ParseUsers parses "alice:$2b$...,bob:$2b$..." from the USERS env var.
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
		SameSite: http.SameSiteDefaultMode,
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
	http.SetCookie(w, &http.Cookie{Name: "session", MaxAge: -1, Path: "/"})
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

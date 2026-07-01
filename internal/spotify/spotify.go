package spotify

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"spotaify-web/internal/db"
)

// ErrTokenExpired is returned by FreshToken when the refresh token has expired
// (Spotify invalid_grant). Callers should discard the stored token and redirect
// the user through the Spotify OAuth flow to obtain a new one.
var ErrTokenExpired = errors.New("spotify refresh token expired — please reconnect your account")

// Config is populated from env vars by main.go.
var Config struct {
	ClientID     string
	ClientSecret string
	RedirectURI  string // SPOTIFY_WEB_REDIRECT_URI
	Scopes       string
}

func init() {
	Config.Scopes = "user-library-read playlist-read-private user-read-recently-played " +
		"playlist-modify-private playlist-modify-public " +
		"user-top-read user-read-private"
}

// AuthURL returns the Spotify authorization URL for the given state token.
func AuthURL(state string) string {
	p := url.Values{}
	p.Set("client_id", Config.ClientID)
	p.Set("response_type", "code")
	p.Set("redirect_uri", Config.RedirectURI)
	p.Set("scope", Config.Scopes)
	p.Set("state", state)
	return "https://accounts.spotify.com/authorize?" + p.Encode()
}

type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	Scope        string `json:"scope"`
	ExpiresIn    int    `json:"expires_in"`
	Error        string `json:"error"`
	ErrorDesc    string `json:"error_description"`
}

func exchangeCode(code string) (*tokenResponse, error) {
	data := url.Values{}
	data.Set("grant_type", "authorization_code")
	data.Set("code", code)
	data.Set("redirect_uri", Config.RedirectURI)
	return postToken(data)
}

func refreshToken(refreshTok string) (*tokenResponse, error) {
	data := url.Values{}
	data.Set("grant_type", "refresh_token")
	data.Set("refresh_token", refreshTok)
	return postToken(data)
}

func postToken(data url.Values) (*tokenResponse, error) {
	req, _ := http.NewRequest("POST", "https://accounts.spotify.com/api/token",
		strings.NewReader(data.Encode()))
	req.SetBasicAuth(Config.ClientID, Config.ClientSecret)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var t tokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&t); err != nil {
		return nil, err
	}
	if t.Error == "invalid_grant" {
		return nil, ErrTokenExpired
	}
	if t.Error != "" {
		return nil, fmt.Errorf("spotify: %s — %s", t.Error, t.ErrorDesc)
	}
	return &t, nil
}

// ExchangeAndStore exchanges an authorization code and persists the token for username.
func ExchangeAndStore(username, code string) error {
	t, err := exchangeCode(code)
	if err != nil {
		return err
	}
	return db.UpsertToken(username, &db.SpotifyToken{
		AccessToken:  t.AccessToken,
		RefreshToken: t.RefreshToken,
		Scope:        t.Scope,
		ExpiresAt:    time.Now().Add(time.Duration(t.ExpiresIn) * time.Second),
	})
}

// FreshToken returns a valid access token for username, refreshing if needed.
func FreshToken(username string) (string, error) {
	tok, err := db.GetToken(username)
	if err != nil {
		return "", fmt.Errorf("no spotify token for %s — please connect your account", username)
	}
	if time.Now().Before(tok.ExpiresAt.Add(-60 * time.Second)) {
		return tok.AccessToken, nil
	}
	// Token expired — refresh.
	t, err := refreshToken(tok.RefreshToken)
	if err != nil {
		if errors.Is(err, ErrTokenExpired) {
			// Refresh token has expired (Spotify invalid_grant). Discard it so the
			// next request doesn't retry a dead token, then surface the sentinel so
			// the caller can redirect the user through the OAuth flow.
			_ = db.DeleteToken(username)
		}
		return "", err
	}
	newTok := &db.SpotifyToken{
		AccessToken:  t.AccessToken,
		Scope:        t.Scope,
		ExpiresAt:    time.Now().Add(time.Duration(t.ExpiresIn) * time.Second),
		RefreshToken: tok.RefreshToken, // Spotify may not return a new refresh token
	}
	if t.RefreshToken != "" {
		newTok.RefreshToken = t.RefreshToken
	}
	_ = db.UpsertToken(username, newTok)
	return t.AccessToken, nil
}

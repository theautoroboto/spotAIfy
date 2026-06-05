package db

import (
	"database/sql"
	"time"

	_ "modernc.org/sqlite"
)

var DB *sql.DB

func Init(path string) error {
	var err error
	DB, err = sql.Open("sqlite", path+"?_journal=WAL&_timeout=5000")
	if err != nil {
		return err
	}
	DB.SetMaxOpenConns(1) // SQLite does not support concurrent writers
	_, err = DB.Exec(`CREATE TABLE IF NOT EXISTS spotify_tokens (
		username     TEXT PRIMARY KEY,
		access_token TEXT NOT NULL,
		refresh_token TEXT NOT NULL,
		scope        TEXT NOT NULL DEFAULT '',
		expires_at   INTEGER NOT NULL
	)`)
	return err
}

type SpotifyToken struct {
	AccessToken  string
	RefreshToken string
	Scope        string
	ExpiresAt    time.Time
}

func GetToken(username string) (*SpotifyToken, error) {
	row := DB.QueryRow(
		"SELECT access_token, refresh_token, scope, expires_at FROM spotify_tokens WHERE username = ?",
		username,
	)
	var t SpotifyToken
	var expiresAt int64
	if err := row.Scan(&t.AccessToken, &t.RefreshToken, &t.Scope, &expiresAt); err != nil {
		return nil, err
	}
	t.ExpiresAt = time.Unix(expiresAt, 0)
	return &t, nil
}

func UpsertToken(username string, t *SpotifyToken) error {
	_, err := DB.Exec(`
		INSERT INTO spotify_tokens (username, access_token, refresh_token, scope, expires_at)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(username) DO UPDATE SET
			access_token  = excluded.access_token,
			refresh_token = excluded.refresh_token,
			scope         = excluded.scope,
			expires_at    = excluded.expires_at`,
		username, t.AccessToken, t.RefreshToken, t.Scope, t.ExpiresAt.Unix(),
	)
	return err
}

func HasToken(username string) bool {
	_, err := GetToken(username)
	return err == nil
}

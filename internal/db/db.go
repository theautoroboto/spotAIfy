package db

import (
	"database/sql"
	"encoding/json"
	"time"

	_ "modernc.org/sqlite"
)

var DB *sql.DB

const runsMax = 20

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
	if err != nil {
		return err
	}
	_, err = DB.Exec(`CREATE TABLE IF NOT EXISTS runs (
		id       INTEGER PRIMARY KEY AUTOINCREMENT,
		username TEXT    NOT NULL,
		ts       INTEGER NOT NULL,
		label    TEXT    NOT NULL,
		lines    TEXT    NOT NULL DEFAULT '[]'
	)`)
	if err != nil {
		return err
	}
	if _, err = DB.Exec(`CREATE INDEX IF NOT EXISTS runs_user_ts ON runs (username, ts DESC)`); err != nil {
		return err
	}
	_, err = DB.Exec(`CREATE TABLE IF NOT EXISTS personas (
		username TEXT PRIMARY KEY,
		data     TEXT NOT NULL
	)`)
	return err
}

func GetPersonaJSON(username string) (string, error) {
	var data string
	err := DB.QueryRow(`SELECT data FROM personas WHERE username = ?`, username).Scan(&data)
	return data, err
}

func SavePersonaJSON(username, data string) error {
	_, err := DB.Exec(`
		INSERT INTO personas (username, data) VALUES (?, ?)
		ON CONFLICT(username) DO UPDATE SET data = excluded.data`,
		username, data)
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

// ── Run history ───────────────────────────────────────────────────────────────

type Run struct {
	Ts    int64    `json:"ts"`    // Unix ms (ready for new Date(ts) in JS)
	Label string   `json:"label"`
	Lines []string `json:"lines"`
}

func InsertRun(username, label string, lines []string) error {
	linesJSON, err := json.Marshal(lines)
	if err != nil {
		return err
	}
	ts := time.Now().Unix()
	if _, err := DB.Exec(
		`INSERT INTO runs (username, ts, label, lines) VALUES (?, ?, ?, ?)`,
		username, ts, label, string(linesJSON),
	); err != nil {
		return err
	}
	// Prune oldest rows beyond the cap.
	_, err = DB.Exec(`
		DELETE FROM runs WHERE username = ? AND id NOT IN (
			SELECT id FROM runs WHERE username = ? ORDER BY ts DESC LIMIT ?
		)`, username, username, runsMax)
	return err
}

func GetRuns(username string) ([]Run, error) {
	rows, err := DB.Query(
		`SELECT ts, label, lines FROM runs WHERE username = ? ORDER BY ts DESC LIMIT ?`,
		username, runsMax,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var runs []Run
	for rows.Next() {
		var r Run
		var ts int64
		var linesJSON string
		if err := rows.Scan(&ts, &r.Label, &linesJSON); err != nil {
			return nil, err
		}
		r.Ts = ts * 1000 // convert seconds → ms for JS new Date()
		if err := json.Unmarshal([]byte(linesJSON), &r.Lines); err != nil {
			r.Lines = []string{}
		}
		runs = append(runs, r)
	}
	return runs, rows.Err()
}

func DeleteRuns(username string) error {
	_, err := DB.Exec(`DELETE FROM runs WHERE username = ?`, username)
	return err
}

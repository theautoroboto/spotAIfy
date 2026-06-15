package history

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func record(id, artist string, ms int64, skipped bool, done bool) map[string]any {
	reason := "fwdbtn"
	if done {
		reason = "trackdone"
	}
	return map[string]any{
		"ts":                                    "2024-03-01T12:00:00Z",
		"ms_played":                             ms,
		"master_metadata_track_name":            "Track " + id,
		"master_metadata_album_artist_name":     artist,
		"master_metadata_album_album_name":      "Album",
		"spotify_track_uri":                     "spotify:track:" + id,
		"reason_end":                            reason,
		"skipped":                               skipped,
	}
}

func writeExport(t *testing.T, dir string, records []map[string]any) {
	t.Helper()
	data, err := json.Marshal(records)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "Streaming_History_Audio_2024.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadExcludesSkippedPlaysFromRankings(t *testing.T) {
	dir := t.TempDir()
	writeExport(t, dir, []map[string]any{
		// "noise": huge listening time but every play skipped.
		record("noise", "Sleep Artist", 90_000, true, false),
		record("noise", "Sleep Artist", 90_000, true, false),
		record("noise", "Sleep Artist", 90_000, true, false),
		// "fav": modest but real listening.
		record("fav", "Real Artist", 30_000, false, true),
		record("fav", "Real Artist", 30_000, false, true),
	})

	stats, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}

	if got := stats.TopTracks[0].TrackID; got != "fav" {
		t.Errorf("TopTracks[0] = %q, want fav (skipped plays must not rank)", got)
	}
	for _, tk := range stats.TopTracks {
		if tk.TrackID == "noise" {
			if tk.MsPlayedTotal != 0 || tk.PlayCount != 0 {
				t.Errorf("noise: ms=%d plays=%d, want 0/0 — skips must not accumulate", tk.MsPlayedTotal, tk.PlayCount)
			}
			if tk.SkipCount != 3 {
				t.Errorf("noise SkipCount = %d, want 3", tk.SkipCount)
			}
		}
	}

	if got := stats.TopArtists[0].ArtistName; got != "Real Artist" {
		t.Errorf("TopArtists[0] = %q, want Real Artist", got)
	}

	// Most Skipped still sees the skips.
	if len(stats.TopSkipped) == 0 || stats.TopSkipped[0].TrackID != "noise" {
		t.Errorf("TopSkipped[0] should be noise, got %+v", stats.TopSkipped)
	}

	// Honest totals: hours listened and skip rate count every play.
	if want := int64(3*90_000 + 2*30_000); stats.TotalMsPlayed != want {
		t.Errorf("TotalMsPlayed = %d, want %d (all plays incl. skipped)", stats.TotalMsPlayed, want)
	}
	if stats.SkipPct != 60 { // 3 of 5 plays skipped
		t.Errorf("SkipPct = %d, want 60", stats.SkipPct)
	}

	// Per-year top lists exclude skips too.
	year := stats.YearlyTrend[0]
	if len(year.TopTracks) == 0 || year.TopTracks[0].TrackName != "Track fav" {
		t.Errorf("year TopTracks[0] = %+v, want Track fav", year.TopTracks)
	}
}

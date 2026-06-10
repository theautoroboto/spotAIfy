// Package history reads Spotify Extended Streaming History JSON exports
// (Streaming_History_Audio_*.json) and computes per-track, per-artist,
// and per-year listening stats for the profile page.
package history

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ── Raw record shape from the export ─────────────────────────────────────────

type rawRecord struct {
	Ts         string `json:"ts"`
	MsPlayed   int64  `json:"ms_played"`
	TrackName  string `json:"master_metadata_track_name"`
	ArtistName string `json:"master_metadata_album_artist_name"`
	AlbumName  string `json:"master_metadata_album_album_name"`
	TrackURI   string `json:"spotify_track_uri"`
	ReasonEnd  string `json:"reason_end"`
	Skipped    *bool  `json:"skipped"`
}

// ── Public types ──────────────────────────────────────────────────────────────

type TrackStats struct {
	TrackID         string
	TrackName       string
	ArtistName      string
	PlayCount       int
	MsPlayedTotal   int64
	CompletionCount int
	SkipCount       int
	LastPlayedAt    time.Time
	// Enriched by Spotify API after load:
	ImageURL   string
	SpotifyURL string
}

func (t *TrackStats) CompletionRate() float64 {
	if t.PlayCount == 0 {
		return 0
	}
	return float64(t.CompletionCount) / float64(t.PlayCount)
}

func (t *TrackStats) HoursPlayed() float64 { return float64(t.MsPlayedTotal) / 3_600_000 }

type ArtistStats struct {
	ArtistName       string
	MsPlayedTotal    int64
	PlayCount        int
	UniqueTrackCount int
	// Enriched by Spotify API after load:
	ImageURL   string
	SpotifyURL string
	Genres     []string
}

func (a *ArtistStats) HoursPlayed() float64 { return float64(a.MsPlayedTotal) / 3_600_000 }

type YearTopArtist struct {
	Name          string
	MsPlayedTotal int64
	PlayCount     int
}

func (y *YearTopArtist) HoursPlayed() float64 { return float64(y.MsPlayedTotal) / 3_600_000 }

type YearTopTrack struct {
	TrackName     string
	ArtistName    string
	MsPlayedTotal int64
	PlayCount     int
}

func (y *YearTopTrack) HoursPlayed() float64 { return float64(y.MsPlayedTotal) / 3_600_000 }

type YearStats struct {
	Year       int
	PlayCount  int
	MsPlayed   int64
	BarPct     int             // 0–100, normalized to busiest year
	TopArtists []YearTopArtist // top 5 by ms played
	TopTracks  []YearTopTrack  // top 5 by ms played
}

func (y *YearStats) HoursPlayed() float64 { return float64(y.MsPlayed) / 3_600_000 }

type Stats struct {
	TotalMsPlayed     int64
	UniqueTrackCount  int
	UniqueArtistCount int
	SkipPct           int // 0–100, percentage of plays that were skipped
	CompletionPct     int // 0–100, percentage of plays that reached trackdone
	EarliestPlay      time.Time
	LatestPlay        time.Time
	TopTracks         []*TrackStats  // sorted by MsPlayedTotal desc, capped at 50
	TopSkipped        []*TrackStats  // sorted by SkipCount desc, capped at 10
	TopArtists        []*ArtistStats // sorted by MsPlayedTotal desc, capped at 20
	HourlyPattern     [24]int        // play events by hour-of-day (local timezone)
	YearlyTrend       []*YearStats   // sorted by year asc
}

func (s *Stats) TotalHours() float64 { return float64(s.TotalMsPlayed) / 3_600_000 }

// ── Loader ────────────────────────────────────────────────────────────────────

// Load reads all Streaming_History_Audio_*.json files from dir and returns
// aggregated stats. Returns an empty Stats (not an error) when no files exist.
func Load(dir string) (*Stats, error) {
	pattern := filepath.Join(dir, "Streaming_History_Audio_*.json")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}

	trackMap  := map[string]*TrackStats{}
	artistMap := map[string]*ArtistStats{}
	yearMap   := map[int]*YearStats{}
	var hourly [24]int
	var earliest, latest time.Time

	// Per-year ms and play-count accumulators keyed by artist name / track ID.
	yearArtistMs := map[int]map[string]int64{}
	yearArtistCt := map[int]map[string]int{}
	yearTrackMs  := map[int]map[string]int64{}
	yearTrackCt  := map[int]map[string]int{}

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var records []rawRecord
		if err := json.Unmarshal(data, &records); err != nil {
			continue
		}
		for _, r := range records {
			// Skip podcasts, videos, very short plays (<1 s).
			if r.TrackURI == "" || !strings.HasPrefix(r.TrackURI, "spotify:track:") {
				continue
			}
			if r.MsPlayed < 1000 {
				continue
			}
			trackID := strings.TrimPrefix(r.TrackURI, "spotify:track:")
			ts, _ := time.Parse(time.RFC3339, r.Ts)

			// Track stats
			tk := trackMap[trackID]
			if tk == nil {
				tk = &TrackStats{
					TrackID:    trackID,
					TrackName:  r.TrackName,
					ArtistName: r.ArtistName,
				}
				trackMap[trackID] = tk
			}
			tk.PlayCount++
			tk.MsPlayedTotal += r.MsPlayed
			if r.ReasonEnd == "trackdone" {
				tk.CompletionCount++
			}
			if r.Skipped != nil && *r.Skipped {
				tk.SkipCount++
			}
			if ts.After(tk.LastPlayedAt) {
				tk.LastPlayedAt = ts
			}

			// Artist stats
			artist := r.ArtistName
			as := artistMap[artist]
			if as == nil {
				as = &ArtistStats{ArtistName: artist}
				artistMap[artist] = as
			}
			as.MsPlayedTotal += r.MsPlayed
			as.PlayCount++

			// Global timeline
			if !ts.IsZero() {
				if earliest.IsZero() || ts.Before(earliest) {
					earliest = ts
				}
				if ts.After(latest) {
					latest = ts
				}
				hourly[ts.Local().Hour()]++

				yr := ts.Year()
				ys := yearMap[yr]
				if ys == nil {
					ys = &YearStats{Year: yr}
					yearMap[yr] = ys
				}
				ys.PlayCount++
				ys.MsPlayed += r.MsPlayed

				// Per-year accumulators
				if yearArtistMs[yr] == nil {
					yearArtistMs[yr] = map[string]int64{}
					yearArtistCt[yr] = map[string]int{}
					yearTrackMs[yr]  = map[string]int64{}
					yearTrackCt[yr]  = map[string]int{}
				}
				yearArtistMs[yr][artist] += r.MsPlayed
				yearArtistCt[yr][artist]++
				yearTrackMs[yr][trackID] += r.MsPlayed
				yearTrackCt[yr][trackID]++
			}
		}
	}

	// Count unique tracks per artist
	for _, tk := range trackMap {
		if as, ok := artistMap[tk.ArtistName]; ok {
			as.UniqueTrackCount++
		}
	}

	// Total ms played + overall skip rate (across all tracks before capping)
	var totalMs int64
	var totalPlays, totalSkips, totalCompletions int
	for _, tk := range trackMap {
		totalMs += tk.MsPlayedTotal
		totalPlays += tk.PlayCount
		totalSkips += tk.SkipCount
		totalCompletions += tk.CompletionCount
	}
	skipPct, completionPct := 0, 0
	if totalPlays > 0 {
		skipPct = int(float64(totalSkips) / float64(totalPlays) * 100)
		completionPct = int(float64(totalCompletions) / float64(totalPlays) * 100)
	}

	// Sort and cap top tracks (50)
	allTracks := make([]*TrackStats, 0, len(trackMap))
	for _, t := range trackMap {
		allTracks = append(allTracks, t)
	}
	sort.Slice(allTracks, func(i, j int) bool {
		return allTracks[i].MsPlayedTotal > allTracks[j].MsPlayedTotal
	})
	if len(allTracks) > 50 {
		allTracks = allTracks[:50]
	}

	// Sort and cap most-skipped tracks (10)
	skipped := make([]*TrackStats, 0, len(trackMap))
	for _, t := range trackMap {
		if t.SkipCount > 0 {
			skipped = append(skipped, t)
		}
	}
	sort.Slice(skipped, func(i, j int) bool {
		if skipped[i].SkipCount != skipped[j].SkipCount {
			return skipped[i].SkipCount > skipped[j].SkipCount
		}
		return skipped[i].MsPlayedTotal > skipped[j].MsPlayedTotal
	})
	if len(skipped) > 10 {
		skipped = skipped[:10]
	}

	// Sort and cap top artists (20)
	allArtists := make([]*ArtistStats, 0, len(artistMap))
	for _, a := range artistMap {
		allArtists = append(allArtists, a)
	}
	sort.Slice(allArtists, func(i, j int) bool {
		return allArtists[i].MsPlayedTotal > allArtists[j].MsPlayedTotal
	})
	if len(allArtists) > 20 {
		allArtists = allArtists[:20]
	}

	// Find busiest year for BarPct normalization
	var maxYearMs int64
	for _, ys := range yearMap {
		if ys.MsPlayed > maxYearMs {
			maxYearMs = ys.MsPlayed
		}
	}

	// Compute per-year top artists and tracks, and BarPct
	type msEntry struct {
		key string
		ms  int64
		ct  int
	}
	for yr, ys := range yearMap {
		if maxYearMs > 0 {
			ys.BarPct = int(ys.MsPlayed * 100 / maxYearMs)
		}

		// Top 5 artists this year
		artistEntries := make([]msEntry, 0, len(yearArtistMs[yr]))
		for k, ms := range yearArtistMs[yr] {
			artistEntries = append(artistEntries, msEntry{k, ms, yearArtistCt[yr][k]})
		}
		sort.Slice(artistEntries, func(i, j int) bool { return artistEntries[i].ms > artistEntries[j].ms })
		for i, e := range artistEntries {
			if i >= 5 {
				break
			}
			ys.TopArtists = append(ys.TopArtists, YearTopArtist{
				Name:          e.key,
				MsPlayedTotal: e.ms,
				PlayCount:     e.ct,
			})
		}

		// Top 5 tracks this year
		trackEntries := make([]msEntry, 0, len(yearTrackMs[yr]))
		for k, ms := range yearTrackMs[yr] {
			trackEntries = append(trackEntries, msEntry{k, ms, yearTrackCt[yr][k]})
		}
		sort.Slice(trackEntries, func(i, j int) bool { return trackEntries[i].ms > trackEntries[j].ms })
		for i, e := range trackEntries {
			if i >= 5 {
				break
			}
			name, artistName := e.key, ""
			if tk := trackMap[e.key]; tk != nil {
				name = tk.TrackName
				artistName = tk.ArtistName
			}
			ys.TopTracks = append(ys.TopTracks, YearTopTrack{
				TrackName:     name,
				ArtistName:    artistName,
				MsPlayedTotal: e.ms,
				PlayCount:     e.ct,
			})
		}
	}

	// Yearly trend sorted ascending
	allYears := make([]*YearStats, 0, len(yearMap))
	for _, y := range yearMap {
		allYears = append(allYears, y)
	}
	sort.Slice(allYears, func(i, j int) bool {
		return allYears[i].Year < allYears[j].Year
	})

	return &Stats{
		TotalMsPlayed:     totalMs,
		UniqueTrackCount:  len(trackMap),
		UniqueArtistCount: len(artistMap),
		SkipPct:           skipPct,
		CompletionPct:     completionPct,
		EarliestPlay:      earliest,
		LatestPlay:        latest,
		TopTracks:         allTracks,
		TopSkipped:        skipped,
		TopArtists:        allArtists,
		HourlyPattern:     hourly,
		YearlyTrend:       allYears,
	}, nil
}

// Package history reads Spotify Extended Streaming History JSON exports
// (Streaming_History_Audio_*.json) and computes per-track and per-artist
// listening stats for the profile page.
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
	Ts          string  `json:"ts"`
	MsPlayed    int64   `json:"ms_played"`
	TrackName   string  `json:"master_metadata_track_name"`
	ArtistName  string  `json:"master_metadata_album_artist_name"`
	AlbumName   string  `json:"master_metadata_album_album_name"`
	TrackURI    string  `json:"spotify_track_uri"`
	ReasonEnd   string  `json:"reason_end"`
	Skipped     *bool   `json:"skipped"`
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

func (t *TrackStats) HoursPlayed() float64 {
	return float64(t.MsPlayedTotal) / 3_600_000
}

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

func (a *ArtistStats) HoursPlayed() float64 {
	return float64(a.MsPlayedTotal) / 3_600_000
}

type YearStats struct {
	Year        int
	PlayCount   int
	MsPlayed    int64
}

func (y *YearStats) HoursPlayed() float64 {
	return float64(y.MsPlayed) / 3_600_000
}

type Stats struct {
	TotalMsPlayed     int64
	UniqueTrackCount  int
	UniqueArtistCount int
	EarliestPlay      time.Time
	LatestPlay        time.Time
	TopTracks         []*TrackStats  // sorted by MsPlayedTotal desc, capped at 50
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

	trackMap := map[string]*TrackStats{}
	artistMap := map[string]*ArtistStats{}
	yearMap := map[int]*YearStats{}
	var hourly [24]int
	var earliest, latest time.Time

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
			ts_ := trackMap[trackID]
			if ts_ == nil {
				ts_ = &TrackStats{
					TrackID:    trackID,
					TrackName:  r.TrackName,
					ArtistName: r.ArtistName,
				}
				trackMap[trackID] = ts_
			}
			ts_.PlayCount++
			ts_.MsPlayedTotal += r.MsPlayed
			if r.ReasonEnd == "trackdone" {
				ts_.CompletionCount++
			}
			if r.Skipped != nil && *r.Skipped {
				ts_.SkipCount++
			}
			if ts.After(ts_.LastPlayedAt) {
				ts_.LastPlayedAt = ts
			}

			// Artist stats
			artist := r.ArtistName
			as := artistMap[artist]
			if as == nil {
				as = &ArtistStats{ArtistName: artist}
				artistMap[artist] = as
				as.UniqueTrackCount = 0
			}
			as.MsPlayedTotal += r.MsPlayed
			as.PlayCount++
			// Count unique tracks per artist (approximated by checking trackMap later)

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
			}
		}
	}

	// Count unique tracks per artist
	for _, ts := range trackMap {
		if as, ok := artistMap[ts.ArtistName]; ok {
			as.UniqueTrackCount++
		}
	}

	// Total ms played
	var totalMs int64
	for _, ts := range trackMap {
		totalMs += ts.MsPlayedTotal
	}

	// Sort and cap top tracks
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

	// Sort and cap top artists
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
		EarliestPlay:      earliest,
		LatestPlay:        latest,
		TopTracks:         allTracks,
		TopArtists:        allArtists,
		HourlyPattern:     hourly,
		YearlyTrend:       allYears,
	}, nil
}

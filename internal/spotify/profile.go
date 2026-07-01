package spotify

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// ── Public data types ─────────────────────────────────────────────────────────

type UserInfo struct {
	DisplayName string
	ImageURL    string
	Followers   int
	SpotifyURL  string
}

type TopArtist struct {
	Name       string
	Genres     []string
	ImageURL   string
	Popularity int
	SpotifyURL string
}

type TopTrack struct {
	Title       string
	Artist      string
	Album       string
	ImageURL    string
	SpotifyURL  string
	Popularity  int
	ReleaseYear int
}

type RecentTrack struct {
	Title      string
	Artist     string
	ImageURL   string
	SpotifyURL string
	PlayedAt   time.Time
}

type GenreCount struct {
	Genre string
	Count int
}

type EraCount struct {
	Era string
	Pct int
}

type TasteInsights struct {
	MainstreamScore int
	ObscurityLabel  string
	TopGenres       []GenreCount
	EraBreakdown    []EraCount
}

type ProfileData struct {
	User          UserInfo
	TopArtists    map[string][]TopArtist // "short" | "medium" | "long"
	TopTracks     map[string][]TopTrack  // "short" | "medium" | "long"
	Recent        []RecentTrack
	Insights      TasteInsights
	MissingScopes bool
}

// ── Internal Spotify response shapes ─────────────────────────────────────────

type spImage struct {
	URL string `json:"url"`
}

type spArtistItem struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Genres       []string          `json:"genres"`
	Images       []spImage         `json:"images"`
	Popularity   int               `json:"popularity"`
	ExternalURLs map[string]string `json:"external_urls"`
}

type spTrackItem struct {
	Name    string `json:"name"`
	Artists []struct {
		Name string `json:"name"`
	} `json:"artists"`
	Album struct {
		Name        string    `json:"name"`
		Images      []spImage `json:"images"`
		ReleaseDate string    `json:"release_date"`
	} `json:"album"`
	Popularity   int               `json:"popularity"`
	ExternalURLs map[string]string `json:"external_urls"`
}

// ── Low-level helpers ─────────────────────────────────────────────────────────

func spotifyGet(token, path string, out any) error {
	req, _ := http.NewRequest("GET", "https://api.spotify.com/v1"+path, nil)
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("scope_missing:%d", resp.StatusCode)
	}
	if resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("spotify %s: HTTP 403", path)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("spotify %s: HTTP %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func bestImage(imgs []spImage) string {
	if len(imgs) == 0 {
		return ""
	}
	return imgs[0].URL
}

func releaseYear(date string) int {
	if len(date) < 4 {
		return 0
	}
	y := 0
	for _, c := range date[:4] {
		if c < '0' || c > '9' {
			return 0
		}
		y = y*10 + int(c-'0')
	}
	return y
}

// ── Main fetch ────────────────────────────────────────────────────────────────

func FetchProfileData(token string) (*ProfileData, error) {
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		pd      ProfileData
	)
	pd.TopArtists = make(map[string][]TopArtist)
	pd.TopTracks = make(map[string][]TopTrack)

	// User profile
	wg.Add(1)
	go func() {
		defer wg.Done()
		var me struct {
			DisplayName  string            `json:"display_name"`
			Images       []spImage         `json:"images"`
			Followers    struct{ Total int } `json:"followers"`
			ExternalURLs map[string]string `json:"external_urls"`
		}
		if err := spotifyGet(token, "/me", &me); err != nil {
			return
		}
		mu.Lock()
		pd.User = UserInfo{
			DisplayName: me.DisplayName,
			ImageURL:    bestImage(me.Images),
			Followers:   me.Followers.Total,
			SpotifyURL:  me.ExternalURLs["spotify"],
		}
		mu.Unlock()
	}()

	// Top artists & tracks — 3 time ranges each
	timeRanges := [3][2]string{
		{"short_term", "short"},
		{"medium_term", "medium"},
		{"long_term", "long"},
	}
	for _, tr := range timeRanges {
		apiRange, key := tr[0], tr[1]

		wg.Add(1)
		go func() {
			defer wg.Done()
			var resp struct {
				Items []spArtistItem `json:"items"`
			}
			err := spotifyGet(token, "/me/top/artists?time_range="+apiRange+"&limit=10", &resp)
			if err != nil {
				if strings.HasPrefix(err.Error(), "scope_missing") {
					mu.Lock()
					pd.MissingScopes = true
					mu.Unlock()
				}
				return
			}
			artists := make([]TopArtist, 0, len(resp.Items))
			for _, a := range resp.Items {
				artists = append(artists, TopArtist{
					Name:       a.Name,
					Genres:     a.Genres,
					ImageURL:   bestImage(a.Images),
					Popularity: a.Popularity,
					SpotifyURL: a.ExternalURLs["spotify"],
				})
			}
			mu.Lock()
			pd.TopArtists[key] = artists
			mu.Unlock()
		}()

		wg.Add(1)
		go func() {
			defer wg.Done()
			var resp struct {
				Items []spTrackItem `json:"items"`
			}
			if err := spotifyGet(token, "/me/top/tracks?time_range="+apiRange+"&limit=10", &resp); err != nil {
				return
			}
			tracks := make([]TopTrack, 0, len(resp.Items))
			for _, t := range resp.Items {
				artist := ""
				if len(t.Artists) > 0 {
					artist = t.Artists[0].Name
				}
				tracks = append(tracks, TopTrack{
					Title:       t.Name,
					Artist:      artist,
					Album:       t.Album.Name,
					ImageURL:    bestImage(t.Album.Images),
					SpotifyURL:  t.ExternalURLs["spotify"],
					Popularity:  t.Popularity,
					ReleaseYear: releaseYear(t.Album.ReleaseDate),
				})
			}
			mu.Lock()
			pd.TopTracks[key] = tracks
			mu.Unlock()
		}()
	}

	// Recently played
	wg.Add(1)
	go func() {
		defer wg.Done()
		var resp struct {
			Items []struct {
				Track    spTrackItem `json:"track"`
				PlayedAt time.Time   `json:"played_at"`
			} `json:"items"`
		}
		if err := spotifyGet(token, "/me/player/recently-played?limit=20", &resp); err != nil {
			return
		}
		recent := make([]RecentTrack, 0, len(resp.Items))
		for _, item := range resp.Items {
			t := item.Track
			artist := ""
			if len(t.Artists) > 0 {
				artist = t.Artists[0].Name
			}
			recent = append(recent, RecentTrack{
				Title:      t.Name,
				Artist:     artist,
				ImageURL:   bestImage(t.Album.Images),
				SpotifyURL: t.ExternalURLs["spotify"],
				PlayedAt:   item.PlayedAt,
			})
		}
		mu.Lock()
		pd.Recent = recent
		mu.Unlock()
	}()

	wg.Wait()
	pd.Insights = deriveInsights(pd.TopArtists, pd.TopTracks)
	return &pd, nil
}

// ── Audio features ────────────────────────────────────────────────────────────

type spAudioFeature struct {
	ID               string  `json:"id"`
	Danceability     float64 `json:"danceability"`
	Energy           float64 `json:"energy"`
	Speechiness      float64 `json:"speechiness"`
	Acousticness     float64 `json:"acousticness"`
	Instrumentalness float64 `json:"instrumentalness"`
	Liveness         float64 `json:"liveness"`
	Valence          float64 `json:"valence"`
	Tempo            float64 `json:"tempo"`
}

// AudioFeaturesSummary holds averaged audio attributes across the user's top tracks.
type AudioFeaturesSummary struct {
	Danceability     float64
	Energy           float64
	Valence          float64
	Acousticness     float64
	Instrumentalness float64
	Liveness         float64
	Tempo            float64
	TrackCount       int
	VibeLabel        string
	VibeEmoji        string
	VibeDesc         string
	PrimaryHue       int // 0–360 — drives canvas colour palette
	Saturation       int // 0–100
}

// FetchAudioFeaturesSummary fetches Spotify audio features for up to 100 track IDs
// and returns averaged attributes. Returns nil on any failure.
func FetchAudioFeaturesSummary(token string, trackIDs []string) *AudioFeaturesSummary {
	if len(trackIDs) == 0 {
		return nil
	}
	ids := trackIDs
	if len(ids) > 100 {
		ids = ids[:100]
	}

	var all []spAudioFeature
	for i := 0; i < len(ids); i += 100 {
		batch := ids[i:]
		if len(batch) > 100 {
			batch = batch[:100]
		}
		var resp struct {
			AudioFeatures []spAudioFeature `json:"audio_features"`
		}
		if err := spotifyGet(token, "/audio-features?ids="+strings.Join(batch, ","), &resp); err != nil {
			log.Printf("FetchAudioFeaturesSummary: batch %d failed: %v", i, err)
			continue
		}
		log.Printf("FetchAudioFeaturesSummary: batch %d got %d features", i, len(resp.AudioFeatures))
		all = append(all, resp.AudioFeatures...)
	}

	var (
		sumD, sumE, sumV, sumA, sumI, sumL, sumT float64
		n                                         float64
	)
	for _, f := range all {
		if f.ID == "" {
			continue
		}
		sumD += f.Danceability
		sumE += f.Energy
		sumV += f.Valence
		sumA += f.Acousticness
		sumI += f.Instrumentalness
		sumL += f.Liveness
		sumT += f.Tempo
		n++
	}
	if n == 0 {
		return nil
	}

	s := &AudioFeaturesSummary{
		Danceability:     sumD / n,
		Energy:           sumE / n,
		Valence:          sumV / n,
		Acousticness:     sumA / n,
		Instrumentalness: sumI / n,
		Liveness:         sumL / n,
		Tempo:            sumT / n,
		TrackCount:       int(n),
	}
	// Hue mapping: valence 0 → 230 (cool blue/indigo), valence 1 → 30 (warm amber)
	s.PrimaryHue = int(230 - s.Valence*200)
	s.Saturation = int(35 + s.Energy*60)
	s.VibeLabel, s.VibeEmoji, s.VibeDesc = computeVibe(s)
	return s
}

func computeVibe(s *AudioFeaturesSummary) (label, emoji, desc string) {
	e, v, d, a, in := s.Energy, s.Valence, s.Danceability, s.Acousticness, s.Instrumentalness
	switch {
	case in > 0.55:
		return "Instrumental Dreamer", "🎹", "Pure sound, no words needed — music that speaks in texture and feeling"
	case e > 0.75 && v > 0.65 && d > 0.65:
		return "Dance Floor Legend", "🕺", "High-energy, euphoric, built to move — the life of every party"
	case e > 0.75 && v < 0.35:
		return "Dark Energy", "⚡", "Intense and raw — music that hits hard with a brooding emotional core"
	case e > 0.70 && d > 0.65:
		return "Relentless Groover", "🔥", "High-tempo, off-the-charts danceability — you simply don't stop moving"
	case e > 0.65 && v >= 0.35 && v <= 0.65:
		return "Driven Explorer", "🚀", "Powerful and focused — music that fuels ambition and momentum"
	case a > 0.65 && e < 0.45:
		return "Acoustic Soul", "🎸", "Intimate and organic — raw sounds stripped to their emotional core"
	case e < 0.35 && v > 0.55:
		return "Zen Wanderer", "🌅", "Calm and luminous — music that soothes without losing warmth"
	case e < 0.40 && v < 0.35:
		return "Midnight Poet", "🌙", "Deep, reflective, beautifully melancholic — music for quiet hours"
	case e < 0.50 && v < 0.45 && a > 0.40:
		return "Introspective", "🌧", "Quiet intensity — acoustic textures carrying real emotional weight"
	case d > 0.70:
		return "Groove Architect", "🎧", "Rhythmically precise and endlessly listenable — built for the groove"
	case v > 0.65:
		return "Eternal Optimist", "☀️", "Consistently warm and uplifting — music as a source of pure joy"
	default:
		return "Eclectic Mind", "🎶", "A rich, balanced palette — curious, open, and impossible to pin down"
	}
}

// ── User playlists ────────────────────────────────────────────────────────────

type UserPlaylist struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	TrackCount int    `json:"track_count"`
	ImageURL   string `json:"image_url"`
	Public     bool   `json:"public"`
}

// FetchUserPlaylists returns all playlists owned or followed by the user,
// paging through the Spotify API until exhausted.
func FetchUserPlaylists(token string) ([]UserPlaylist, error) {
	const base = "https://api.spotify.com/v1"
	var all []UserPlaylist
	path := "/me/playlists?limit=50"
	for path != "" {
		var resp struct {
			Items []struct {
				ID     string    `json:"id"`
				Name   string    `json:"name"`
				Images []spImage `json:"images"`
				Tracks struct {
					Total int `json:"total"`
				} `json:"tracks"`
				Public *bool `json:"public"`
			} `json:"items"`
			Next string `json:"next"`
		}
		if err := spotifyGet(token, path, &resp); err != nil {
			return nil, err
		}
		for i, item := range resp.Items {
			if item.ID == "" || item.Name == "" {
				continue
			}
			pub := false
			if item.Public != nil {
				pub = *item.Public
			}
			if len(all) == 0 && i < 3 {
				log.Printf("[FetchUserPlaylists] item[%d]: id=%q name=%q tracks_total=%d", i, item.ID, item.Name, item.Tracks.Total)
			}
			all = append(all, UserPlaylist{
				ID:         item.ID,
				Name:       item.Name,
				TrackCount: item.Tracks.Total,
				ImageURL:   bestImage(item.Images),
				Public:     pub,
			})
		}
		if resp.Next == "" {
			break
		}
		if strings.HasPrefix(resp.Next, base) {
			path = resp.Next[len(base):]
		} else {
			break
		}
	}
	return all, nil
}

// ── Playlist preview ──────────────────────────────────────────────────────────

type PreviewTrack struct {
	Title    string `json:"title"`
	Artist   string `json:"artist"`
	ImageURL string `json:"image_url"`
}

// FetchPlaylistPreview returns up to limit tracks from a playlist, with title,
// artist, and album art — enough to render a compact preview list.
func FetchPlaylistPreview(token, playlistID string, limit int) ([]PreviewTrack, error) {
	if limit <= 0 || limit > 20 {
		limit = 20
	}
	var resp struct {
		Items []struct {
			Track *spTrackItem `json:"track"`
		} `json:"items"`
	}
	path := fmt.Sprintf("/playlists/%s/tracks?limit=%d", playlistID, limit)
	if err := spotifyGet(token, path, &resp); err != nil {
		return nil, err
	}
	var tracks []PreviewTrack
	for _, item := range resp.Items {
		if item.Track == nil || item.Track.Name == "" {
			continue
		}
		artist := ""
		if len(item.Track.Artists) > 0 {
			artist = item.Track.Artists[0].Name
		}
		tracks = append(tracks, PreviewTrack{
			Title:    item.Track.Name,
			Artist:   artist,
			ImageURL: bestImage(item.Track.Album.Images),
		})
	}
	return tracks, nil
}

// ── Metadata enrichment ───────────────────────────────────────────────────────

type TrackMeta struct {
	ImageURL   string
	SpotifyURL string
	ArtistID   string // first artist's ID — used to batch-fetch artist images
}

// FetchTracksMeta fetches track metadata for up to 50 IDs per call and returns
// a map keyed by track ID. IDs beyond 50 are silently ignored.
func FetchTracksMeta(token string, trackIDs []string) map[string]TrackMeta {
	result := map[string]TrackMeta{}
	for i := 0; i < len(trackIDs); i += 50 {
		batch := trackIDs[i:]
		if len(batch) > 50 {
			batch = batch[:50]
		}
		var resp struct {
			Tracks []struct {
				ID      string `json:"id"`
				Artists []struct {
					ID string `json:"id"`
				} `json:"artists"`
				Album        struct{ Images []spImage } `json:"album"`
				ExternalURLs map[string]string          `json:"external_urls"`
			} `json:"tracks"`
		}
		ids := ""
		for j, id := range batch {
			if j > 0 {
				ids += ","
			}
			ids += id
		}
		if err := spotifyGet(token, "/tracks?ids="+ids, &resp); err != nil {
			continue
		}
		for _, t := range resp.Tracks {
			if t.ID == "" {
				continue
			}
			artistID := ""
			if len(t.Artists) > 0 {
				artistID = t.Artists[0].ID
			}
			result[t.ID] = TrackMeta{
				ImageURL:   bestImage(t.Album.Images),
				SpotifyURL: t.ExternalURLs["spotify"],
				ArtistID:   artistID,
			}
		}
	}
	return result
}

type ArtistMeta struct {
	ImageURL   string
	SpotifyURL string
	Genres     []string
}

// FetchArtistsMeta fetches artist metadata for up to 50 IDs per call.
func FetchArtistsMeta(token string, artistIDs []string) map[string]ArtistMeta {
	result := map[string]ArtistMeta{}
	for i := 0; i < len(artistIDs); i += 50 {
		batch := artistIDs[i:]
		if len(batch) > 50 {
			batch = batch[:50]
		}
		var resp struct {
			Artists []spArtistItem `json:"artists"`
		}
		ids := ""
		for j, id := range batch {
			if j > 0 {
				ids += ","
			}
			ids += id
		}
		if err := spotifyGet(token, "/artists?ids="+ids, &resp); err != nil {
			continue
		}
		for _, a := range resp.Artists {
			if a.ID == "" {
				continue
			}
			result[a.ID] = ArtistMeta{
				ImageURL:   bestImage(a.Images),
				SpotifyURL: a.ExternalURLs["spotify"],
				Genres:     a.Genres,
			}
		}
	}
	return result
}

// ── Taste insights ────────────────────────────────────────────────────────────

func deriveInsights(artists map[string][]TopArtist, tracks map[string][]TopTrack) TasteInsights {
	longArtists := artists["long"]
	longTracks := tracks["long"]

	// Mainstream score: average popularity of long-term top tracks.
	score := 0
	if len(longTracks) > 0 {
		sum := 0
		for _, t := range longTracks {
			sum += t.Popularity
		}
		score = sum / len(longTracks)
	}
	label := "Underground"
	switch {
	case score >= 75:
		label = "Chart-Topper"
	case score >= 55:
		label = "Mainstream"
	case score >= 35:
		label = "Indie"
	}

	// Genre frequencies from long-term top artists.
	genreFreq := map[string]int{}
	for _, a := range longArtists {
		for _, g := range a.Genres {
			genreFreq[g]++
		}
	}
	type kv struct{ g string; n int }
	var glist []kv
	for g, n := range genreFreq {
		glist = append(glist, kv{g, n})
	}
	sort.Slice(glist, func(i, j int) bool { return glist[i].n > glist[j].n })
	topGenres := make([]GenreCount, 0, 8)
	for i, g := range glist {
		if i >= 8 {
			break
		}
		topGenres = append(topGenres, GenreCount{Genre: g.g, Count: g.n})
	}

	// Era breakdown from long-term top tracks.
	eraFreq := map[string]int{}
	total := 0
	for _, t := range longTracks {
		if t.ReleaseYear == 0 {
			continue
		}
		era := fmt.Sprintf("%ds", (t.ReleaseYear/10)*10)
		eraFreq[era]++
		total++
	}
	var eras []EraCount
	for era, n := range eraFreq {
		pct := 0
		if total > 0 {
			pct = (n * 100) / total
		}
		eras = append(eras, EraCount{Era: era, Pct: pct})
	}
	sort.Slice(eras, func(i, j int) bool { return eras[i].Era < eras[j].Era })

	return TasteInsights{
		MainstreamScore: score,
		ObscurityLabel:  label,
		TopGenres:       topGenres,
		EraBreakdown:    eras,
	}
}

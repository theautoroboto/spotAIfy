package spotify

import (
	"fmt"
	"strings"
)

// Recommendation is an artist the user hasn't heard, matched to their taste.
type Recommendation struct {
	ArtistName  string
	ArtistID    string
	ImageURL    string
	SpotifyURL  string
	Genres      []string
	SampleTrack string // representative track name
	TrackURL    string // Spotify link for the sample track
}

// FetchRecommendations returns up to 10 undiscovered artists matched to the
// user's audio feature profile. knownArtists is the set of artist names
// already in their listening history.
func FetchRecommendations(token string, seedArtistIDs []string, af *AudioFeaturesSummary, knownArtists map[string]struct{}) []Recommendation {
	if len(seedArtistIDs) == 0 || af == nil {
		return nil
	}
	seeds := seedArtistIDs
	if len(seeds) > 5 {
		seeds = seeds[:5]
	}

	path := fmt.Sprintf(
		"/recommendations?seed_artists=%s&limit=40&target_energy=%.2f&target_valence=%.2f&target_danceability=%.2f&target_acousticness=%.2f",
		strings.Join(seeds, ","),
		af.Energy, af.Valence, af.Danceability, af.Acousticness,
	)

	var resp struct {
		Tracks []struct {
			Name    string `json:"name"`
			Artists []struct {
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"artists"`
			Album        struct{ Images []spImage } `json:"album"`
			ExternalURLs map[string]string          `json:"external_urls"`
		} `json:"tracks"`
	}
	if err := spotifyGet(token, path, &resp); err != nil {
		return nil
	}

	seenArtist := map[string]bool{}
	var recs []Recommendation
	for _, t := range resp.Tracks {
		if len(t.Artists) == 0 {
			continue
		}
		a := t.Artists[0]
		_, known := knownArtists[a.Name]
		if known || seenArtist[a.ID] {
			continue
		}
		seenArtist[a.ID] = true
		recs = append(recs, Recommendation{
			ArtistName:  a.Name,
			ArtistID:    a.ID,
			ImageURL:    bestImage(t.Album.Images),
			SampleTrack: t.Name,
			TrackURL:    t.ExternalURLs["spotify"],
		})
		if len(recs) >= 10 {
			break
		}
	}

	// Batch-enrich with artist images, genres, Spotify URLs.
	if len(recs) > 0 {
		ids := make([]string, len(recs))
		for i, r := range recs {
			ids[i] = r.ArtistID
		}
		meta := FetchArtistsMeta(token, ids)
		for i := range recs {
			if m, ok := meta[recs[i].ArtistID]; ok {
				if m.ImageURL != "" {
					recs[i].ImageURL = m.ImageURL
				}
				recs[i].SpotifyURL = m.SpotifyURL
				recs[i].Genres = m.Genres
			}
		}
	}
	return recs
}

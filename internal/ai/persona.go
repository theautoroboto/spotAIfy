// Package ai provides Claude-powered analysis of listening data.
package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type PersonaInput struct {
	Energy            float64
	Danceability      float64
	Valence           float64
	Acousticness      float64
	Instrumentalness  float64
	Liveness          float64
	Tempo             float64
	VibeLabel         string
	VibeDesc          string
	TotalHours        float64
	EarliestYear      int
	YearsActive       int
	UniqueArtists     int
	UniqueTracks      int
	TopArtistNames    []string
	TopGenres         []string
	PeakHour          int
	AvgSkipRate       float64
	AvgCompletionRate float64
}

type Persona struct {
	Archetype string   `json:"archetype"`
	Headline  string   `json:"headline"`
	Narrative string   `json:"narrative"`
	Traits    []string `json:"traits"`
}

// GeneratePersona calls Claude Haiku to produce a psychological listener persona.
// Returns nil (not an error) if the API key is absent — caller treats it as optional.
func GeneratePersona(inp PersonaInput) (*Persona, error) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return nil, nil
	}

	body, _ := json.Marshal(map[string]any{
		"model":      "claude-haiku-4-5-20251001",
		"max_tokens": 900,
		"messages":   []map[string]any{{"role": "user", "content": buildPrompt(inp)}},
	})

	req, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{Timeout: 18 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var cr struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nil, err
	}
	if len(cr.Content) == 0 {
		return nil, fmt.Errorf("empty response")
	}

	// Extract JSON object from response (handles ```json ... ``` wrapping)
	text := cr.Content[0].Text
	if i := strings.Index(text, "{"); i >= 0 {
		if j := strings.LastIndex(text, "}"); j > i {
			text = text[i : j+1]
		}
	}

	var p Persona
	if err := json.Unmarshal([]byte(text), &p); err != nil {
		return nil, fmt.Errorf("parse persona json: %w", err)
	}
	return &p, nil
}

func buildPrompt(inp PersonaInput) string {
	artists := strings.Join(inp.TopArtistNames, ", ")
	genres := strings.Join(inp.TopGenres, ", ")
	if genres == "" {
		genres = "varied"
	}

	return fmt.Sprintf(`You are a music psychologist and cultural analyst. Analyze this listener's data and generate a persona.

LISTENING DATA:
- Total time: %.0f hours across %d years (listening since %d)
- Library: %d unique artists, %d unique tracks
- Peak listening: around %s
- Skip rate: %.0f%% | Song completion rate: %.0f%%

AUDIO FINGERPRINT:
- Energy %d%% | Danceability %d%% | Positivity (valence) %d%%
- Acousticness %d%% | Instrumental %d%% | Liveness %d%%
- Avg tempo: %.0f BPM
- Vibe archetype: %s — %s

TOP ARTISTS: %s
TOP GENRES: %s

Return ONLY a JSON object, no markdown fences, no explanation:
{
  "archetype": "A poetic 3-4 word archetype title (e.g. 'The Restless Romantic', 'The Focused Architect', 'The Midnight Wanderer')",
  "headline": "One powerful sentence capturing their musical soul — specific, evocative, mention the hours or years of listening",
  "narrative": "Three paragraphs separated by \\n\\n. P1: core musical personality and what drives their choices. P2: emotional landscape — how music serves their inner life and what they are seeking. P3: lifestyle identity — what this taste reveals about who they are as a person. Be specific, reference actual numbers/artists, be insightful not generic.",
  "traits": ["Trait1", "Trait2", "Trait3", "Trait4", "Trait5"]
}`,
		inp.TotalHours, inp.YearsActive, inp.EarliestYear,
		inp.UniqueArtists, inp.UniqueTracks,
		hourLabel(inp.PeakHour),
		inp.AvgSkipRate*100, inp.AvgCompletionRate*100,
		int(inp.Energy*100), int(inp.Danceability*100), int(inp.Valence*100),
		int(inp.Acousticness*100), int(inp.Instrumentalness*100), int(inp.Liveness*100),
		inp.Tempo,
		inp.VibeLabel, inp.VibeDesc,
		artists, genres,
	)
}

func hourLabel(h int) string {
	switch {
	case h == 0:
		return "midnight"
	case h < 12:
		return fmt.Sprintf("%dam", h)
	case h == 12:
		return "noon"
	default:
		return fmt.Sprintf("%dpm", h-12)
	}
}

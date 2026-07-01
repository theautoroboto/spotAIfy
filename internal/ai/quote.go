package ai

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// QuoteInput carries the listener's free-text mood/situation plus their music
// taste context for personalized quote matching.
type QuoteInput struct {
	Text       string
	TopArtists []string
	TopGenres  []string
}

// QuoteResult is a relatable song lyric matched to the listener's input, with
// its artist/song attribution and a short line of inspiration.
type QuoteResult struct {
	Artist      string `json:"artist"`
	Song        string `json:"song"`
	Lyric       string `json:"lyric"`
	Inspiration string `json:"inspiration"`
}

// GenerateQuote asks Claude to find a short, relatable lyric excerpt that matches
// the listener's mood/situation, strongly preferring an artist from their own
// listening history when a genuine fit exists, plus a short inspirational line.
func GenerateQuote(inp QuoteInput) (*QuoteResult, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	var sb strings.Builder
	sb.WriteString("A music listener just described their mood or situation. Find a short, relatable song lyric that captures it.\n\n")
	fmt.Fprintf(&sb, "THEIR INPUT: %q\n\n", inp.Text)
	if len(inp.TopArtists) > 0 {
		fmt.Fprintf(&sb, "Their top artists — STRONGLY prefer a song by one of these if it genuinely fits the mood; only reach for someone else if none of these fit at all: %s\n", strings.Join(inp.TopArtists, ", "))
	}
	if len(inp.TopGenres) > 0 {
		fmt.Fprintf(&sb, "Their top genres: %s\n", strings.Join(inp.TopGenres, ", "))
	}
	sb.WriteString("\nFirst privately identify the real artist and song. Then respond with ONLY a JSON object, no other text, no markdown fences, in this exact shape:\n")
	sb.WriteString(`{"artist": "...", "song": "...", "lyric": "...", "inspiration": "..."}` + "\n")
	sb.WriteString("- \"lyric\": a short, real, verbatim excerpt (one line, under 15 words) from that song that best matches their input.\n")
	sb.WriteString("- \"inspiration\": one short, uplifting line of perspective or encouragement inspired by the lyric and their situation — not just a restatement of the lyric.\n")

	raw, err := callClaudePrompt(apiKey, sb.String())
	if err != nil {
		return nil, err
	}

	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)

	var q QuoteResult
	if err := json.Unmarshal([]byte(raw), &q); err != nil {
		return nil, fmt.Errorf("parsing quote JSON: %w (raw: %s)", err, raw)
	}
	if q.Artist == "" || q.Lyric == "" {
		return nil, fmt.Errorf("incomplete quote response")
	}
	return &q, nil
}

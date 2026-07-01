package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// BackgroundInput carries persona and listening data for image prompt construction.
type BackgroundInput struct {
	Archetype  string
	Traits     []string
	Headline   string
	TopGenres  []string
	TopArtists []string
	Hue        int
	Hue2       int
	Energy     float64
	Valence    float64
	Acoustic   float64
}

// api-inference.huggingface.co doesn't resolve in this environment;
// router.huggingface.co is the accessible endpoint.
const hfBgURL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

// GenerateBackground builds a persona-driven image prompt via Claude, then
// calls the HF Inference API to render it. Returns raw image bytes (JPEG/PNG).
func GenerateBackground(inp BackgroundInput) ([]byte, error) {
	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		return nil, fmt.Errorf("HF_TOKEN not configured")
	}

	// Step 1: ask Claude to write a prompt that truly captures the persona.
	prompt, err := personaImagePrompt(inp)
	if err != nil {
		log.Printf("background: Claude prompt failed (%v), using fallback", err)
		prompt = fallbackPrompt(inp)
	}
	log.Printf("background: image prompt — %q", prompt)

	// Step 2: render via HF FLUX.1-schnell.
	return renderImage(hfToken, prompt, 1024, 512)
}

// GenerateAvatar builds a persona-driven portrait prompt via Claude, then calls
// the HF Inference API to render a square image suitable for a circular avatar crop.
func GenerateAvatar(inp BackgroundInput) ([]byte, error) {
	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		return nil, fmt.Errorf("HF_TOKEN not configured")
	}

	prompt, err := personaAvatarPrompt(inp)
	if err != nil {
		log.Printf("avatar: Claude prompt failed (%v), using fallback", err)
		prompt = fallbackAvatarPrompt(inp)
	}
	log.Printf("avatar: image prompt — %q", prompt)

	return renderImage(hfToken, prompt, 512, 512)
}

// genreLine renders the listener's top genres as a short driving phrase, falling
// back to an empty string when no genre data is available.
func genreLine(inp BackgroundInput) string {
	if len(inp.TopGenres) == 0 {
		return ""
	}
	return strings.Join(inp.TopGenres, ", ")
}

// personaImagePrompt calls Claude Haiku to write a Stable Diffusion prompt
// that reflects the listener's persona archetype, traits, and musical identity.
func personaImagePrompt(inp BackgroundInput) (string, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	var sb strings.Builder
	sb.WriteString("Write a Stable Diffusion image generation prompt for a music listener's profile banner image.\n\n")
	sb.WriteString("THEME: Alice in Wonderland characters (Alice, the White Rabbit, the Mad Hatter, the Cheshire Cat, the Queen of Hearts, the Caterpillar, the March Hare, the Dormouse) reimagined as musicians performing together in a surreal Wonderland setting — a rabbit hole stage, a giant teacup drum kit, playing-card banners, oversized mushrooms and flowers. The scene should feel dreamlike and curious while reflecting the listener's specific music taste.\n\n")
	sb.WriteString("This listener's persona:\n")
	if genres := genreLine(inp); genres != "" {
		fmt.Fprintf(&sb, "- PRIMARY STYLE DRIVER — top genres: %s. Let this genre define the overall aesthetic, instruments, color palette, and how each character is costumed, above everything else.\n", genres)
	}
	if inp.Archetype != "" {
		fmt.Fprintf(&sb, "- Archetype: %q\n", inp.Archetype)
	}
	if len(inp.Traits) > 0 {
		fmt.Fprintf(&sb, "- Traits: %s\n", strings.Join(inp.Traits, ", "))
	}
	if inp.Headline != "" {
		fmt.Fprintf(&sb, "- Headline: %q\n", inp.Headline)
	}
	if len(inp.TopArtists) > 0 {
		fmt.Fprintf(&sb, "- Top artists: %s\n", strings.Join(inp.TopArtists, ", "))
	}

	// Energy + valence provide secondary shading (lighting, mood) once genre has
	// set the primary aesthetic.
	var costumeStyle, sceneStyle string
	switch {
	case inp.Energy > 0.72 && inp.Valence < 0.40:
		// Dark energy: high intensity, low happiness
		costumeStyle = "black leather jackets, fishnets, band tees, studded belts, dark eyeliner, chains, goth and metal aesthetic"
		sceneStyle = "dark dramatic Wonderland stage with moody lighting, fog curling from a rabbit hole, electric atmosphere"
	case inp.Energy > 0.72 && inp.Valence >= 0.40:
		// Dance floor: high intensity, high happiness
		costumeStyle = "neon sequin outfits, glitter, festival wristbands, vibrant colors, rave aesthetic"
		sceneStyle = "Wonderland festival stage with colorful lasers, playing cards as confetti, electric and euphoric"
	case inp.Energy > 0.45 && inp.Valence >= 0.65:
		// Upbeat pop/indie
		costumeStyle = "bright colorful band tees, fun accessories, cheerful patterns, vivid costumes"
		sceneStyle = "sunlit garden of giant flowers, warm golden light, joyful tea-party energy"
	case inp.Energy > 0.45:
		// Mid-energy indie/alternative
		costumeStyle = "vintage band tees, flannel shirts, denim jackets, indie look"
		sceneStyle = "cozy candlelit tea party under giant mushrooms, warm amber light"
	case inp.Valence < 0.40:
		// Quiet and dark
		costumeStyle = "dark velvet, somber Victorian coats, brooding poet aesthetic, dark colors"
		sceneStyle = "candlelit stage deep in the rabbit hole, intimate and melancholic atmosphere, deep shadows"
	default:
		// Calm and mellow
		costumeStyle = "soft pastel Victorian wear, comfortable folk-tea-party style"
		sceneStyle = "quiet toadstool glade with warm string lights and drifting playing cards"
	}

	fmt.Fprintf(&sb, "- Energy: %.2f, Mood/Valence: %.2f\n\n", inp.Energy, inp.Valence)

	sb.WriteString("Write a single Stable Diffusion prompt (under 120 words) showing Alice in Wonderland characters as musicians dressed to match this persona, genre-first.\n")
	fmt.Fprintf(&sb, "SECONDARY MOOD/COSTUME SHADING: %s\n", costumeStyle)
	fmt.Fprintf(&sb, "SCENE: %s\n", sceneStyle)
	sb.WriteString("Dress each character (Alice, White Rabbit, Mad Hatter, Cheshire Cat, Queen of Hearts, Caterpillar) in a way that fuses the top genre with this mood — do not make it generically whimsical if the persona and genre call for something darker or heavier. ")
	sb.WriteString("Soft watercolor or storybook illustration style. The image must have: no text, no letters, no words.\n")
	sb.WriteString("Output only the prompt itself, nothing else.")

	return callClaudePrompt(apiKey, sb.String())
}

// personaAvatarPrompt calls Claude Haiku to pick a single Alice in Wonderland
// character that best fits the listener's persona/genre and write a close-up
// portrait prompt suitable for a circular profile avatar.
func personaAvatarPrompt(inp BackgroundInput) (string, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	var sb strings.Builder
	sb.WriteString("Write a Stable Diffusion image generation prompt for a music listener's circular profile avatar.\n\n")
	sb.WriteString("THEME: a single Alice in Wonderland character (choose exactly one of: Alice, the White Rabbit, the Mad Hatter, the Cheshire Cat, the Queen of Hearts, the Caterpillar, the March Hare, the Dormouse) reimagined as a musician, chosen and styled to best match this listener's persona and genre below.\n\n")
	sb.WriteString("This listener's persona:\n")
	if genres := genreLine(inp); genres != "" {
		fmt.Fprintf(&sb, "- PRIMARY DRIVER — top genres: %s. Pick whichever Wonderland character best embodies this genre, and let the genre define costume, instrument, color palette, and expression above everything else.\n", genres)
	}
	if inp.Archetype != "" {
		fmt.Fprintf(&sb, "- Archetype: %q\n", inp.Archetype)
	}
	if len(inp.Traits) > 0 {
		fmt.Fprintf(&sb, "- Traits: %s\n", strings.Join(inp.Traits, ", "))
	}
	if inp.Headline != "" {
		fmt.Fprintf(&sb, "- Headline: %q\n", inp.Headline)
	}
	fmt.Fprintf(&sb, "- Energy: %.2f, Mood/Valence: %.2f\n\n", inp.Energy, inp.Valence)

	sb.WriteString("Write a single Stable Diffusion prompt (under 100 words) for a close-up shoulders-up portrait of ONE chosen character as a musician, genre-first styling. ")
	sb.WriteString("Centered composition, simple uncluttered background (soft Wonderland color wash, no busy scene), so the portrait crops cleanly into a circle. ")
	sb.WriteString("Soft watercolor or storybook illustration style. The image must have: no text, no letters, no words.\n")
	sb.WriteString("Output only the prompt itself, nothing else.")

	return callClaudePrompt(apiKey, sb.String())
}

// callClaudePrompt sends a single-turn prompt-writing request to Claude Haiku
// and returns the trimmed text response.
func callClaudePrompt(apiKey, userPrompt string) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"model":      "claude-haiku-4-5-20251001",
		"max_tokens": 200,
		"messages": []map[string]any{
			{"role": "user", "content": userPrompt},
		},
	})

	req, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("Claude API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return "", fmt.Errorf("Claude API %d: %s", resp.StatusCode, snippet)
	}

	var result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if len(result.Content) == 0 {
		return "", fmt.Errorf("empty Claude response")
	}
	return strings.TrimSpace(result.Content[0].Text), nil
}

func renderImage(hfToken, prompt string, width, height int) ([]byte, error) {
	payload, err := json.Marshal(map[string]any{
		"inputs": prompt,
		"parameters": map[string]any{
			"num_inference_steps": 4,
			"width":               width,
			"height":              height,
		},
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", hfBgURL, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+hfToken)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HF request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("HF %d: %s", resp.StatusCode, snippet)
	}

	return io.ReadAll(resp.Body)
}

// fallbackPrompt is used when Claude is unavailable.
func fallbackPrompt(inp BackgroundInput) string {
	var costumeStyle, sceneStyle string
	switch {
	case inp.Energy > 0.72 && inp.Valence < 0.40:
		costumeStyle = "black leather jackets fishnets studded belts goth metal aesthetic dark eyeliner"
		sceneStyle = "dark dramatic Wonderland stage moody lighting fog curling from a rabbit hole"
	case inp.Energy > 0.72 && inp.Valence >= 0.40:
		costumeStyle = "neon sequin outfits glitter festival wristbands vibrant rave aesthetic"
		sceneStyle = "Wonderland festival stage colorful lasers playing-card confetti euphoric"
	case inp.Energy > 0.45 && inp.Valence >= 0.65:
		costumeStyle = "bright colorful band tees fun accessories cheerful vivid costumes"
		sceneStyle = "sunlit garden of giant flowers warm golden light"
	case inp.Valence < 0.40:
		costumeStyle = "dark velvet somber Victorian coats brooding poet aesthetic"
		sceneStyle = "candlelit stage deep in the rabbit hole intimate melancholic deep shadows"
	default:
		costumeStyle = "soft pastel Victorian wear folk-tea-party style"
		sceneStyle = "quiet toadstool glade warm string lights drifting playing cards"
	}
	genreSuffix := ""
	if genres := genreLine(inp); genres != "" {
		genreSuffix = fmt.Sprintf(", %s-inspired styling and instruments", genres)
	}
	return fmt.Sprintf(
		"Alice in Wonderland characters as musicians wearing %s%s, Alice on vocals, White Rabbit on pocket-watch drum kit, Mad Hatter on keys, Cheshire Cat on guitar, Queen of Hearts conducting, %s, soft watercolor storybook illustration style, no text, ultra detailed",
		costumeStyle, genreSuffix, sceneStyle,
	)
}

// fallbackAvatarPrompt is used when Claude is unavailable. It maps energy/valence
// to a single Wonderland character rather than the full ensemble.
func fallbackAvatarPrompt(inp BackgroundInput) string {
	var character, mood string
	switch {
	case inp.Energy > 0.72 && inp.Valence < 0.40:
		character = "the Queen of Hearts, fierce and commanding"
		mood = "dark dramatic lighting, goth and metal aesthetic"
	case inp.Energy > 0.72 && inp.Valence >= 0.40:
		character = "the Mad Hatter, wild and exuberant"
		mood = "neon festival colors, glitter, euphoric energy"
	case inp.Energy > 0.45 && inp.Valence >= 0.65:
		character = "the White Rabbit, cheerful and animated"
		mood = "bright warm golden light, cheerful vivid colors"
	case inp.Energy > 0.45:
		character = "Alice herself, curious and adventurous"
		mood = "cozy amber tea-party light, vintage indie styling"
	case inp.Valence < 0.40:
		character = "the Caterpillar atop a mushroom, contemplative and mysterious"
		mood = "candlelit shadows, somber Victorian styling"
	default:
		character = "the Dormouse, sleepy and whimsical"
		mood = "soft pastel colors, gentle string lights"
	}
	genreSuffix := ""
	if genres := genreLine(inp); genres != "" {
		genreSuffix = fmt.Sprintf(" as a %s musician", genres)
	}
	return fmt.Sprintf(
		"Close-up shoulders-up portrait of %s%s, %s, simple uncluttered background for a circular crop, soft watercolor storybook illustration style, no text, ultra detailed",
		character, genreSuffix, mood,
	)
}

func hueToColor(hue int) string {
	hue = ((hue % 360) + 360) % 360
	switch {
	case hue < 15 || hue >= 345:
		return "deep crimson"
	case hue < 40:
		return "burnt orange"
	case hue < 65:
		return "golden amber"
	case hue < 90:
		return "olive"
	case hue < 150:
		return "emerald green"
	case hue < 175:
		return "teal"
	case hue < 210:
		return "cerulean blue"
	case hue < 250:
		return "deep indigo"
	case hue < 290:
		return "violet purple"
	case hue < 320:
		return "magenta"
	default:
		return "rose pink"
	}
}

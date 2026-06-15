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
	return renderImage(hfToken, prompt)
}

// GenerateAvatar renders a square profile picture: a single Alice in
// Wonderland character styled after the listener's top 5 artists. The prompt
// is written by Claude, with a deterministic fallback (artist-based when
// history exists, mood-based otherwise).
func GenerateAvatar(inp BackgroundInput) ([]byte, error) {
	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		return nil, fmt.Errorf("HF_TOKEN not configured")
	}

	prompt, err := artistAvatarPrompt(inp)
	if err != nil {
		log.Printf("avatar: Claude prompt failed (%v), using fallback", err)
		prompt = fallbackAvatarPrompt(inp)
	}
	log.Printf("avatar: image prompt — %q", prompt)
	return renderImageSized(hfToken, prompt, 512, 512)
}

// topFive caps an artist list at five entries.
func topFive(artists []string) []string {
	if len(artists) > 5 {
		return artists[:5]
	}
	return artists
}

// avatarMoodLook maps energy/valence to outfit and mood descriptors shared by
// the Claude-written and fallback avatar prompts.
func avatarMoodLook(energy, valence float64) (look, mood string) {
	switch {
	case energy > 0.72 && valence < 0.40:
		return "black leather jacket, dark eyeliner, goth metal aesthetic",
			"intense brooding expression, dramatic moody stage lighting"
	case energy > 0.72 && valence >= 0.40:
		return "neon sequin outfit, glitter, rave aesthetic",
			"euphoric grin, colorful festival lights"
	case energy > 0.45 && valence >= 0.65:
		return "bright colorful band tee, fun accessories",
			"beaming joyful expression, sunny golden light"
	case energy > 0.45:
		return "vintage band tee, flannel shirt, denim jacket, indie look",
			"relaxed contented smile, warm amber venue light"
	case valence < 0.40:
		return "dark velvet, brooding poet aesthetic",
			"wistful melancholic expression, candlelit shadows"
	default:
		return "soft earth tones, comfortable folk wear",
			"calm gentle smile, cozy warm string lights"
	}
}

// artistAvatarPrompt asks Claude Haiku to write the avatar prompt from the
// listener's top 5 artists.
func artistAvatarPrompt(inp BackgroundInput) (string, error) {
	if len(inp.TopArtists) == 0 {
		return "", fmt.Errorf("no top artists available")
	}

	var sb strings.Builder
	sb.WriteString("Write a Stable Diffusion image generation prompt for a music listener's square profile avatar.\n\n")
	sb.WriteString("THEME: a portrait of a single character from Lewis Carroll's \"Alice's Adventures in Wonderland\" — pick whichever best embodies the listener's favourite artists (Alice, the Mad Hatter, the White Rabbit, the Cheshire Cat, the Queen of Hearts, or the Caterpillar) — as a musician channeling those artists.\n\n")
	fmt.Fprintf(&sb, "The listener's top artists, most-listened first: %s\n", strings.Join(topFive(inp.TopArtists), ", "))
	if len(inp.TopGenres) > 0 {
		fmt.Fprintf(&sb, "Their genres: %s\n", strings.Join(inp.TopGenres, ", "))
	}
	fmt.Fprintf(&sb, "BACKGROUND: soft %s\n\n", hueToColor(inp.Hue))
	sb.WriteString("Write a single Stable Diffusion prompt (under 80 words) showing a centered close-up portrait of one character. The character's outfit, instrument, accessories, expression, and lighting must visibly channel these artists' iconic looks and the feel of their music — specific, not generic, and not cute or cheerful if the artists are dark. ")
	sb.WriteString("The art style must be the authentic style of John Tenniel's original 1865 illustrations for Lewis Carroll's book: Victorian pen-and-ink engraving, fine crosshatching, muted hand-tinted watercolor, antique storybook plate. The prompt must explicitly state: no anime, no manga, no Disney, no modern cartoon, no 3D render. The image must have: no text, no letters, no words.\n")
	sb.WriteString("Output only the prompt itself, nothing else.")

	return askClaude(sb.String())
}

// fallbackAvatarPrompt is used when Claude is unavailable. Artist-based when
// listening history exists, mood-based otherwise.
func fallbackAvatarPrompt(inp BackgroundInput) string {
	if len(inp.TopArtists) > 0 {
		return fmt.Sprintf(
			"Portrait of a single character from Lewis Carroll's Alice's Adventures in Wonderland as a musician styled after %s, centered close-up on a soft %s background, authentic John Tenniel 1865 illustration style, Victorian pen-and-ink engraving, fine crosshatching, muted hand-tinted watercolor, antique storybook plate, no anime, no manga, no Disney, no modern cartoon, no 3D render, no text, ultra detailed",
			strings.Join(topFive(inp.TopArtists), ", "), hueToColor(inp.Hue),
		)
	}
	look, mood := avatarMoodLook(inp.Energy, inp.Valence)
	return fmt.Sprintf(
		"Portrait of a single character from Lewis Carroll's Alice's Adventures in Wonderland as a musician wearing %s, %s, centered close-up on a soft %s background, authentic John Tenniel 1865 illustration style, Victorian pen-and-ink engraving, fine crosshatching, muted hand-tinted watercolor, antique storybook plate, no anime, no manga, no Disney, no modern cartoon, no 3D render, no text, ultra detailed",
		look, mood, hueToColor(inp.Hue),
	)
}

// personaImagePrompt calls Claude Haiku to write a Stable Diffusion prompt
// that reflects the listener's persona archetype, traits, and musical identity.
func personaImagePrompt(inp BackgroundInput) (string, error) {
	var sb strings.Builder
	sb.WriteString("Write a Stable Diffusion image generation prompt for a music listener's profile banner image.\n\n")
	sb.WriteString("THEME: characters from Lewis Carroll's \"Alice's Adventures in Wonderland\" (Alice, the Mad Hatter, the White Rabbit, the Cheshire Cat, the Queen of Hearts, the Caterpillar) dressed as musicians and styled to match this listener's musical persona. The scene should feel warm, whimsical, and charming while reflecting the listener's specific music taste.\n\n")
	sb.WriteString("This listener's persona:\n")
	if inp.Archetype != "" {
		fmt.Fprintf(&sb, "- Archetype: %q\n", inp.Archetype)
	}
	if len(inp.Traits) > 0 {
		fmt.Fprintf(&sb, "- Traits: %s\n", strings.Join(inp.Traits, ", "))
	}
	if inp.Headline != "" {
		fmt.Fprintf(&sb, "- Headline: %q\n", inp.Headline)
	}
	if len(inp.TopGenres) > 0 {
		fmt.Fprintf(&sb, "- Top genres: %s\n", strings.Join(inp.TopGenres, ", "))
	}
	if len(inp.TopArtists) > 0 {
		fmt.Fprintf(&sb, "- Top artists: %s\n", strings.Join(inp.TopArtists, ", "))
	}

	// Derive costume and scene style from energy + valence
	var costumeStyle, sceneStyle string
	switch {
	case inp.Energy > 0.72 && inp.Valence < 0.40:
		// Dark energy: high intensity, low happiness
		costumeStyle = "black leather jackets, fishnets, band tees, studded belts, dark eyeliner, chains, goth and metal aesthetic"
		sceneStyle = "dark dramatic stage with moody lighting, fog machine, electric atmosphere"
	case inp.Energy > 0.72 && inp.Valence >= 0.40:
		// Dance floor: high intensity, high happiness
		costumeStyle = "neon sequin outfits, glitter, festival wristbands, vibrant colors, rave aesthetic"
		sceneStyle = "festival stage with colorful lasers and confetti, electric and euphoric"
	case inp.Energy > 0.45 && inp.Valence >= 0.65:
		// Upbeat pop/indie
		costumeStyle = "bright colorful band tees, fun accessories, cheerful patterns, vivid costumes"
		sceneStyle = "sunny outdoor stage, warm golden light, joyful crowd energy"
	case inp.Energy > 0.45:
		// Mid-energy indie/alternative
		costumeStyle = "vintage band tees, flannel shirts, denim jackets, indie look"
		sceneStyle = "cozy intimate venue, warm amber stage lights"
	case inp.Valence < 0.40:
		// Quiet and dark
		costumeStyle = "dark velvet, somber academic robes, brooding poet aesthetic, dark colors"
		sceneStyle = "candlelit stage, intimate and melancholic atmosphere, deep shadows"
	default:
		// Calm and mellow
		costumeStyle = "soft earth tones, comfortable folk wear, acoustic musician style"
		sceneStyle = "whimsical Wonderland tea-party clearing with warm lanterns and oversized mushrooms"
	}

	fmt.Fprintf(&sb, "- Energy: %.2f, Mood/Valence: %.2f\n\n", inp.Energy, inp.Valence)

	sb.WriteString("Write a single Stable Diffusion prompt (under 120 words) showing Alice in Wonderland characters as a band of musicians dressed to match this persona.\n")
	fmt.Fprintf(&sb, "COSTUME STYLE: %s\n", costumeStyle)
	fmt.Fprintf(&sb, "SCENE: %s\n", sceneStyle)
	sb.WriteString("Dress each character (Mad Hatter on bass, Alice at keyboard, White Rabbit with guitar, Cheshire Cat at drums, Queen of Hearts singing) in this style. The costumes and scene must match the persona — do not make it generically cute or cheerful if the persona is dark. ")
	sb.WriteString("The art style must be the authentic style of John Tenniel's original 1865 illustrations for Lewis Carroll's book: Victorian pen-and-ink engraving, fine crosshatching, muted hand-tinted watercolor, antique storybook plate. The prompt must explicitly state: no anime, no manga, no Disney, no modern cartoon, no 3D render. The image must have: no text, no letters, no words.\n")
	sb.WriteString("Output only the prompt itself, nothing else.")

	return askClaude(sb.String())
}

// askClaude sends a single-message prompt to Claude Haiku and returns the
// trimmed text response.
func askClaude(message string) (string, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	body, _ := json.Marshal(map[string]any{
		"model":      "claude-haiku-4-5-20251001",
		"max_tokens": 200,
		"messages": []map[string]any{
			{"role": "user", "content": message},
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

func renderImage(hfToken, prompt string) ([]byte, error) {
	return renderImageSized(hfToken, prompt, 1024, 512)
}

func renderImageSized(hfToken, prompt string, width, height int) ([]byte, error) {
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
		sceneStyle = "dark dramatic stage moody lighting fog machine"
	case inp.Energy > 0.72 && inp.Valence >= 0.40:
		costumeStyle = "neon sequin outfits glitter festival wristbands vibrant rave aesthetic"
		sceneStyle = "festival stage colorful lasers confetti euphoric"
	case inp.Energy > 0.45 && inp.Valence >= 0.65:
		costumeStyle = "bright colorful band tees fun accessories cheerful vivid costumes"
		sceneStyle = "sunny outdoor stage warm golden light"
	case inp.Valence < 0.40:
		costumeStyle = "dark velvet somber academic robes brooding poet aesthetic"
		sceneStyle = "candlelit stage intimate melancholic deep shadows"
	default:
		costumeStyle = "vintage band tees flannel shirts denim jackets indie look"
		sceneStyle = "whimsical Wonderland tea-party clearing warm lanterns oversized mushrooms"
	}
	return fmt.Sprintf(
		"Characters from Lewis Carroll's Alice's Adventures in Wonderland as a band of musicians wearing %s, Mad Hatter on bass, White Rabbit with guitar, Cheshire Cat at drums, Alice at keyboard, Queen of Hearts singing, %s, authentic John Tenniel 1865 illustration style, Victorian pen-and-ink engraving, fine crosshatching, muted hand-tinted watercolor, antique storybook plate, no anime, no manga, no Disney, no modern cartoon, no 3D render, no text, ultra detailed",
		costumeStyle, sceneStyle,
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

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

// personaImagePrompt calls Claude Haiku to write a Stable Diffusion prompt
// that reflects the listener's persona archetype, traits, and musical identity.
func personaImagePrompt(inp BackgroundInput) (string, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	var sb strings.Builder
	sb.WriteString("Write a Stable Diffusion image generation prompt for a music listener's profile banner image.\n\n")
	sb.WriteString("THEME: Winnie the Pooh characters (Pooh, Tigger, Eeyore, Piglet, Rabbit, Owl, Roo) dressed as musicians and styled to match this listener's musical persona. The scene should feel warm, whimsical, and charming while reflecting the listener's specific music taste.\n\n")
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
		sceneStyle = "cozy Hundred Acre Wood clearing with warm string lights"
	}

	fmt.Fprintf(&sb, "- Energy: %.2f, Mood/Valence: %.2f\n\n", inp.Energy, inp.Valence)

	sb.WriteString("Write a single Stable Diffusion prompt (under 120 words) showing Winnie the Pooh characters as musicians dressed to match this persona.\n")
	fmt.Fprintf(&sb, "COSTUME STYLE: %s\n", costumeStyle)
	fmt.Fprintf(&sb, "SCENE: %s\n", sceneStyle)
	sb.WriteString("Dress each character (Pooh, Tigger, Eeyore, Piglet, Rabbit, Owl) in this style. The costumes and scene must match the persona — do not make it generically cute or cheerful if the persona is dark. ")
	sb.WriteString("Soft watercolor or storybook illustration style. The image must have: no text, no letters, no words.\n")
	sb.WriteString("Output only the prompt itself, nothing else.")

	body, _ := json.Marshal(map[string]any{
		"model":      "claude-haiku-4-5-20251001",
		"max_tokens": 200,
		"messages": []map[string]any{
			{"role": "user", "content": sb.String()},
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
	payload, err := json.Marshal(map[string]any{
		"inputs": prompt,
		"parameters": map[string]any{
			"num_inference_steps": 4,
			"width":               1024,
			"height":              512,
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
		sceneStyle = "cozy Hundred Acre Wood clearing warm string lights"
	}
	return fmt.Sprintf(
		"Winnie the Pooh characters as musicians wearing %s, Pooh on bass, Tigger with guitar, Eeyore with headphones, Piglet at keyboard, Rabbit conducting, %s, soft watercolor storybook illustration style, no text, ultra detailed",
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

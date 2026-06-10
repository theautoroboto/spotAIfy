package ai

import (
	"strings"
	"testing"
)

func TestHueToColor(t *testing.T) {
	cases := []struct {
		hue  int
		want string
	}{
		{0, "deep crimson"},
		{350, "deep crimson"},
		{30, "burnt orange"},
		{50, "golden amber"},
		{120, "emerald green"},
		{160, "teal"},
		{200, "cerulean blue"},
		{230, "deep indigo"},
		{270, "violet purple"},
		{300, "magenta"},
		{330, "rose pink"},
		{360, "deep crimson"},  // wraps
		{-10, "deep crimson"},  // negative wraps to 350
		{485, "emerald green"}, // 485 → 125
	}
	for _, c := range cases {
		if got := hueToColor(c.hue); got != c.want {
			t.Errorf("hueToColor(%d) = %q, want %q", c.hue, got, c.want)
		}
	}
}

func TestFallbackPromptMoodBranches(t *testing.T) {
	cases := []struct {
		name             string
		energy, valence  float64
		wantSubstring    string
	}{
		{"dark high-energy", 0.80, 0.20, "goth metal"},
		{"euphoric dance", 0.80, 0.60, "rave aesthetic"},
		{"upbeat pop", 0.50, 0.70, "bright colorful band tees"},
		{"quiet melancholy", 0.30, 0.20, "brooding poet"},
		{"calm default", 0.30, 0.50, "indie look"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := fallbackPrompt(BackgroundInput{Energy: c.energy, Valence: c.valence})
			if !strings.Contains(got, c.wantSubstring) {
				t.Errorf("fallbackPrompt(energy=%.2f, valence=%.2f) = %q, want substring %q",
					c.energy, c.valence, got, c.wantSubstring)
			}
			if !strings.Contains(got, "Wonderland") || !strings.Contains(got, "Tenniel") {
				t.Errorf("fallbackPrompt missing theme/style anchors, got %q", got)
			}
			if !strings.Contains(got, "no text") {
				t.Errorf("fallbackPrompt missing no-text guard, got %q", got)
			}
		})
	}
}

func TestFallbackAvatarPromptMoodBranches(t *testing.T) {
	cases := []struct {
		name            string
		energy, valence float64
		wantSubstring   string
	}{
		{"dark high-energy", 0.80, 0.20, "goth metal"},
		{"euphoric dance", 0.80, 0.60, "euphoric grin"},
		{"upbeat pop", 0.50, 0.70, "beaming joyful"},
		{"mid-energy indie", 0.50, 0.50, "indie look"},
		{"quiet melancholy", 0.30, 0.20, "brooding poet"},
		{"calm default", 0.30, 0.50, "calm gentle smile"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := fallbackAvatarPrompt(BackgroundInput{Energy: c.energy, Valence: c.valence, Hue: 120})
			if !strings.Contains(got, c.wantSubstring) {
				t.Errorf("fallbackAvatarPrompt(energy=%.2f, valence=%.2f) = %q, want substring %q",
					c.energy, c.valence, got, c.wantSubstring)
			}
			if !strings.Contains(got, "Wonderland") || !strings.Contains(got, "Tenniel") {
				t.Errorf("fallbackAvatarPrompt missing theme/style anchors, got %q", got)
			}
			if !strings.Contains(got, "emerald green") {
				t.Errorf("fallbackAvatarPrompt missing aura color, got %q", got)
			}
		})
	}
}

func TestFallbackAvatarPromptUsesTopArtists(t *testing.T) {
	inp := BackgroundInput{
		TopArtists: []string{"Nine Inch Nails", "Primus", "Tool", "Ministry", "Girl Talk", "Sixth Artist"},
		Hue:        200,
	}
	got := fallbackAvatarPrompt(inp)
	for _, artist := range inp.TopArtists[:5] {
		if !strings.Contains(got, artist) {
			t.Errorf("fallbackAvatarPrompt missing top artist %q, got %q", artist, got)
		}
	}
	if strings.Contains(got, "Sixth Artist") {
		t.Errorf("fallbackAvatarPrompt should cap at 5 artists, got %q", got)
	}
	if !strings.Contains(got, "cerulean blue") {
		t.Errorf("fallbackAvatarPrompt missing aura color, got %q", got)
	}
}

func TestGenerateAvatarRequiresToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	if _, err := GenerateAvatar(BackgroundInput{}); err == nil {
		t.Error("GenerateAvatar without HF_TOKEN should error, got nil")
	}
}

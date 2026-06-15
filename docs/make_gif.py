"""
Stitch screenshots into an optimized animated GIF using Pillow.
Frames: home → sonic prompt → connection → DNA → filters → profile banner → profile stats
"""
from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = "docs/screenshots"
GIF_PATH = "docs/demo.gif"
WIDTH = 900
DURATIONS = [2800, 2200, 2200, 2200, 2200, 3000, 3500]  # ms per frame

FRAMES = [
    ("01_home.png",        "Seven playlist modes"),
    ("02_sonic_prompt.png","Sonic — describe any vibe"),
    ("03_connection_mode.png", "Connection — explore an artist's network"),
    ("04_dna_mode.png",    "DNA — map a track's sample lineage"),
    ("05_form_filters.png","Audio filters — energy, valence, tempo"),
    ("06_profile_top.png", "Profile — AI-generated listening portrait"),
    ("07_profile_stats.png","Profile — year-by-year listening history"),
]

def resize(img, width):
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.LANCZOS)

def add_caption(img, text):
    """Add a semi-transparent caption bar at the bottom."""
    draw = ImageDraw.Draw(img)
    bar_h = 36
    y = img.height - bar_h
    draw.rectangle([0, y, img.width, img.height], fill=(0, 0, 0, 180))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, y + 8), text, fill=(255, 255, 255), font=font)
    return img

frames = []
for (filename, caption), duration in zip(FRAMES, DURATIONS):
    path = os.path.join(OUT_DIR, filename)
    img = Image.open(path).convert("RGBA")
    img = resize(img, WIDTH)
    img = add_caption(img, caption)
    # Convert to P mode with optimized palette for GIF
    frames.append((img.convert("RGB").quantize(colors=256, dither=Image.Dither.FLOYDSTEINBERG), duration))

images = [f for f, _ in frames]
durations = [d for _, d in frames]

images[0].save(
    GIF_PATH,
    save_all=True,
    append_images=images[1:],
    duration=durations,
    loop=0,
    optimize=True,
)

size_kb = os.path.getsize(GIF_PATH) // 1024
print(f"Saved {GIF_PATH} ({size_kb} KB, {len(images)} frames)")

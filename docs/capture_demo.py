"""Capture demo screenshots for the README.

Usage:
    SPOTAIFY_DEMO_USER=brian SPOTAIFY_DEMO_PASS=yourpassword \\
        uv run --with playwright python docs/capture_demo.py
"""
import os, time
from playwright.sync_api import sync_playwright

PORT = int(os.environ.get("PORT", "8000"))
BASE = f"http://localhost:{PORT}"
OUT = "docs/screenshots"
USER = os.environ["SPOTAIFY_DEMO_USER"]
PASS = os.environ["SPOTAIFY_DEMO_PASS"]

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()

        # ── 1. Login ──────────────────────────────────────────────────────────
        page.goto(f"{BASE}/login")
        page.fill('input[name="username"]', USER)
        page.fill('input[name="password"]', PASS)
        page.click('button[type="submit"]')
        page.wait_for_url(f"{BASE}/")
        time.sleep(1)

        # ── 2. Home — all mode tabs visible ──────────────────────────────────
        page.screenshot(path=f"{OUT}/01_home.png", full_page=False)
        print("01_home.png")

        def click_mode(value):
            page.locator(f'input[name="mode"][value="{value}"]').click(force=True)
            time.sleep(0.3)

        # ── 3. Sonic mode — type a prompt ────────────────────────────────────
        click_mode("sonic")
        page.fill('input[name="prompt"]', "melancholic post-rock for a rainy evening")
        time.sleep(0.3)
        page.screenshot(path=f"{OUT}/02_sonic_prompt.png", full_page=False)
        print("02_sonic_prompt.png")

        # ── 4. Connection mode tab ────────────────────────────────────────────
        click_mode("connection")
        page.fill('input[name="artist"]', "Radiohead")
        time.sleep(0.3)
        page.screenshot(path=f"{OUT}/03_connection_mode.png", full_page=False)
        print("03_connection_mode.png")

        # ── 5. DNA mode tab ───────────────────────────────────────────────────
        click_mode("dna")
        page.fill('input[name="dna_track"]', "Hurt")
        page.fill('input[name="dna_artist"]', "Nine Inch Nails")
        time.sleep(0.3)
        page.screenshot(path=f"{OUT}/04_dna_mode.png", full_page=False)
        print("04_dna_mode.png")

        # ── 6. Filters open + full form visible ──────────────────────────────
        click_mode("sonic")
        page.fill('input[name="prompt"]', "melancholic post-rock for a rainy evening")
        page.locator('details.filters').evaluate("el => el.setAttribute('open', '')")
        time.sleep(0.4)
        page.screenshot(path=f"{OUT}/05_form_filters.png", full_page=False)
        print("05_form_filters.png")

        # ── 7. Profile page ───────────────────────────────────────────────────
        page.goto(f"{BASE}/profile")
        page.wait_for_load_state("networkidle", timeout=15000)
        time.sleep(2)
        page.screenshot(path=f"{OUT}/06_profile_top.png", full_page=False)
        print("06_profile_top.png")

        # Scroll down to show stats / persona
        page.evaluate("window.scrollBy(0, 600)")
        time.sleep(0.5)
        page.screenshot(path=f"{OUT}/07_profile_stats.png", full_page=False)
        print("07_profile_stats.png")

        browser.close()
        print("Done.")

if __name__ == "__main__":
    run()

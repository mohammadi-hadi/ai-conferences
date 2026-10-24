"""Still images of the interactive overlap map, taken from the page itself (needs playwright and Chrome):
figures/overlap_preview.png for the story and the hub, figures/overlap_card.png (1200x630) for link previews."""
import asyncio
from playwright.async_api import async_playwright
from config import ROOT

FIGS = ROOT / "docs/2026/figures"
URL = (ROOT / "docs/2026/overlap.html").as_uri()

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        for name, w, h in [("overlap_preview.png", 1280, 800), ("overlap_card.png", 1200, 630)]:
            pg = await b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            await pg.goto(URL, wait_until="networkidle")
            await pg.evaluate("document.fonts.ready")
            await pg.add_style_tag(content=".links{display:none!important}")
            await pg.wait_for_timeout(900)
            await pg.screenshot(path=str(FIGS / name))
            print("wrote", name)
        await b.close()

asyncio.run(main())

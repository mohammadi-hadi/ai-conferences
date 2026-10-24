"""Build the GitHub Pages site in the style of mohammadi.cv: docs/index.html (the hub that lists the atlases and the
comparison) and docs/2026/index.html (the story, from report/BLOG.md). Needs pandoc on PATH."""
import html, json, re, shutil, subprocess
from config import OUT, ROOT
from site_template import page, atlas_nav

SITE = "https://mohammadi.cv/ai-conferences/2026/"
DOCS = ROOT / "docs/2026"
(DOCS / "figures").mkdir(parents=True, exist_ok=True)
for f in (OUT / "figures").glob("*.png"):
    shutil.copy(f, DOCS / "figures" / f.name)

HERO = ('</div><section class="container atlas-hero" aria-label="map of both conferences">'
        '<div class="map-frame"><img class="map-preview" src="figures/map_years.png" '
        'alt="NeurIPS and EMNLP papers of 2025 and 2026 on one map">'
        '<button class="map-cover" type="button" onclick="openMap(this)"><span>Explore the map</span></button></div>'
        '<div class="actions"><a class="btn" href="atlas.html">Open the map full screen</a></div></section>'
        '<script>function openMap(b){if(window.innerWidth<900){location.href="atlas.html";return;}'
        'b.parentNode.innerHTML=\'<iframe src="atlas.html" title="NeurIPS and EMNLP on one map"></iframe>\';}</script>'
        '<div class="post-wrap">')


def img_size(src):
    from PIL import Image
    f = DOCS / src
    if not f.exists():
        return ""
    w, h = Image.open(f).size
    return f'width="{w}" height="{h}" '


def md_to_html(md_path):
    body = subprocess.run(["pandoc", str(md_path), "-f", "gfm", "-t", "html5", "--wrap=none"], capture_output=True,
                          text=True, check=True).stdout
    body = body.replace("../outputs/figures/", "figures/").replace(SITE, "")
    title = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S)
    body = re.sub(r'<p><img\s+src="([^"]+)"\s+alt="([^"]*)"\s*/?></p>',
                  lambda m: f'<figure><a href="{m.group(1)}"><img src="{m.group(1)}" alt="{m.group(2)}" '
                            f'{img_size(m.group(1))}loading="lazy"></a>'
                            f'<figcaption>{m.group(2)}<span class="tap"> · Tap to enlarge.</span></figcaption></figure>', body)
    body = re.sub(r"<!--\s*atlas-hero\s*-->\s*<p><strong><a [^>]*>[^<]*</a></strong></p>", HERO, body)
    return (html.unescape(re.sub(r"<[^>]+>", "", title.group(1))) if title else ""), body


CRUMBS = '<a href="/blog/">Blog</a><span class="sep">/</span><a href="../">AI Conferences</a>'
blog = ROOT / "report/BLOG.md"
meta = dict(re.findall(r"<!--\s*(dek|date):\s*(.*?)\s*-->", blog.read_text()))
words = len(re.sub(r"[#*\[\]()!<>-]", " ", blog.read_text()).split())
title, body = md_to_html(blog)
body = f"""{atlas_nav("", "index.html")}<main id="main"><article class="post">
<header class="post-head"><div class="post-wrap">
<p class="post-crumbs">{CRUMBS}</p><h1>{title}</h1><p class="post-dek">{meta.get("dek", "")}</p>
<p class="post-byline"><span>{round(words / 220)} min read · {meta.get("date", "")}</span></p></div></header>
<div class="post-body"><div class="post-wrap">{body}</div></div></article></main>"""
(DOCS / "index.html").write_text(page(title, meta.get("dek", ""), SITE, body, image=f"{SITE}figures/map_years.png"))
print("wrote", DOCS / "index.html")

root = DOCS.parent
(root / ".nojekyll").write_text("")
counts = json.load(open(ROOT / "data/processed/load.json"))["counts"]
LANDING = f"""{atlas_nav("2026/")}<main id="main"><section class="hero"><div class="container">
<span class="kicker">Project</span><h1>AI Conferences</h1>
<p class="section-lede">Open maps of the papers at the big AI conferences, made the same way each time, so they can be compared.</p>
</div></section>
<section class="section"><div class="container">
<a href="2026/atlas.html"><img src="2026/figures/map_years.png" alt="NeurIPS and EMNLP papers of 2025 and 2026 on one map"
 class="landing-map"></a>
<div class="landing-actions"><a class="btn" href="2026/atlas.html">Open the joint map</a>
<a class="btn btn-ghost" href="2026/">Read the comparison</a></div>
<div class="prose" style="margin-top:28px">
<h2>2026</h2>
<p><a href="2026/"><strong>NeurIPS and EMNLP moved closer in 2026.</strong></a> Both conferences, 2025 and 2026, on one map:
what brought them together and what still sets them apart.</p>
<p><a href="/neurips/2026/"><strong>NeurIPS 2026 atlas.</strong></a> All {counts["NeurIPS"]["2026"]:,} papers on one map, with the
city, day and time of every poster.</p>
<p><a href="/emnlp/2026/"><strong>EMNLP 2026 atlas.</strong></a> All {counts["EMNLP"]["2026"]:,} papers on one map, with the room
and time of every talk and poster.</p>
</div></div></section></main>"""
(root / "index.html").write_text(page("AI Conferences", "Open maps of the papers at the big AI conferences, made the same way each time.",
                                      "https://mohammadi.cv/ai-conferences/", LANDING,
                                      image=f"{SITE}figures/map_years.png", kind="website"))
print("wrote", root / "index.html")

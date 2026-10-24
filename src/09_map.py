"""Interactive joint map (DataMapPlot), in the style of the two atlases: all NeurIPS and EMNLP papers of 2025 and
2026, coloured by conference and year, labelled with the shared topics."""
import json, re
import numpy as np
import pandas as pd
import datamapplot
from config import P, ROOT

DOCS = ROOT / "docs/2026"
DOCS.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(P / "papers.parquet")
df = pd.concat([df, pd.read_parquet(P / "umap.parquet"), pd.read_parquet(P / "topics.parquet")[["topic"]]], axis=1)
names = json.load(open(ROOT / "data/labels/topic_names.json"))
topic = df["topic"].map(lambda t: names[str(t)]["name"]).to_numpy()
area = df["topic"].map(lambda t: names[str(t)]["area"]).to_numpy()

COL = {("NeurIPS", 2025): "#91a7ff", ("NeurIPS", 2026): "#3b5bdb", ("EMNLP", 2025): "#ffa94d", ("EMNLP", 2026): "#d9480f"}
color = np.array([COL[(c, y)] for c, y in zip(df["conf"], df["year"])])
group = (df["conf"] + "-" + df["year"].astype(str)).to_numpy()
title = df["title"].str.strip()
extra = pd.DataFrame({
    "title": title, "safe_title": title.str.replace(r"[`$\\]", "", regex=True), "url": df["url"],
    "authors": df["authors"].map(lambda a: ", ".join(a))[:].str.slice(0, 160),
    "conf": df["conf"] + " " + df["year"].astype(str), "topic": topic,
    "abstract": df["abstract"].str.slice(0, 380) + "…",
})
hover = (title + " — " + df["conf"] + " " + df["year"].astype(str)).to_numpy()
cmap_raw = [df["conf"].to_numpy(), df["year"].astype(str).to_numpy(), area]
cmap_meta = [
    {"field": "conference", "description": "Conference", "kind": "categorical",
     "color_mapping": {"NeurIPS": "#3b5bdb", "EMNLP": "#d9480f"}},
    {"field": "year", "description": "Year", "kind": "categorical", "color_mapping": {"2025": "#adb5bd", "2026": "#1e3a5f"}},
    {"field": "area", "description": "Broad area", "kind": "categorical",
     "color_mapping": dict(zip(sorted(set(area)), ["#1e3a5f", "#d9480f", "#2b8a3e", "#7048e8", "#f08c00", "#0c8599",
                                                   "#c2255c", "#5c940d", "#868e96", "#e8590c", "#364fc7"]))},
]
CSS = """
#title-container{background:rgba(255,255,255,.94)!important;border-radius:10px;box-shadow:0 2px 12px rgba(31,42,68,.08);
  padding:14px 18px 13px!important;max-width:min(600px,calc(100vw - 48px));line-height:1.2!important}
#title-container br{display:none}
#main-title{display:block;font-family:"Source Serif 4",Georgia,serif!important;font-weight:600!important;font-size:28px!important;
  line-height:1.15!important;color:#1a1a1a!important;margin:0 0 6px}
#title-container>span:not(#main-title){display:block;font-family:Inter,system-ui,sans-serif!important;font-size:14px!important;
  line-height:1.45!important;color:#444!important}
#paper-links{display:none;position:absolute;right:14px;bottom:60px;z-index:20;width:340px;background:#fff;border-radius:10px;
  box-shadow:0 6px 24px rgba(31,42,68,.18);padding:14px 16px;font:13px/1.4 Inter,system-ui,sans-serif;color:#1f2a44}
#paper-links .t{font-weight:700;margin:0 22px 10px 0}
#paper-links .n{font-size:12px;color:#6b7489;margin:0 0 8px}
#paper-links a{display:inline-block;margin:0 6px 6px 0;padding:6px 10px;border-radius:7px;background:#1e3a5f;color:#fff;text-decoration:none;font-weight:600}
#paper-links .x{position:absolute;top:8px;right:10px;cursor:pointer;font-size:16px;color:#6b7489;border:0;background:none}
.atlas-back{position:absolute;bottom:14px;right:14px;z-index:10;font:600 13px Inter,system-ui,sans-serif;background:#1e3a5f;
  color:#fff;padding:7px 12px;border-radius:8px;text-decoration:none}
.embedded #title-container,.embedded .atlas-back,.embedded #topic-tree{display:none!important}
@media (max-width:768px){
  #main-title{font-size:21px!important;margin-bottom:4px}
  #title-container{padding:10px 12px!important}
  #title-container>span:not(#main-title){font-size:12.5px!important;line-height:1.4!important}
  .stack.top-left{max-height:44vh!important}
  #topic-tree{display:none!important}
  .stack.bottom-left{display:flex!important}  /* keep the colour menu on phones */
  #colormap-selector-container{max-width:calc(100vw - 32px)!important;box-sizing:border-box}
  #colormap-selector-container .color-map-dropdown,#colormap-selector-container .color-map-selected{max-width:100%!important}
  .atlas-back{bottom:88px;right:12px;padding:7px 11px;font-size:12px}
  body:has(.color-map-options[style*="block"]) .atlas-back{visibility:hidden}
  #paper-links{left:12px;right:12px;width:auto;bottom:132px}
}
"""
plot = datamapplot.create_interactive_plot(
    df[["x", "y"]].to_numpy(), topic, area,
    hover_text=hover,
    extra_point_data=extra,
    marker_color_array=color,
    marker_size_array=np.full(len(df), 2.2),
    hover_text_html_template="""<div style="max-width:440px;font-family:Inter,system-ui,sans-serif">
<div style="font-weight:700;font-size:14px;line-height:1.3">{title}</div>
<div style="opacity:.7;font-size:12px;margin:2px 0 6px">{authors}</div>
<div style="font-size:12px"><b>{conf}</b> · <i>{topic}</i></div>
<div style="font-size:12px;opacity:.85;margin-top:6px">{abstract}</div>
<div style="font-size:11px;margin-top:6px;color:#1f2a44"><b>Click</b> for the link to the paper.</div></div>""",
    on_click="showLinks(`{url}`, `{safe_title}`, `{conf}`)",
    title="NeurIPS and EMNLP, 2025 and 2026, on one map",
    sub_title=f"Each dot is one of {len(df):,} papers; similar papers sit close together. Blue is NeurIPS, orange is EMNLP, "
              "lighter dots are 2025. The menu at the bottom left colours the map by year or broad area.",
    enable_search=True,
    enable_topic_tree=True,
    colormap_rawdata=cmap_raw,
    colormap_metadata=cmap_meta,
    custom_css=CSS,
    custom_html='<a class="atlas-back" href="index.html">← Back to the story</a>'
                '<div id="paper-links"><button class="x" onclick="this.parentNode.style.display=\'none\'" '
                'aria-label="close">×</button><div class="t"></div><div class="n"></div><div class="l"></div></div>',
    custom_js="""
if (window.self !== window.top) document.documentElement.classList.add('embedded');
if (window.innerWidth < 600) {
  const vs = datamap.deckgl.props.initialViewState;
  datamap.deckgl.setProps({initialViewState: {...vs, zoom: vs.zoom - 0.9}});
}
function showLinks(url, title, conf) {
  const box = document.getElementById('paper-links');
  box.querySelector('.t').textContent = title;
  box.querySelector('.n').textContent = conf;
  box.querySelector('.l').innerHTML = '<a href="' + url + '" target="_blank" rel="noopener">Open the paper</a>';
  box.style.display = 'block';
}
""",
    darkmode=False,
    font_family="Inter",
    cluster_boundary_polygons=True,
    initial_zoom_fraction=0.95,
)
plot.save(str(DOCS / "atlas.html"))
page = (DOCS / "atlas.html").read_text().replace("apache-arrow@latest", "apache-arrow@21.2.0").replace("d3@latest", "d3@7")
desc = f"All {len(df):,} NeurIPS and EMNLP papers of 2025 and 2026 on one interactive map."
url = "https://mohammadi.cv/ai-conferences/2026/atlas.html"
head = (f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="{desc}"><link rel="canonical" href="{url}">\n'
        f'<meta property="og:title" content="NeurIPS and EMNLP on one map"><meta property="og:description" content="{desc}">\n'
        f'<meta property="og:url" content="{url}"><meta property="og:type" content="website"><meta property="og:site_name" content="Hadi Mohammadi">\n'
        f'<meta property="og:image" content="https://mohammadi.cv/ai-conferences/2026/figures/map_years.png"><meta name="twitter:card" content="summary_large_image">\n'
        '<link rel="icon" href="/assets/img/favicon.png"><link rel="apple-touch-icon" href="/assets/img/apple-touch-icon.png">\n'
        '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&display=swap" rel="stylesheet" />\n')
assert page.count("<head>") == 1
page = page.replace("<head>", "<head>\n" + head)
old = re.search(r"<title>(.*?)</title>", page, re.S)
t = "<title>NeurIPS and EMNLP on one map — Hadi Mohammadi</title>"
page = page.replace(old.group(0), t) if old else page.replace("</head>", t + "</head>")
# the default colouring is ours (conference and year), but DataMapPlot calls it "Clusters"
assert page.count('"Clusters"') >= 1, "label to rename not found"
page = page.replace('"Clusters"', '"Conference and year"')
(DOCS / "atlas.html").write_text(page)
print("saved", DOCS / "atlas.html", len(df))

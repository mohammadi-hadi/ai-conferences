"""The hexagon grid shared by the overlap figure, the interactive map and the cards.

Pointy-top hexagons, GRID columns across the joint map, odd rows shifted by half a hexagon. Each paper goes to the
hexagon whose centre is nearest, so the counts match the hexagons as they are drawn.
"""
import numpy as np

GRID, MIN_N = 34, 4


def assign(x, y, grid=GRID):
    """Returns hexagon id, centre x and centre y for every point, and the hexagon width."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, x1, y0 = x.min(), x.max(), y.min()
    w = (x1 - x0) / grid
    h = w * np.sqrt(3) / 2
    r0 = np.round((y - y0) / h).astype(int)
    best = np.full(len(x), np.inf)
    row = np.zeros(len(x), int); col = np.zeros(len(x), int)
    for dr in (-1, 0, 1):  # the nearest centre can sit in the row above or below
        r = r0 + dr
        c = np.round((x - x0 - (r % 2) * w / 2) / w).astype(int)
        cx, cy = x0 + c * w + (r % 2) * w / 2, y0 + r * h
        d = np.hypot(x - cx, y - cy)
        take = d < best
        best[take], row[take], col[take] = d[take], r[take], c[take]
    return row * 1000 + col, x0 + col * w + (row % 2) * w / 2, y0 + row * h, w


def category(share):
    """Mostly NeurIPS (share of EMNLP after size correction up to 0.3), shared, or mostly EMNLP (above 0.7)."""
    return None if share is None else ("N" if share <= .3 else ("E" if share > .7 else "S"))

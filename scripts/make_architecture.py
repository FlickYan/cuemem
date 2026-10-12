"""生成 README 用的架构图（浅色 / 深色两版 SVG，写到 docs/assets/）。

uv run python scripts/make_architecture.py
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
WIDTH, HEIGHT = 1120, 500
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    "light": {
        "bg": "#ffffff",
        "box": "#ffffff",
        "stroke": "#d0d7de",
        "ink": "#1f2328",
        "muted": "#59636e",
        "accent_fill": "#ddf4ff",
        "accent_stroke": "#0969da",
        "lane": "#f6f8fa",
        "arrow": "#59636e",
        "hatch": "#bc4c00",
    },
    "dark": {
        "bg": "#0d1117",
        "box": "#0d1117",
        "stroke": "#3d444d",
        "ink": "#f0f6fc",
        "muted": "#9198a1",
        "accent_fill": "#0c2d6b",
        "accent_stroke": "#4493f8",
        "lane": "#151b23",
        "arrow": "#9198a1",
        "hatch": "#f0883e",
    },
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    title: str
    lines: tuple[str, ...] = ()
    accent: bool = False

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


# 读路径（每条消息）
USER = Box(24, 112, 132, 72, "User message", ("text · photo",))
PROBE = Box(190, 104, 176, 88, "Memory probe", ("one L1 scoring pass", "no LLM · 54 features"))
CTRL = Box(
    400,
    96,
    196,
    104,
    "Retrieval controller",
    ("gate + reasoning type", "argmax P̂(correct) − λ·cost", "λ = business knob"),
    accent=True,
)
L0 = Box(636, 48, 196, 44, "L0  profile only")
L1 = Box(636, 104, 196, 44, "L1  fast recall · no LLM")
L2 = Box(636, 160, 196, 44, "L2  plan · re-check · notes")
READER = Box(
    872,
    96,
    220,
    104,
    "Reader LLM / VLM",
    ("answers with citations", "sees original photos", "may call memory_search"),
)
# 写路径（后台）
WRITE = Box(
    24, 352, 236, 88, "Online write", ("append turns · embed · store photos", "no LLM · ~100 ms")
)
STORE = Box(
    300,
    344,
    292,
    104,
    "Memory store",
    (
        "immutable log (SQLite) — source of truth",
        "vectors (Qdrant) · photos by content hash",
        "derived memory, rebuildable",
    ),
)
CONS = Box(
    636,
    344,
    296,
    104,
    "Async consolidation",
    (
        "facts with bi-temporal validity",
        "entities · summaries · photo captions",
        "core profile · forget = delete + replay",
    ),
)


def box_svg(b: Box, t: dict[str, str]) -> str:
    fill = t["accent_fill"] if b.accent else t["box"]
    stroke = t["accent_stroke"] if b.accent else t["stroke"]
    width = 1.6 if b.accent else 1.0
    out = [
        f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="10" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    ]
    n = len(b.lines)
    # 标题与说明行整体垂直居中
    block = 18 + 16 * n
    top = b.cy - block / 2 + 13
    out.append(
        f'<text x="{b.cx}" y="{top}" text-anchor="middle" font-size="14" font-weight="600" '
        f'fill="{t["ink"]}">{escape(b.title)}</text>'
    )
    for i, line in enumerate(b.lines):
        out.append(
            f'<text x="{b.cx}" y="{top + 20 + 16 * i}" text-anchor="middle" font-size="11.5" '
            f'fill="{t["muted"]}">{escape(line)}</text>'
        )
    return "\n".join(out)


def arrow(
    points: list[tuple[float, float]],
    t: dict[str, str],
    *,
    dashed: bool = False,
    color: str | None = None,
    head: bool = True,
) -> str:
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    marker = ' marker-end="url(#head)"' if head and color is None else ""
    if head and color is not None:
        marker = ' marker-end="url(#headHatch)"'
    stroke = color or t["arrow"]
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="1.4"{dash}{marker}/>'


def label(x: float, y: float, text: str, t: dict[str, str], *, anchor: str = "middle",
          color: str | None = None, size: float = 11) -> str:  # fmt: skip
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'fill="{color or t["muted"]}">{escape(text)}</text>'
    )


def render(theme: str) -> str:
    t = THEMES[theme]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">',
        "<defs>",
        f'<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{t["arrow"]}"/></marker>',
        f'<marker id="headHatch" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{t["hatch"]}"/></marker>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{t["bg"]}"/>',
        # 两条泳道
        f'<rect x="8" y="14" width="{WIDTH - 16}" height="222" rx="14" fill="{t["lane"]}"/>',
        f'<rect x="8" y="300" width="{WIDTH - 16}" height="186" rx="14" fill="{t["lane"]}"/>',
        label(24, 36, "READ PATH · every message", t, anchor="start", size=11.5),
        label(112, 322, "WRITE PATH · online + background", t, anchor="start", size=11.5),
    ]
    for b in (USER, PROBE, CTRL, L0, L1, L2, READER, WRITE, STORE, CONS):
        parts.append(box_svg(b, t))
    # 读路径
    parts.append(arrow([(USER.right, USER.cy), (PROBE.x - 2, PROBE.cy)], t))
    parts.append(arrow([(PROBE.right, PROBE.cy), (CTRL.x - 2, CTRL.cy)], t))
    fork = CTRL.right + 18
    for lv in (L0, L1, L2):
        parts.append(
            arrow([(CTRL.right, CTRL.cy), (fork, CTRL.cy), (fork, lv.cy), (lv.x - 2, lv.cy)], t)
        )
    join = L0.right + 20
    for lv in (L0, L1, L2):
        parts.append(
            arrow(
                [(lv.right, lv.cy), (join, lv.cy), (join, READER.cy), (READER.x - 2, READER.cy)],
                t,
                head=lv is L1,
            )
        )
    parts.append(label(L1.cx, L2.bottom + 18, "± attach top-3 original photos", t))
    # 写路径
    parts.append(arrow([(USER.cx, USER.bottom), (USER.cx, WRITE.y - 2)], t))
    parts.append(arrow([(WRITE.right, WRITE.cy), (STORE.x - 2, WRITE.cy)], t))
    parts.append(arrow([(CONS.x, STORE.cy - 14), (STORE.right + 2, STORE.cy - 14)], t))
    parts.append(arrow([(STORE.right, STORE.cy + 14), (CONS.x - 2, STORE.cy + 14)], t))
    parts.append(label((STORE.right + CONS.x) / 2, STORE.cy - 22, "derived", t, size=10.5))
    parts.append(label((STORE.right + CONS.x) / 2, STORE.cy + 32, "log", t, size=10.5))
    # 存储 → 探针 / 检索
    sx = STORE.x + 40
    parts.append(
        arrow([(sx, STORE.y), (sx, 262), (PROBE.cx, 262), (PROBE.cx, PROBE.bottom + 2)], t)
    )
    parts.append(label(sx + 8, 284, "retrieve (probe · L1 · L2)", t, anchor="start"))
    # 逃生通道：Reader 自己查记忆
    hx = READER.cx + 40
    parts.append(
        arrow(
            [
                (hx, READER.bottom),
                (hx, 268),
                (STORE.right - 40, 268),
                (STORE.right - 40, STORE.y - 2),
            ],
            t,
            dashed=True,
            color=t["hatch"],
        )
    )
    parts.append(label(hx + 8, 228, "memory_search", t, anchor="start", color=t["hatch"]))
    parts.append(label(hx + 8, 242, "(escape hatch)", t, anchor="start", color=t["hatch"]))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = OUT / f"architecture-{theme}.svg"
        path.write_text(render(theme), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()

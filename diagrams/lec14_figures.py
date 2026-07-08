"""Generate figures for Lecture 14: Tokenization & Pretraining.

One new matplotlib figure supporting the Andrew-Ng output-first build-up:

  tokenizer_pipeline.svg  — THE OBJECT FIRST: raw text -> subword tokens ->
  integer token IDs -> rows of the embedding table -> Transformer. Makes the
  abstract claim "a tokenizer outputs integer IDs" concrete and visual.

Palette + rcParams copied from lec00_figures.py so this sits beside the
hand-authored lec14 SVGs (bpe_merges, mlm_vs_causal, ...) without clashing.
Outputs SVG into figures/lec14/svg/.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) -------------------
PAPER = "#F7F3E9"
PAPER_ALT = "#EFEADA"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
RUST_FILL = "#EAD3C5"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Georgia", "DejaVu Serif"],
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec14" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _box(ax, cx, cy, w, h, text, fc, ec, fontsize=12, tc=INK, lw=1.2,
         weight="normal", family=None):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.9",
        fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=tc, zorder=4, weight=weight, family=family)


def _harrow(ax, x0, x1, y, color=MUTED, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=15, color=color, lw=lw, zorder=2))


def _varrow(ax, x, y0, y1, color=MUTED, lw=1.5):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=12, color=color, lw=lw, zorder=2))


# ---- tokenizer pipeline: text -> tokens -> integer IDs -> embedding rows ------
def tokenizer_pipeline():
    fig, ax = plt.subplots(figsize=(12.4, 3.5))
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 42)
    ax.axis("off")

    tokens = ["learning", "un", "believ", "able"]
    ids = ["4673", "555", "6667", "719"]
    fills = [SAGE_FILL, RUST_FILL, RUST_FILL, RUST_FILL]
    edges = [SAGE, RUST, RUST, RUST]

    ytitle = 38.5
    ytok = 27.5
    yid = 15.5

    # -- stage 1: raw text ----------------------------------------------------
    ax.text(12, ytitle, "raw text", ha="center", fontsize=12,
            color=MUTED, style="italic")
    _box(ax, 12, 21.5, 21, 8.5, '"learning\nunbelievable"',
         PAPER_ALT, INK, fontsize=12.5)
    _harrow(ax, 23.5, 30.5, 21.5)

    # -- stage 2: subword tokens ---------------------------------------------
    ax.text(56.5, ytitle, "subword tokens", ha="center", fontsize=12,
            color=MUTED, style="italic")
    txc = [38, 51, 64, 77]
    tw = 12.0
    for cx, t, fc, ec in zip(txc, tokens, fills, edges):
        _box(ax, cx, ytok, tw, 5.6, t, fc, ec, fontsize=11.5)
    # micro-labels above the two kinds of token
    ax.text(38, 32.0, "frequent word\n$\\rightarrow$ one token", ha="center",
            fontsize=8.3, color=SAGE, linespacing=1.05)
    ax.text(70.5, 32.0, r"rare word $\rightarrow$ a few subwords", ha="center",
            fontsize=8.3, color=RUST)

    # -- stage 3: integer IDs (aligned under the tokens) ----------------------
    for cx, i, ec in zip(txc, ids, edges):
        _varrow(ax, cx, ytok - 2.9, yid + 2.9)
        _box(ax, cx, yid, tw, 5.2, i, PAPER, INK, fontsize=12,
             weight="bold", family="monospace")
    ax.text(56.5, 9.6, "integer token IDs — the only thing the model sees",
            ha="center", fontsize=10, color=INK, style="italic")

    # arrow from the ID row into the embedding table
    _harrow(ax, 83.5, 91.5, yid)

    # -- stage 4: embedding table (ID = row index) ---------------------------
    ax.text(108, ytitle, "embedding table", ha="center", fontsize=12,
            color=MUTED, style="italic")
    ex0, ex1 = 93.5, 122.0
    ey0, ey1 = 12.0, 34.0
    nrows = 7
    rh = (ey1 - ey0) / nrows
    # base table
    ax.add_patch(FancyBboxPatch((ex0, ey0), ex1 - ex0, ey1 - ey0,
                 boxstyle="round,pad=0.1,rounding_size=0.6",
                 fc=PAPER, ec=MUTED, lw=1.3, zorder=3))
    # row separators
    for k in range(1, nrows):
        yk = ey0 + k * rh
        ax.plot([ex0, ex1], [yk, yk], color="#D8D2C0", lw=0.8, zorder=4)
    # highlight two rows as the ones an ID points to
    for r, lab in [(2, "555"), (4, "4673")]:
        yb = ey0 + r * rh
        ax.add_patch(FancyBboxPatch((ex0 + 0.4, yb + 0.4),
                     ex1 - ex0 - 0.8, rh - 0.8,
                     boxstyle="square,pad=0", fc=RUST_FILL, ec=RUST,
                     lw=1.1, zorder=5))
        ax.text(ex0 + 3.0, yb + rh / 2, f"row {lab}", ha="left", va="center",
                fontsize=8.6, color=RUST, family="monospace", zorder=6)
        ax.text(ex1 - 2.5, yb + rh / 2, "[ · · · ]", ha="right", va="center",
                fontsize=8.6, color=MUTED, zorder=6)
    ax.text(108, 8.2, "one row per vocab entry — the ID is the row number",
            ha="center", fontsize=9, color=MUTED, style="italic")

    fig.tight_layout()
    save(fig, "tokenizer_pipeline.svg")


if __name__ == "__main__":
    tokenizer_pipeline()
    print("done.")

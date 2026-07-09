"""
figs_a.py — Beamer/metropolis-style SVG figures for Deep Learning Lecture 1.

Generates 9 transparent-background SVGs into lecture1/figures/.
Run from the repository root:  python3 lecture1/diagrams/figs_a.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

INK = '#23373B'; ACC = '#EB811B'; TEAL = '#2C7A7B'; GREEN = '#14B03D'; MUTED = '#6E7F82'

mpl.rcParams.update({
    'figure.facecolor': 'none', 'axes.facecolor': 'none',
    'savefig.facecolor': 'none', 'savefig.transparent': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Fira Sans', 'DejaVu Sans', 'Arial'],
    'text.color': INK, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 1.0,
    'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
})


def save(fig, name):
    fig.savefig(f'lecture1/figures/{name}.svg', bbox_inches='tight', transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. net_to_distribution.svg — tiny net -> distribution
# ---------------------------------------------------------------------------
def net_to_distribution():
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    layers = [3, 4, 4]           # input, hidden1, hidden2
    xs = [1.0, 3.0, 5.0]
    node_r = 0.22
    coords = []
    for x, n in zip(xs, layers):
        ys = np.linspace(1.2, 4.8, n)
        coords.append([(x, y) for y in ys])

    # connections
    for a, b in zip(coords[:-1], coords[1:]):
        for (x0, y0) in a:
            for (x1, y1) in b:
                ax.plot([x0, x1], [y0, y1], color=MUTED, lw=0.5, alpha=0.35, zorder=1)

    # nodes
    palette = [INK, TEAL, TEAL]
    for layer, col in zip(coords, palette):
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), node_r, facecolor='white',
                                 edgecolor=col, lw=1.4, zorder=3))

    # arrow from net to distribution
    ax.add_patch(FancyArrowPatch((5.5, 3.0), (6.7, 3.0),
                 arrowstyle='-|>', mutation_scale=16, color=INK, lw=1.4))

    # inset bell curve (the "distribution")
    bell = ax.inset_axes([0.70, 0.18, 0.28, 0.64])
    xx = np.linspace(-4, 4, 300)
    yy = np.exp(-xx ** 2 / 2)
    bell.plot(xx, yy, color=ACC, lw=2.0)
    bell.fill_between(xx, yy, color=ACC, alpha=0.18)
    bell.axis('off')

    ax.text(1.0, 5.4, 'input', ha='center', fontsize=11, color=MUTED)
    ax.text(4.0, 5.4, 'hidden', ha='center', fontsize=11, color=MUTED)
    ax.text(8.3, 5.0, r'$p_\theta(y\mid x)$', ha='center', fontsize=13, color=ACC)
    save(fig, 'net_to_distribution')


# ---------------------------------------------------------------------------
# 2. density_area.svg — density with shaded P(a<=Y<=b)
# ---------------------------------------------------------------------------
def density_area():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.linspace(-4, 5, 500)
    # gentle bimodal-ish smooth density
    y = 0.6 * np.exp(-(x - 0.2) ** 2 / 2.0) + 0.4 * np.exp(-(x - 2.6) ** 2 / 1.2)
    ax.plot(x, y, color=INK, lw=2.0)

    a, b = -0.6, 2.2
    mask = (x >= a) & (x <= b)
    ax.fill_between(x[mask], y[mask], color=ACC, alpha=0.30)

    ymax = y.max()
    ax.set_ylim(0, ymax * 1.25)
    ax.set_xlim(-4, 5)
    ax.set_yticks([])
    ax.set_xticks([a, b])
    ax.set_xticklabels(['a', 'b'])
    ax.tick_params(axis='x', labelsize=14)
    ax.spines['left'].set_visible(False)

    ax.annotate(r'$P(a \leq Y \leq b) = \mathrm{area}$',
                xy=(0.9, 0.22), xytext=(1.6, 0.62),
                fontsize=13, color=INK,
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
    ax.set_xlabel('y', fontsize=13)
    save(fig, 'density_area')


# ---------------------------------------------------------------------------
# 3. uniform.svg — uniform density with -log 0 = +inf outside support
# ---------------------------------------------------------------------------
def uniform():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    a, b = 1.0, 4.0
    h = 1.0 / (b - a)

    # zero segments outside support
    ax.plot([-1.5, a], [0, 0], color=INK, lw=2.0)
    ax.plot([b, 6.5], [0, 0], color=INK, lw=2.0)
    # flat top
    ax.plot([a, b], [h, h], color=INK, lw=2.0)
    ax.fill_between([a, b], [h, h], color=ACC, alpha=0.20)
    # vertical drops (dashed)
    ax.plot([a, a], [0, h], color=INK, lw=1.2, ls='--')
    ax.plot([b, b], [0, h], color=INK, lw=1.2, ls='--')

    ax.set_xlim(-1.5, 6.5)
    ax.set_ylim(-0.05, h * 1.9)
    ax.set_xticks([a, b])
    ax.set_xticklabels(['a', 'b'])
    ax.tick_params(axis='x', labelsize=14)
    ax.set_yticks([h])
    ax.set_yticklabels([r'$\frac{1}{b-a}$'])
    ax.tick_params(axis='y', labelsize=14)

    ax.annotate(r'$-\log 0 = +\infty$', xy=(5.1, 0.02), xytext=(4.2, h * 1.35),
                fontsize=13, color=ACC, ha='center',
                arrowprops=dict(arrowstyle='-|>', color=ACC, lw=1.2))
    ax.text(-0.6, h * 1.35, r'$-\log 0 = +\infty$', fontsize=13,
            color=ACC, ha='center')
    ax.set_xlabel('y', fontsize=13)
    save(fig, 'uniform')


# ---------------------------------------------------------------------------
# 4. gaussian_params.svg — changing mu | changing sigma
# ---------------------------------------------------------------------------
def gaussian_params():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    x = np.linspace(-6, 6, 500)

    def g(x, mu, sig):
        return np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))

    cols = [TEAL, INK, ACC]
    # left: changing mu
    for mu, c in zip([-2.2, 0.0, 2.2], cols):
        axes[0].plot(x, g(x, mu, 1.0), color=c, lw=2.0)
    axes[0].set_title(r'changing $\mu$', fontsize=14, color=INK, pad=8)

    # right: changing sigma
    for sig, c in zip([0.7, 1.3, 2.2], cols):
        axes[1].plot(x, g(x, 0.0, sig), color=c, lw=2.0)
    axes[1].set_title(r'changing $\sigma$', fontsize=14, color=INK, pad=8)

    for ax in axes:
        ax.set_yticks([]); ax.set_xticks([0])
        ax.set_xticklabels(['0'])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, 'gaussian_params')


# ---------------------------------------------------------------------------
# 5. gaussian_to_square.svg — 4-panel transform strip
# ---------------------------------------------------------------------------
def gaussian_to_square():
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.7))
    x = np.linspace(-3, 3, 400)
    mu = 0.0

    p = np.exp(-(x - mu) ** 2 / 2) / np.sqrt(2 * np.pi)   # density
    logp = -0.5 * (x - mu) ** 2 - 0.5 * np.log(2 * np.pi)  # log density
    neglog = -logp                                          # neg log density
    sq = (x - mu) ** 2                                      # squared error

    panels = [
        (p, r'$p(y)$', ACC, True),
        (logp, r'$\log p(y)$', TEAL, False),
        (neglog, r'$-\log p(y)$', INK, False),
        (sq, r'$(y-\mu)^2$', GREEN, False),
    ]
    for ax, (yv, title, col, fill) in zip(axes, panels):
        ax.plot(x, yv, color=col, lw=2.0)
        if fill:
            ax.fill_between(x, yv, color=col, alpha=0.15)
        ax.set_title(title, fontsize=13, color=INK, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

    # captions between panels (figure coords)
    for xc, txt in zip([0.275, 0.505, 0.735], [r'$\log$', r'$\times(-1)$', 'drop\nconst']):
        fig.text(xc, 0.5, r'$\rightarrow$', ha='center', va='center',
                 fontsize=18, color=MUTED)
        fig.text(xc, 0.80, txt, ha='center', va='center', fontsize=10, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(wspace=0.45)
    save(fig, 'gaussian_to_square')


# ---------------------------------------------------------------------------
# 6. mvn_triptych.svg — 2-D Gaussian contours
# ---------------------------------------------------------------------------
def mvn_triptych():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    lim = 3.2
    g = np.linspace(-lim, lim, 220)
    X, Y = np.meshgrid(g, g)
    pos = np.dstack((X, Y))

    def pdf(Sigma):
        inv = np.linalg.inv(Sigma)
        det = np.linalg.det(Sigma)
        d = np.einsum('...i,ij,...j->...', pos, inv, pos)
        return np.exp(-0.5 * d) / (2 * np.pi * np.sqrt(det))

    sigmas = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), r'$\Sigma = \tau^2 I$'),
        (np.array([[1.8, 0.0], [0.0, 0.55]]), r'diagonal $\Sigma$'),
        (np.array([[1.4, 0.95], [0.95, 1.0]]), r'full $\Sigma$'),
    ]
    for ax, (S, title) in zip(axes, sigmas):
        Z = pdf(S)
        ax.contour(X, Y, Z, levels=6, colors=TEAL, linewidths=1.1)
        ax.plot(0, 0, marker='+', color=ACC, ms=9, mew=1.8)
        ax.set_title(title, fontsize=13, color=INK, pad=6)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    fig.tight_layout()
    save(fig, 'mvn_triptych')


# ---------------------------------------------------------------------------
# 7. regression_conditionals.svg — curve + conditional Gaussians + scatter
# ---------------------------------------------------------------------------
def regression_conditionals():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rng = np.random.default_rng(3)

    def f(x):
        return 1.4 * np.sin(0.9 * x) + 0.25 * x

    xg = np.linspace(0, 10, 400)
    ax.plot(xg, f(xg), color=INK, lw=2.2, zorder=3, label=r'$f(x)$')

    sigma = 0.55
    # noisy scatter
    xs = rng.uniform(0.3, 9.7, 22)
    ys = f(xs) + rng.normal(0, sigma, xs.size)
    ax.scatter(xs, ys, s=22, color=MUTED, alpha=0.75, zorder=2, edgecolors='none')

    # conditional bells at 4 x-locations (rotated 90 deg)
    for x0 in [1.6, 4.0, 6.4, 8.8]:
        yv = np.linspace(f(x0) - 3 * sigma, f(x0) + 3 * sigma, 120)
        amp = 0.95
        bell = amp * np.exp(-(yv - f(x0)) ** 2 / (2 * sigma ** 2))
        ax.plot(x0 + bell, yv, color=TEAL, lw=1.6, zorder=4)
        ax.fill_betweenx(yv, x0, x0 + bell, color=TEAL, alpha=0.18, zorder=1)
        ax.plot([x0, x0], [yv[0], yv[-1]], color=TEAL, lw=0.8, ls=':', zorder=1)

    ax.set_xlim(0, 10.6)
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_yticks([]); ax.set_xticks([])
    ax.text(9.0, f(8.8) + 3 * sigma + 0.15, r'$y \sim \mathcal{N}(f(x),\sigma^2)$',
            fontsize=12, color=TEAL, ha='center')
    save(fig, 'regression_conditionals')


# ---------------------------------------------------------------------------
# 8. residual_losses.svg — four -log p(r) curves
# ---------------------------------------------------------------------------
def residual_losses():
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    r = np.linspace(-4, 4, 500)

    gauss = 0.5 * r ** 2                       # Gaussian
    lap = np.abs(r)                            # Laplace
    nu = 2.0
    stud = np.log(1 + r ** 2 / nu)             # Student-t (up to const)

    ax.plot(r, gauss, color=ACC, lw=2.2, label=r'Gaussian  $r^2$')
    ax.plot(r, lap, color=TEAL, lw=2.2, label=r'Laplace  $|r|$')
    ax.plot(r, stud, color=GREEN, lw=2.2, label=r'Student-$t$  $\log(1+r^2/\nu)$')

    # Uniform hard barrier: flat inside [-c, c], walls at +/- c
    c = 2.6
    ax.plot([-c, c], [0, 0], color=MUTED, lw=2.2, label=r'Uniform (barrier)')
    ax.plot([-c, -c], [0, 8], color=MUTED, lw=2.2, ls='--')
    ax.plot([c, c], [0, 8], color=MUTED, lw=2.2, ls='--')

    ax.set_ylim(0, 8)
    ax.set_xlim(-4, 4)
    ax.set_xlabel('residual  r', fontsize=13)
    ax.set_ylabel(r'$-\log p(r)$', fontsize=13)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.legend(frameon=False, fontsize=10.5, loc='upper center', ncol=1,
              handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    save(fig, 'residual_losses')


# ---------------------------------------------------------------------------
# 9. softmax_pipeline.svg — input -> net -> logits -> softmax -> probs
# ---------------------------------------------------------------------------
def softmax_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axis('off'); ax.set_xlim(0, 14); ax.set_ylim(0, 6)

    def box(x0, y0, w, h, label, fc='white', ec=INK):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                     boxstyle='round,pad=0.06,rounding_size=0.18',
                     facecolor=fc, edgecolor=ec, lw=1.6, zorder=2))
        ax.text(x0 + w / 2, y0 + h / 2, label, ha='center', va='center',
                fontsize=12, color=INK, zorder=3)

    def arrow(x0, x1, y=3.0):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
                     mutation_scale=15, color=INK, lw=1.5, zorder=1))

    # input
    box(0.2, 2.3, 1.5, 1.4, r'$x$', fc='#EFEEEB')
    arrow(1.8, 2.6)
    # neural net box
    box(2.7, 1.9, 2.1, 2.2, 'neural\nnet', fc='white')
    arrow(4.9, 5.7)

    # logits inset (can be negative)
    logits = [1.8, -0.9, 0.4]
    ax_l = ax.inset_axes([5.85, 1.7, 2.0, 2.6], transform=ax.transData)
    cols = [ACC, TEAL, GREEN]
    ax_l.bar([0, 1, 2], logits, color=cols, width=0.62)
    ax_l.axhline(0, color=INK, lw=0.9)
    ax_l.set_ylim(-1.6, 2.4)
    ax_l.set_xticks([]); ax_l.set_yticks([])
    for s in ax_l.spines.values():
        s.set_visible(False)
    ax.text(6.85, 4.55, 'logits', ha='center', fontsize=11, color=MUTED)

    arrow(8.0, 8.9)
    # softmax box
    box(9.0, 1.9, 2.0, 2.2, 'softmax', fc='white', ec=ACC)
    arrow(11.1, 11.9)

    # probabilities inset (positive, sum to 1)
    z = np.array(logits)
    p = np.exp(z) / np.exp(z).sum()
    ax_p = ax.inset_axes([11.95, 1.7, 2.0, 2.6], transform=ax.transData)
    ax_p.bar([0, 1, 2], p, color=cols, width=0.62)
    ax_p.axhline(0, color=INK, lw=0.9)
    ax_p.set_ylim(0, 1.0)
    ax_p.set_xticks([]); ax_p.set_yticks([])
    for s in ax_p.spines.values():
        s.set_visible(False)
    ax.text(12.95, 4.55, r'probabilities ($\Sigma=1$)', ha='center',
            fontsize=10.5, color=MUTED)
    save(fig, 'softmax_pipeline')


if __name__ == '__main__':
    net_to_distribution()
    density_area()
    uniform()
    gaussian_params()
    gaussian_to_square()
    mvn_triptych()
    regression_conditionals()
    residual_losses()
    softmax_pipeline()
    print('done: 9 figures written to lecture1/figures/')

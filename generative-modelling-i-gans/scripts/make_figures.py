"""Every data figure and every quoted number in the GAN lecture comes from this script.

    python scripts/make_figures.py

Reads the checkpoints written by train_models.py and writes
figures/data/*.svg|png and figures/data/numbers.json.
The plotting code is the GAN part of the Generative Modelling I (VAE) lecture's make_figures.py.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_models import ConvD, load_mnist, toy_data  # noqa: E402

MODELS = ROOT / "assets" / "models"
FIG = ROOT / "figures" / "data"
FIG.mkdir(parents=True, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NUM = {}  # numbers quoted on slides
DIGIT3 = 90  # MNIST test digit #90: the running "3" of the VAE lecture (chosen there as a typical 3)

C = dict(x="#245EDB", x_t="#E4ECFF", enc="#8B2CDE", enc_t="#F1E5FC", dec="#0F766E", dec_t="#D9F2EF",
         z="#AA4E08", z_t="#FCEFD9", loss="#BE123C", loss_t="#FDE2E7", disc="#334155", disc_t="#E2E8F0",
         ink="#14171F", ink2="#4A5160", ink3="#6B7280", line="#D9DDE5", grid="#ECEEF2")

plt.rcParams.update({
    "font.family": "Avenir", "font.size": 15, "svg.fonttype": "path", "mathtext.fontset": "cm", "axes.unicode_minus": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9AA1AD",
    "axes.labelcolor": C["ink2"], "axes.labelsize": 15, "xtick.color": C["ink3"], "ytick.color": C["ink3"],
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 1.0,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "legend.frameon": False, "legend.fontsize": 14, "lines.linewidth": 2.4,
})


def save(fig, name):
    path = FIG / f"{name}.svg"
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    svg = path.read_text()
    svg = svg.replace("'Avenir'", "'Avenir Next', 'Avenir', 'Segoe UI', sans-serif")
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    path.write_text(svg)


def r(v, k=2):
    return float(round(float(v), k))


# ----------------------------------------------------------------------------- image helpers
INK = np.array([0x14, 0x17, 0x1F]) / 255.0
PAPER = np.array([1.0, 1.0, 1.0])


def to_rgb(img):
    img = np.clip(np.asarray(img, dtype=float), 0, 1)[..., None]
    return PAPER * (1 - img) + INK * img


def grid_png(imgs, rows, cols, name, scale=4, pad=2, pad_rgb=(0.92, 0.93, 0.95)):
    imgs = np.asarray(imgs).reshape(-1, 28, 28)
    H = rows * 28 + (rows + 1) * pad
    W = cols * 28 + (cols + 1) * pad
    out = np.ones((H, W, 3)) * np.array(pad_rgb)
    for i in range(min(len(imgs), rows * cols)):
        rr, cc = divmod(i, cols)
        y0, x0 = pad + rr * (28 + pad), pad + cc * (28 + pad)
        out[y0:y0 + 28, x0:x0 + 28] = to_rgb(imgs[i])
    out = np.repeat(np.repeat(out, scale, 0), scale, 1)
    plt.imsave(FIG / f"{name}.png", out)


def tile_png(img, name, scale=8):
    out = np.repeat(np.repeat(to_rgb(np.asarray(img).reshape(28, 28)), scale, 0), scale, 1)
    plt.imsave(FIG / f"{name}.png", out)


def col_panel(samples, d, grid, label, name, xs, mix, xlim=(-5, 5), first=False, note=None):
    """One column of a small-multiples film: data and fakes on top, D(x) below.
    Fixed figure size and axes positions (no tight bbox), so neighbouring columns line up exactly."""
    fig = plt.figure(figsize=(2.6, 4.0))
    a1 = fig.add_axes([0.22, 0.49, 0.74, 0.41])
    a2 = fig.add_axes([0.22, 0.17, 0.74, 0.27], sharex=a1)
    s = np.clip(np.asarray(samples), *xlim)
    a1.fill_between(xs, 0, mix, color=C["x_t"])
    a1.plot(xs, mix, color=C["x"], lw=1.5)
    a1.hist(s, bins=np.linspace(*xlim, 61), density=True, histtype="stepfilled", color=C["dec"], alpha=.55, lw=0)
    a1.set_ylim(0, 1.0)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    plt.setp(a1.get_xticklabels(), visible=False)
    a1.set_title(label, loc="left", fontsize=16, fontweight="bold", color=C["ink"], pad=4)
    if note:
        a1.text(xlim[1], 0.97, note, ha="right", va="top", fontsize=13, color=C["ink3"])
    a2.plot(grid, d, color=C["disc"], lw=2.2)
    a2.axhline(0.5, color=C["ink3"], lw=1, ls=(0, (3, 3)))
    a2.set_ylim(0, 1.05)
    a2.set_yticks([0, 0.5, 1])
    a2.set_xlim(*xlim)
    a2.set_xticks([-8, 0, 8] if xlim[1] > 6 else [-4, 0, 4])
    a2.set_xlabel(r"$x$", labelpad=1)
    a2.grid(axis="y", color=C["grid"])
    if first:
        a2.set_ylabel(r"$D(x)$")
    else:
        plt.setp(a2.get_yticklabels(), visible=False)
    path = FIG / f"{name}.svg"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    svg = path.read_text().replace("'Avenir'", "'Avenir Next', 'Avenir', 'Segoe UI', sans-serif")
    path.write_text(re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S))


def main():
    meta = json.loads((MODELS / "training.json").read_text())

    # ---------------- the 1-D toy data: two equal bumps at -2 and +2, std 0.5
    xs = np.linspace(-5, 5, 600)

    def npdf(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))

    mix = 0.5 * npdf(xs, -2, .5) + 0.5 * npdf(xs, 2, .5)

    # ---------------- GAN: saturation of the two generator losses
    a = np.linspace(-7, 3, 400)
    D = 1 / (1 + np.exp(-a))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    a1.plot(a, np.log(1 - D), color=C["ink2"], ls=(0, (5, 3)))
    a1.plot(a, -np.log(D), color=C["dec"])
    a1.set_title("generator loss", loc="left", fontsize=15, color=C["ink2"])
    a1.text(-4.2, 5.9, r"$-\log D$  (used in practice)", color=C["dec"], fontsize=14)
    a1.text(-4.3, -2.45, r"$\log(1 - D)$  (minimax)", color=C["ink2"], fontsize=14)
    a1.set_ylim(-3.4, 7.6)
    a2.plot(a, -D, color=C["ink2"], ls=(0, (5, 3)))
    a2.plot(a, D - 1, color=C["dec"])
    a2.set_title(r"its slope with respect to $a$", loc="left", fontsize=15, color=C["ink2"])
    a0 = math.log(0.01 / 0.99)
    for ax_ in (a1, a2):
        ax_.axvline(a0, color=C["disc"], lw=1.4, ls=(0, (2, 2)))
        ax_.set_xlabel(r"D's logit $a$ for a fake,  $D = \mathrm{sigmoid}(a)$")
        ax_.grid(color=C["grid"])
    a2.plot([a0], [-0.01], "o", color=C["ink2"], ms=8)
    a2.plot([a0], [-0.99], "o", color=C["dec"], ms=8)
    a2.text(a0 + 0.3, -0.13, r"$-0.01$", ha="left", fontsize=16, color=C["ink2"])
    a2.text(a0 - 0.25, -0.93, r"$-0.99$", ha="right", fontsize=16, color=C["dec"])
    a1.text(a0 + 0.15, 7.0, r"$D = 0.01$", color=C["disc"], fontsize=14)
    save(fig, "saturation")

    # ---------------- GAN on the 1-D toy data
    runs = json.loads((MODELS / "toy_gan_runs.json").read_text())
    good = max(runs, key=lambda q_: min(q_["left"], q_["right"]))
    bad = json.loads((MODELS / "toy_gan_collapse.json").read_text())  # generator with 4 hidden units
    NUM["toygan"] = dict(good_seed=good["seed"], bad_seed=bad["seed"], steps=good["steps"], runs=len(runs),
                         good=dict(left=r(good["left"], 2), right=r(good["right"], 2)),
                         bad=dict(left=r(bad["left"], 2), right=r(bad["right"], 2)),
                         all=[dict(seed=q_["seed"], left=r(q_["left"], 2), right=r(q_["right"], 2)) for q_ in runs])
    grid1 = np.array(good["grid"])
    dbad = np.array(bad["record"][str(bad["steps"])]["d"])
    NUM["collapse"] = {k: r(np.interp(v, grid1, dbad), 2) for k, v in (("D_A", -2.0), ("D_mid", 0.0), ("D_B", 2.0))}

    # step 25: how many runs had (almost) all their fakes on one bump?
    def on_one_bump(run, step):
        s = np.array(run["record"][str(step)]["samples"])
        return max(np.mean(np.abs(s + 2) < 1.0), np.mean(np.abs(s - 2) < 1.0))
    NUM["toygan"]["step25_one_bump_runs"] = int(sum(on_one_bump(q_, 25) > 0.9 for q_ in runs))

    # one worked example from the recorded run at step 25: a real point at the centre of bump B, and
    # the fake that landed closest to 0, in the empty gap between the bumps. Losses use the rounded D
    # values shown on the slide, so the arithmetic on the slide checks out exactly.
    rec25 = good["record"]["25"]
    d25 = np.array(rec25["d"])
    s25 = np.array(rec25["samples"])
    x_real = 2.0
    x_fake = float(s25[np.argmin(np.abs(s25))])
    D_real = r(np.interp(x_real, grid1, d25), 2)
    D_fake = r(np.interp(x_fake, grid1, d25), 2)
    slope = r(np.interp(x_fake, grid1, np.gradient(d25, grid1)), 2)
    l_real, l_fake = -math.log(D_real), -math.log(1 - D_fake)
    NUM["dloss"] = dict(step=25, x_real=r(x_real, 1), x_fake=r(x_fake, 2), D_real=D_real, D_fake=D_fake,
                        l_real=r(l_real, 3), l_fake=r(l_fake, 3), total=r(r(l_real, 3) + r(l_fake, 3), 3),
                        lg_fake=r(-math.log(D_fake), 2), inv_D=r(-1 / D_fake, 2), slope=slope,
                        dLdx=r(-1 / D_fake * slope, 2))
    # the collapsed run at its last step: D at the two bump centres and between them, and how flat D is
    # where the fakes sit (central 90% of them)
    sb = np.array(bad["record"][str(bad["steps"])]["samples"])
    lo, hi = np.quantile(sb, [0.05, 0.95])
    gb = np.abs(np.gradient(dbad, grid1))[(grid1 >= lo) & (grid1 <= hi)]
    NUM["collapse"]["slope_A"] = r(gb.max(), 2)

    def gan_panel(run, step, name, arrows=False, marks=()):
        rec = run["record"][str(step)]
        s = np.array(rec["samples"])
        d = np.array(rec["d"])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
        a1.fill_between(xs, 0, mix, color=C["x_t"])
        a1.plot(xs, mix, color=C["x"], lw=1.5)
        a1.hist(np.clip(s, -5, 5), bins=np.linspace(-5, 5, 61), density=True, histtype="stepfilled",
                color=C["dec"], alpha=.55, lw=0)
        a1.set_ylim(0, 1.0)
        a1.set_yticks([])
        a1.spines["left"].set_visible(False)
        a1.text(-4.9, 0.85, f"step {step}", fontsize=16, fontweight="bold", color=C["ink"])
        if arrows:
            dD = np.gradient(d, grid1)
            pts = np.sort(s)[np.linspace(40, len(s) - 40, 14).astype(int)]
            gmax = max(abs(np.interp(pts, grid1, dD)).max(), 1e-9)
            for i_, v in enumerate(pts):
                gv = np.interp(v, grid1, dD)
                y0 = 0.55 + 0.06 * (i_ % 4)
                a1.plot([v], [y0], "o", color=C["dec"], ms=6, zorder=4)
                a1.annotate("", xy=(v + 0.9 * gv / gmax, y0), xytext=(v, y0),
                            arrowprops=dict(arrowstyle="-|>", color=C["loss"], lw=2.2, mutation_scale=16), zorder=5)
        a2.plot(grid1, d, color=C["disc"])
        for mx, my, col, txt, ha in marks:  # worked examples: one point, its D value
            a2.plot([mx, mx], [0, my], color=col, lw=1.4, ls=(0, (2, 2)))
            a2.plot([mx], [my], "o", color=col, ms=9, zorder=5, mec="white", mew=1.5)
            ty = 0.16 if my > 0.5 else 0.72  # below a high point, above a low one: never on the curve
            a2.text(mx + (0.15 if ha == "left" else -0.15), ty, txt, color=col, fontsize=15, ha=ha, va="center",
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=6)
            a1.annotate("", xy=(mx, 0.02), xytext=(mx, 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=14), zorder=5)
        a2.axhline(0.5, color=C["ink3"], lw=1, ls=(0, (3, 3)))
        a2.set_ylim(0, 1.02)
        a2.set_yticks([0, 0.5, 1])
        a2.set_ylabel(r"$D(x)$")
        a2.set_xlim(-5, 5)
        a2.set_xlabel(r"$x$")
        a2.grid(axis="y", color=C["grid"])
        save(fig, name)

    for st in (0, 25, 100, 250, 1000, 3000):
        gan_panel(good, st, f"gan1d-{st}")
    gan_panel(good, 25, "gan1d-arrows", arrows=True)
    dl = NUM["dloss"]
    gan_panel(good, 25, "gan1d-dloss", marks=((x_real, D_real, C["x"], f"real: D = {dl['D_real']:.2f}", "left"),
                                              (x_fake, D_fake, C["dec"], f"fake: D = {dl['D_fake']:.2f}", "right")))
    gan_panel(bad, 3000, "gan1d-collapse")
    for i_, st in enumerate((25, 100, 250, 1000, 3000)):
        rec = good["record"][str(st)]
        col_panel(rec["samples"], rec["d"], grid1, f"step {st}", f"film-game-{st}", xs, mix, first=i_ == 0)

    # ---------------- each half of the game on its own (train_models.train_toy_one_player)
    one = json.loads((MODELS / "toy_one_player.json").read_text())
    g1 = np.array(one["grid"])
    for i_, (st, rec) in enumerate(sorted(one["d_only"].items(), key=lambda kv: int(kv[0]))):
        if int(st) == 100:
            continue
        col_panel(rec["samples"], rec["d"], g1, f"{st} D steps", f"film-donly-{st}", xs, mix, first=i_ == 0)
    xw = np.linspace(-10, 10, 800)
    mixw = 0.5 * npdf(xw, -2, .5) + 0.5 * npdf(xw, 2, .5)
    for i_, (st, rec) in enumerate(sorted(one["g_only"].items(), key=lambda kv: int(kv[0]))):
        col_panel(rec["samples"], rec["d"], g1, f"{st} G steps", f"film-gonly-{st}", xw, mixw, xlim=(-10, 10), first=i_ == 0)
    dfin = np.array(one["d_only"][str(one["d_steps"])]["d"])
    sg = np.array(one["g_only"][str(one["g_steps"])]["samples"])
    NUM["oneplayer"] = dict(d_steps=one["d_steps"], g_steps=one["g_steps"],
                            D0=r(np.interp(0.0, g1, dfin), 2), D2=r(np.interp(2.0, g1, dfin), 2), D4=r(np.interp(4.0, g1, dfin), 2),
                            Dm4=r(np.interp(-4.0, g1, dfin), 2),
                            beyond3=r(np.mean(np.abs(sg) > 3), 2), beyond10=r(np.mean(np.abs(sg) > 10), 2),
                            near_bumps=r(np.mean(np.minimum(np.abs(sg + 2), np.abs(sg - 2)) < 1.0), 2))

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.2), sharey=True)
    for ax_, run, title in ((axs[0], good, "run that covers both bumps"), (axs[1], bad, "run that collapses to one bump")):
        Ls = np.array(run["losses"])
        k = 50
        sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")  # noqa: E731
        ax_.plot(sm(Ls[:, 0]), color=C["disc"], lw=2)
        ax_.plot(sm(Ls[:, 1]), color=C["dec"], lw=2)
        ax_.set_title(title, loc="left", fontsize=15, color=C["ink2"])
        ax_.set_xlabel("step")
        ax_.grid(color=C["grid"])
    for ax_, run in ((axs[0], good), (axs[1], bad)):
        Ls = np.array(run["losses"])
        dl, gl = Ls[-500:, 0].mean(), Ls[-500:, 1].mean()
        up, lo = (0.09, -0.13)
        ax_.text(2050, dl + (up if dl > gl else lo), "D loss", color=C["disc"], fontsize=14)
        ax_.text(2050, gl + (up if gl > dl else lo), "G loss", color=C["dec"], fontsize=14)
    axs[0].set_ylim(0.3, 1.8)
    save(fig, "gan1d-losses")

    # ---------------- MNIST GAN: watch it learn, then vote
    snaps = np.load(MODELS / "gan_mnist_snapshots.npz")
    keys = ["0", "step50", "step150", "step300", "1", "3", "10", "30"]
    film = np.concatenate([snaps[k][:8] for k in keys])
    grid_png(film, len(keys), 8, "gan-film", scale=3, pad=2)
    grid_png(snaps["30"][:48], 6, 8, "gan-samples", scale=3)
    ck = torch.load(MODELS / "gan_mnist.pt", map_location=DEV)
    Dnet = ConvD().to(DEV)
    Dnet.load_state_dict(ck["D"])
    Dnet.eval()
    real_idx = torch.randperm(10000, generator=torch.Generator().manual_seed(31))[:6]
    xgray = load_mnist(binary=False)["test"][0]
    real = xgray[real_idx].numpy()
    fake = snaps["30"][8:14]
    items = [("real", im) for im in real] + [("generated", im) for im in fake]
    order = np.random.default_rng(8).permutation(12)
    vote = []
    for k_, j in enumerate(order):
        kind, im = items[j]
        with torch.no_grad():
            pr = torch.sigmoid(Dnet(torch.tensor(im * 2 - 1, dtype=torch.float32).reshape(1, 1, 28, 28).to(DEV))).item()
        tile_png(im, f"vote-{k_ + 1:02d}", scale=6)
        vote.append(dict(n=k_ + 1, kind=kind, d_real=r(pr, 2)))
    NUM["vote"] = vote
    tile_png(snaps["30"][2], "gan-fake-one", scale=6)          # a GAN sample (an 8)
    tile_png(xgray[DIGIT3].numpy(), "gan-real-one", scale=6)    # our 3, grey levels
    tile_png(snaps["30"][47], "gan-seven", scale=6)             # a GAN sample that happens to be a 7
    NUM["digit3_index"] = DIGIT3
    NUM["gan_final"] = meta["gan"]["history"][-1]
    NUM["gan"] = dict(epochs=meta["gan"]["epochs"], zdim=meta["gan"]["zdim"],
                      params_G=sum(v.numel() for k, v in ck["G"].items() if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))),
                      params_D=sum(v.numel() for v in ck["D"].values()))

    (FIG / "numbers.json").write_text(json.dumps(NUM, indent=1))
    print(json.dumps({k: NUM[k] for k in ("toygan", "dloss", "gan")}, indent=1)[:3000])


if __name__ == "__main__":
    main()

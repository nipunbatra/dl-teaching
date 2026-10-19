"""Train every model whose output appears in the GAN lecture.

    python scripts/train_models.py              # both (MNIST GAN ~30 epochs, eight 1-D GANs)
    python scripts/train_models.py --only gan   # or --only toygan, --only oneplayer

All figures are drawn from these checkpoints (see make_figures.py). The code is copied
unchanged from the Generative Modelling I (VAE) lecture, where these runs were first made.

Data: grey-level MNIST for the image GAN. Set MNIST_ROOT to a folder that
contains MNIST/raw/*-ubyte; nothing is downloaded.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "models"
MNIST_ROOT = os.environ.get("MNIST_ROOT", "/Users/nipun/git/psdv-teaching/datasets")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def load_mnist(binary=True):
    """binary=True: pixels thresholded at 0.5 (a Bernoulli observation, used by AE/VAE).
    binary=False: grey levels in [0, 1] (used by the image GAN)."""
    from torchvision.datasets import MNIST

    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = MNIST(MNIST_ROOT, train=train, download=False)
        x = ds.data.float() / 255.0
        out[split] = ((x > 0.5).float() if binary else x, ds.targets.clone())
    return out


def batches(x, y, batch_size, generator):
    order = torch.randperm(len(x), generator=generator)
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


# ----------------------------------------------------------------------------- MNIST GAN
class ConvG(nn.Module):
    """z [B, 100] -> image [B, 1, 28, 28] with pixels in [-1, 1]."""

    def __init__(self, zdim=100):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(zdim, 256 * 7 * 7), nn.BatchNorm1d(256 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),    # 14 -> 28
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 7, 7))


class ConvD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),     # 28 -> 14
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),   # 14 -> 7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),                           # one raw logit
        )

    def forward(self, x):
        return self.net(x)


def gan_step(G, D, opt_G, opt_D, real, zdim):
    """The exact two-phase update taught on the slides."""
    b = real.shape[0]
    # D phase: real -> 1, fake -> 0, fake detached so G receives nothing.
    z = torch.randn(b, zdim, device=real.device)
    fake = G(z)
    real_logits = D(real)
    fake_logits = D(fake.detach())
    loss_D = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
              + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
    opt_D.zero_grad(set_to_none=True)
    loss_D.backward()
    opt_D.step()
    # G phase: fresh fakes, D's parameters frozen, gradient flows through D's input.
    D.requires_grad_(False)
    z = torch.randn(b, zdim, device=real.device)
    fake_logits = D(G(z))
    loss_G = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()
    D.requires_grad_(True)
    return loss_D.item(), loss_G.item(), torch.sigmoid(real_logits).mean().item(), torch.sigmoid(fake_logits).mean().item()


def train_mnist_gan(data, epochs=30, seed=1, zdim=100):
    """Grey-level MNIST scaled to [-1, 1]; same gan_step as the 1-D toy and the slides."""
    torch.manual_seed(seed)
    G, D = ConvG(zdim).to(DEVICE), ConvD().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    xtr, ytr = data["train"]
    fixed_z = torch.randn(64, zdim, generator=torch.Generator().manual_seed(123)).to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    snapshots = {}

    def snap(epoch):
        G.eval()
        with torch.no_grad():
            snapshots[epoch] = (G(fixed_z).cpu().numpy()[:, 0] + 1) / 2
        G.train()

    snap(0)
    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        rows = []
        for xb, _ in batches(xtr, ytr, 128, gen):
            real = (xb.unsqueeze(1) * 2 - 1).to(DEVICE)
            rows.append(gan_step(G, D, opt_G, opt_D, real, zdim))
            step += 1
            if epoch == 1 and step in (50, 150, 300):
                snap(f"step{step}")
        m = np.mean(rows, axis=0)
        history.append(dict(epoch=epoch, loss_D=m[0], loss_G=m[1], D_real=m[2], D_fake=m[3]))
        print(f"GAN epoch {epoch:2d}  D {m[0]:.3f}  G {m[1]:.3f}  D(real) {m[2]:.2f}  D(fake) {m[3]:.2f}", flush=True)
        if epoch in (1, 2, 3, 5, 10, 20, epochs):
            snap(epoch)
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, OUT / "gan_mnist.pt")
    np.savez_compressed(OUT / "gan_mnist_snapshots.npz", **{str(k): v for k, v in snapshots.items()})
    return dict(name="gan_mnist", epochs=epochs, seed=seed, zdim=zdim, history=history)


# ----------------------------------------------------------------------------- 1-D toy GAN
def toy_data(n, generator):
    """The lecture's running 1-D data: two equal bumps at -2 and +2, std 0.5."""
    side = torch.where(torch.rand(n, generator=generator) < 0.5, -2.0, 2.0)
    return side + 0.5 * torch.randn(n, generator=generator)


def train_toy_gan(seed, steps=3000, lr=1e-3, lr_g=None, hidden=64):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(1, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))
    D = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g or lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(1000 + seed)
    fixed_z = torch.randn(2000, 1, generator=torch.Generator().manual_seed(7))
    grid = torch.linspace(-5, 5, 401).unsqueeze(1)
    keep = {0, 25, 100, 250, 500, 1000, 2000, steps}
    record = {}

    def snap(step):
        with torch.no_grad():
            record[step] = dict(samples=G(fixed_z)[:, 0].numpy().tolist(),
                                d=torch.sigmoid(D(grid))[:, 0].numpy().tolist())

    snap(0)
    losses = []
    for step in range(1, steps + 1):
        real = toy_data(128, gen).unsqueeze(1)
        losses.append(gan_step(G, D, opt_G, opt_D, real, 1)[:2])
        if step in keep:
            snap(step)
    final = np.array(record[steps]["samples"])
    left, right = np.mean(np.abs(final + 2) < 1.0), np.mean(np.abs(final - 2) < 1.0)
    return dict(seed=seed, steps=steps, lr=lr, left=float(left), right=float(right),
                grid=grid[:, 0].numpy().tolist(), record=record, losses=np.array(losses).tolist())


def train_toy_one_player(seed=4, d_steps=200, g_steps=300, lr=1e-3):
    """Each half of gan_step on its own, on the toy data.

    Phase 1: G frozen at its initial weights (the same G as step 0 of the main run with this seed),
             only the D step runs. Shows what D learns from one fixed batch of fakes.
    Phase 2: D frozen where phase 1 left it, only the G step runs. Shows where G sends the fakes
             when the judge stops learning.
    """
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    D = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(1000 + seed)
    fixed_z = torch.randn(2000, 1, generator=torch.Generator().manual_seed(7))
    grid = torch.linspace(-10, 10, 801).unsqueeze(1)  # wider than the main run: G overshoots past +-5

    def snap():
        with torch.no_grad():
            return dict(samples=G(fixed_z)[:, 0].numpy().tolist(), d=torch.sigmoid(D(grid))[:, 0].numpy().tolist())

    d_record, g_record = {0: snap()}, {}
    for step in range(1, d_steps + 1):          # the D step of gan_step, G never updated
        real = toy_data(128, gen).unsqueeze(1)
        fake = G(torch.randn(128, 1))
        real_logits, fake_logits = D(real), D(fake.detach())
        loss_D = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                  + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()
        if step in (5, 25, 100, d_steps):
            d_record[step] = snap()
    D.requires_grad_(False)
    g_record[0] = snap()
    for step in range(1, g_steps + 1):          # the G step of gan_step, D never updated
        fake_logits = D(G(torch.randn(128, 1)))
        loss_G = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()
        if step in (25, 100, g_steps):
            g_record[step] = snap()
    return dict(seed=seed, d_steps=d_steps, g_steps=g_steps, lr=lr, grid=grid[:, 0].numpy().tolist(),
                d_only=d_record, g_only=g_record)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all", choices=("all", "gan", "toygan", "oneplayer"))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "training.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t0 = time.time()
    if a.only in ("all", "gan"):
        meta["gan"] = train_mnist_gan(load_mnist(binary=False), epochs=30)
        meta_path.write_text(json.dumps(meta, indent=1))
    if a.only in ("all", "toygan"):
        runs = [train_toy_gan(seed) for seed in range(8)]
        # Same data and training loop, but a generator with only 4 hidden units per layer.
        # In our runs this small generator settles on a single bump (mode collapse).
        collapse = train_toy_gan(0, hidden=4)
        collapse["variant"] = "generator hidden width 4"
        for r in runs + [collapse]:
            print(f"toy GAN seed {r['seed']} {r.get('variant', '')}: fraction near -2 = {r['left']:.2f}, "
                  f"near +2 = {r['right']:.2f}", flush=True)
        (OUT / "toy_gan_runs.json").write_text(json.dumps(runs))
        (OUT / "toy_gan_collapse.json").write_text(json.dumps(collapse))
        meta["toygan"] = [dict(seed=r["seed"], left=r["left"], right=r["right"], steps=r["steps"], lr=r["lr"]) for r in runs]
        meta["toygan_collapse"] = dict(seed=0, hidden=4, left=collapse["left"], right=collapse["right"])
    if a.only in ("all", "oneplayer"):
        one = train_toy_one_player()
        (OUT / "toy_one_player.json").write_text(json.dumps(one))
        meta["toy_one_player"] = {k: one[k] for k in ("seed", "d_steps", "g_steps", "lr")}
    meta["device"] = DEVICE
    meta["torch"] = torch.__version__
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Train every model the rebuilt lecture shows. All runs are local, on MNIST.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/train_models.py uncond      # pixel DDPM, the lecture's main model
    $PY scripts/train_models.py cond        # same network + a digit label (CFG section)
    $PY scripts/train_models.py ae          # small conv autoencoder: 1x28x28 -> 4x7x7
    $PY scripts/train_models.py latent      # DDPM on the autoencoder's 4x7x7 latents
    $PY scripts/train_models.py bench       # 200-step speed check

Schedule everywhere: linear beta from 1e-4 to 0.02, T = 1000, one-based t, alpha_bar[0] = 1.
Target: the noise epsilon. Sampler: DDPM with posterior variance (beta-tilde).
Outputs go to models/v2/.
"""
import gzip
import os
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "work" / "data"
DEV = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
T = 1000


# ------------------------------------------------------------------ data
def load_mnist(train=True, labels=False):
    name = "train" if train else "t10k"
    x = np.frombuffer(gzip.decompress((DATA / f"{name}-images-idx3-ubyte.gz").read_bytes()), np.uint8, offset=16)
    x = torch.from_numpy(x.copy().reshape(-1, 1, 28, 28)).float() / 127.5 - 1
    if not labels:
        return x
    lab_file = DATA / f"{name}-labels-idx1-ubyte.gz"
    if not lab_file.exists():
        import urllib.request
        urllib.request.urlretrieve(f"https://ossci-datasets.s3.amazonaws.com/mnist/{name}-labels-idx1-ubyte.gz", lab_file)
    y = np.frombuffer(gzip.decompress(lab_file.read_bytes()), np.uint8, offset=8)
    return x, torch.from_numpy(y.copy()).long()


# ------------------------------------------------------------------ schedule
def schedule(device="cpu"):
    """Index 0 is the clean sentinel: beta[0] = 0, alpha_bar[0] = 1."""
    beta = torch.zeros(T + 1, device=device)
    beta[1:] = torch.linspace(1e-4, 2e-2, T, device=device)
    alpha = 1 - beta
    ab = torch.cumprod(alpha, 0)
    var = torch.zeros_like(beta)
    var[1:] = beta[1:] * (1 - ab[:-1]) / (1 - ab[1:])  # posterior variance beta-tilde
    return beta, alpha, ab, var


# ------------------------------------------------------------------ the denoiser shown on the U-Net slide
class TimeBlock(nn.Module):
    def __init__(self, cin, cout, temb=64):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.time = nn.Linear(temb, cout)          # project the time embedding to this block's channels
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, h):
        y = F.silu(self.norm1(self.conv1(x))) + self.time(h)[:, :, None, None]   # broadcast over H x W
        return F.silu(self.norm2(self.conv2(y)) + self.skip(x))


def time_features(t, dim=64):
    freq = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2 - 1))
    ang = t.float()[:, None] * freq
    return torch.cat([ang.sin(), ang.cos()], 1)


class TinyUNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28 with channels 32, 64, 128, 64, 32. n_classes > 0 adds a label (10 = no label)."""

    def __init__(self, n_classes=0):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.label = nn.Embedding(n_classes + 1, 64) if n_classes else None
        self.encoder1 = TimeBlock(1, 32)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.encoder2 = TimeBlock(64, 64)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.middle = TimeBlock(128, 128)
        self.up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.decoder1 = TimeBlock(128, 64)
        self.up2 = nn.Conv2d(64, 32, 3, padding=1)
        self.decoder2 = TimeBlock(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x, t, c=None, keep=None):
        h = self.time_mlp(time_features(t))
        if self.label is not None:
            h = h + self.label(c)
        e1 = self.encoder1(x, h)
        e2 = self.encoder2(self.down1(e1), h)
        m = self.middle(self.down2(e2), h)
        u1 = self.up1(F.interpolate(m, size=e2.shape[-2:], mode="nearest"))
        d1 = self.decoder1(torch.cat([u1, e2], 1), h)
        u2 = self.up2(F.interpolate(d1, size=e1.shape[-2:], mode="nearest"))
        d2 = self.decoder2(torch.cat([u2, e1], 1), h)
        if keep is not None:
            keep.update(e1=e1, e2=e2, m=m, d1=d1, d2=d2)
        return self.head(d2)


# ------------------------------------------------------------------ latent autoencoder and latent denoiser
class ConvAE(nn.Module):
    """1x28x28 image <-> 4x7x7 spatial latent (196 numbers, 4x fewer than 784)."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Conv2d(64, 4, 3, 1, 1))
        self.dec = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1), nn.SiLU(), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 1, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        return self.dec(self.enc(x))


class LatentDenoiser(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.inp = nn.Conv2d(4, ch, 3, padding=1)
        self.blocks = nn.ModuleList([TimeBlock(ch, ch) for _ in range(4)])
        self.out = nn.Conv2d(ch, 4, 3, padding=1)

    def forward(self, z, t):
        h = self.time_mlp(time_features(t))
        y = self.inp(z)
        for b in self.blocks:
            y = b(y, h)
        return self.out(y)


class Classifier(nn.Module):
    """A plain MNIST classifier, used only to measure what guided samples look like."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 49, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def train_clf(epochs=3):
    torch.manual_seed(0)
    x, y = load_mnist(True, labels=True)
    xt, yt = load_mnist(False, labels=True)
    m = Classifier()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            ix = perm[i:i + 256]
            loss = F.cross_entropy(m(x[ix]), y[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = float((m(xt).argmax(1) == yt).float().mean())
        print("clf epoch", ep + 1, "test accuracy", round(acc, 4), flush=True)
    torch.save(m.state_dict(), OUT / "clf.pt")
    (OUT / "clf-log.json").write_text(json.dumps(dict(epochs=epochs, test_accuracy=acc)))


# ------------------------------------------------------------------ the training step shown on the code slide
def training_step(model, x0, ab, opt, c=None, p_drop=0.1):
    B = x0.shape[0]
    t = torch.randint(1, T + 1, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    a = ab[t].reshape(B, *([1] * (x0.ndim - 1)))
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    if c is not None:                       # classifier-free guidance training: sometimes hide the label
        c = torch.where(torch.rand(B, device=x0.device) < p_drop, torch.full_like(c, 10), c)
        eps_hat = model(xt, t, c)
    else:
        eps_hat = model(xt, t)
    loss = F.mse_loss(eps_hat, eps)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def train_diffusion(name, model, data, labels=None, steps=30000, batch=128, lr=2e-4, seed=667):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEV)
    _, _, ab, _ = schedule(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model)
    hist, t0 = [], time.time()
    data = data.to(DEV)
    if labels is not None:
        labels = labels.to(DEV)
    run = 0.0
    for step in range(1, steps + 1):
        ix = torch.randint(len(data), (batch,), device=DEV)
        loss = training_step(model, data[ix], ab, opt, None if labels is None else labels[ix])
        ema.update(model)
        run = loss if step == 1 else 0.98 * run + 0.02 * loss
        if step == 1 or step % 250 == 0:
            hist.append(dict(step=step, loss=round(loss, 5), smooth=round(run, 5)))
        if step % 1000 == 0 or step == steps:
            print(f"{name} step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            torch.save(dict(model=ema.shadow, raw=model.state_dict(), steps=step, seed=seed, batch=batch, lr=lr),
                       OUT / f"{name}.pt")
    meta = dict(name=name, steps=steps, batch=batch, lr=lr, seed=seed, ema=0.999, device=DEV, torch=torch.__version__,
                params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0, 1), history=hist)
    (OUT / f"{name}-log.json").write_text(json.dumps(meta, indent=1))
    return meta


def train_ae(epochs=15, seed=667):
    torch.manual_seed(seed)
    x = load_mnist(True).to(DEV)
    ae = ConvAE().to(DEV)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(x), device=DEV)
        tot = 0
        for i in range(0, len(x), 256):
            xb = x[perm[i:i + 256]]
            loss = F.mse_loss(ae(xb), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        hist.append(dict(epoch=ep + 1, mse=round(tot / len(x), 5)))
        print("ae", hist[-1], flush=True)
    with torch.no_grad():
        z = torch.cat([ae.enc(x[i:i + 2000]) for i in range(0, len(x), 2000)])
    scale = float(z.std())
    torch.save(dict(model=ae.state_dict(), latent_std=scale), OUT / "ae.pt")
    (OUT / "ae-log.json").write_text(json.dumps(dict(epochs=epochs, latent_std=scale, seconds=round(time.time() - t0, 1),
                                                     history=hist), indent=1))


def main():
    what = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    if what == "bench":
        x = load_mnist(True)
        m = TinyUNet().to(DEV)
        _, _, ab, _ = schedule(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=2e-4)
        xd = x[:20000].to(DEV)
        for b in (128,):
            t0 = time.time()
            for _ in range(200):
                ix = torch.randint(len(xd), (b,), device=DEV)
                training_step(m, xd[ix], ab, opt)
            torch.mps.synchronize() if DEV == "mps" else None
            print(b, (time.time() - t0) / 200, "s/step", sum(p.numel() for p in m.parameters()), "params")
    elif what == "uncond":
        train_diffusion("uncond", TinyUNet(), load_mnist(True), steps=steps)
    elif what == "cond":
        x, y = load_mnist(True, labels=True)
        train_diffusion("cond", TinyUNet(n_classes=10), x, y, steps=steps)
    elif what == "ae":
        train_ae()
    elif what == "clf":
        train_clf()
    elif what == "latent":
        ck = torch.load(OUT / "ae.pt", map_location=DEV)
        ae = ConvAE().to(DEV)
        ae.load_state_dict(ck["model"])
        x = load_mnist(True).to(DEV)
        with torch.no_grad():
            z = torch.cat([ae.enc(x[i:i + 2000]) for i in range(0, len(x), 2000)]) / ck["latent_std"]
        train_diffusion("latent", LatentDenoiser(), z.cpu(), steps=steps, batch=256)


if __name__ == "__main__":
    main()

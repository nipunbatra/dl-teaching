"""Verify every worked number that appears in the L19 deck.
Run from repo root:  python3 lecture19/diagrams/verify_numbers.py
"""
import numpy as np

def kl_diag_gauss(mu, sigma):
    """KL( N(mu, diag(sigma^2)) || N(0, I) ) = 1/2 sum (mu^2 + sigma^2 - log sigma^2 - 1)."""
    mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    return 0.5 * np.sum(mu**2 + sigma**2 - np.log(sigma**2) - 1.0)

print("== Reparameterization (slide: board calc) ==")
mu    = np.array([1.0, -1.0])
sigma = np.array([0.5,  0.2])
eps   = np.array([-0.4, 1.5])
z = mu + sigma * eps
print("  z = mu + sigma*eps =", z, " expected [0.8, -0.7]")
assert np.allclose(z, [0.8, -0.7]), "reparam mismatch"

print("\n== Gaussian KL sanity: KL(N(0,1)||N(0,1)) should be 0 ==")
print("  KL =", kl_diag_gauss([0.0], [1.0]))
assert abs(kl_diag_gauss([0.0], [1.0])) < 1e-12

print("\n== Worked Gaussian KL (mu=[0.8,-0.7], sigma=[0.5,0.2]) ==")
klval = kl_diag_gauss([0.8, -0.7], [0.5, 0.2])
# per-dim breakdown
for m, s in ((0.8, 0.5), (-0.7, 0.2)):
    term = m**2 + s**2 - np.log(s**2) - 1.0
    print(f"  dim mu={m:+.2f} sigma={s:.2f}:  mu^2={m**2:.3f} sigma^2={s**2:.3f}"
          f" -log sigma^2={-np.log(s**2):+.4f}  -> term={term:.4f}")
print(f"  KL = 1/2 * sum = {klval:.4f} nats  (rounded 2.01)")
assert abs(klval - 2.0126) < 1e-3

print("\n== Worked ELBO loss (one data point, sigma_x^2 = 1) ==")
recon_sq = 3.0                      # ||x - xhat||^2
recon_term = 0.5 * recon_sq / 1.0   # -E[log p(x|z)] up to constant, Gaussian decoder
kl_term = klval
loss = recon_term + kl_term
print(f"  reconstruction term = 1/2 ||x-xhat||^2 = {recon_term:.3f}")
print(f"  KL term             = {kl_term:.4f}")
print(f"  VAE loss = recon + KL = {loss:.4f}  (rounded 3.51)")
assert abs(recon_term - 1.5) < 1e-9
assert abs(loss - 3.5126) < 1e-3

print("\n== log-variance <-> sigma (slide: why output log-variance) ==")
ell = -1.3863     # log sigma^2  = log 0.25
sig = np.exp(0.5 * ell)
print(f"  ell = log sigma^2 = {ell}  ->  sigma = exp(ell/2) = {sig:.4f}  (expect 0.5)")
assert abs(sig - 0.5) < 1e-3

print("\n== KL trade-off two-encoder totals (slide 53) ==")
for name, recon, kl in (("individualized", 1, 100), ("near-prior", 10, 1)):
    print(f"  {name:16s}: recon={recon:3d}  KL={kl:3d}  total={recon+kl}")

print("\n== gaussian_kl figure: min of KL(N(mu,s^2)||N(0,1)) over s at mu=0 ==")
s = np.linspace(0.05, 3, 4000)
kl0 = 0.5 * (0.0 + s**2 - np.log(s**2) - 1.0)
smin = s[np.argmin(kl0)]
print(f"  argmin_s KL(mu=0) = {smin:.4f} (expect 1.0), KL there = {kl0.argmin() and kl0.min():.4f} (expect 0)")
assert abs(smin - 1.0) < 2e-3
assert abs(kl0.min()) < 1e-4

print("\nALL CHECKS PASSED")

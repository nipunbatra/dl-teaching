"""Metropolis-style figures for Lecture 6 (Making Deep Networks Trainable).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Run from repo root:  python3 lecture6/diagrams/l6_figs.py
matplotlib mathtext: NO \\le / \\ge — use unicode  ≤ ≥ ."""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture6/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def sig(z): return 1/(1+np.exp(-z))

# ── 1 — four failure modes: vanishing/exploding × activations/gradients ──
def f_failure_modes():
    L=np.arange(0,41)
    fig,axes=plt.subplots(2,2,figsize=(8.4,4.4))
    panels=[
      (axes[0,0],'activations vanish  →0', 1.0*0.72**L, RED,   r'$q_\ell=\mathrm{Var}(h^{(\ell)})$'),
      (axes[0,1],'activations explode  →∞', 1.0*1.30**L, ACC,  r'$q_\ell=\mathrm{Var}(h^{(\ell)})$'),
      (axes[1,0],'gradients vanish  →0',   1.0*0.68**L, BLUE,  r'$g_\ell=\Vert\partial L/\partial h^{(\ell)}\Vert^2$'),
      (axes[1,1],'gradients explode  →∞',  1.0*1.34**L, TEAL,  r'$g_\ell=\Vert\partial L/\partial h^{(\ell)}\Vert^2$'),
    ]
    for ax,ttl,y,c,yl in panels:
        ax.semilogy(L,y,'-o',color=c,ms=2.6,lw=1.8)
        ax.axhline(1.0,color=MUTED,lw=.8,ls=':')
        ax.set_title(ttl,fontsize=12,color=c)
        ax.set_ylabel(yl,fontsize=10)
        ax.set_xlabel('layer  ℓ',fontsize=10)
        ax.set_ylim(1e-6,1e6)
        ax.set_yticks([1e-6,1e-3,1,1e3,1e6])
    fig.tight_layout(w_pad=2.0,h_pad=1.6)
    save(fig,'failure_modes')

# ── 2 — activation zoo: sigmoid / tanh / ReLU / LeakyReLU / GELU / SiLU + derivatives ──
def f_activation_zoo():
    z=np.linspace(-4,4,400)
    def gelu(z): return z*sig(1.702*z)                        # sigmoid approx
    def gelu_d(z): s=sig(1.702*z); return s+1.702*z*s*(1-s)
    def silu(z): return z*sig(z)
    def silu_d(z): s=sig(z); return s+z*s*(1-s)
    lk=0.1
    acts=[
      ('sigmoid',      sig(z),                sig(z)*(1-sig(z))),
      ('tanh',         np.tanh(z),            1-np.tanh(z)**2),
      ('ReLU',         np.maximum(0,z),       (z>0).astype(float)),
      ('Leaky ReLU',   np.where(z>0,z,lk*z),  np.where(z>0,1.0,lk)),
      ('GELU',         gelu(z),               gelu_d(z)),
      ('SiLU / Swish', silu(z),               silu_d(z)),
    ]
    fig,axes=plt.subplots(2,3,figsize=(8.8,4.2))
    for ax,(nm,a,d) in zip(axes.ravel(),acts):
        ax.plot(z,a,color=INK,lw=2.4)
        ax.plot(z,d,color=ACC,lw=1.6,ls='--',alpha=.9)
        ax.axhline(0,color=MUTED,lw=.6); ax.axvline(0,color=MUTED,lw=.6)
        ax.set_title(nm,fontsize=12,pad=4)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-1.4,2.2)
    axes[0,0].plot([],[],color=INK,label=r'$\phi$')
    axes[0,0].plot([],[],color=ACC,ls='--',label=r"$\phi'$")
    axes[0,0].legend(frameon=False,fontsize=10,loc='upper left')
    fig.tight_layout(w_pad=1.4,h_pad=1.4)
    save(fig,'activation_zoo')

# ── 3 — sigmoid saturation: σ, σ' with ≤1/4 cap + inset (1/4)^L decay ──
def f_sigmoid_saturation():
    z=np.linspace(-8,8,400)
    fig,ax=plt.subplots(figsize=(6.6,3.2))
    ax.plot(z,sig(z),color=INK,lw=2.6,label=r'$\sigma(z)$')
    ax.plot(z,sig(z)*(1-sig(z)),color=ACC,lw=2.4,ls='--',label=r"$\sigma'(z)$")
    ax.axhline(0.25,color=RED,lw=1.2,ls=':')
    ax.text(-7.6,0.28,r"max $\sigma'=1/4$",color=RED,fontsize=11)
    ax.axvspan(3,8,color=MUTED,alpha=.10); ax.axvspan(-8,-3,color=MUTED,alpha=.10)
    ax.text(4.3,0.7,'saturated\n$\\sigma\'\\approx0$',color=MUTED,fontsize=10,ha='center')
    ax.set_xlabel('z'); ax.set_yticks([0,.25,.5,1])
    ax.legend(frameon=False,fontsize=12,loc='center left')
    # inset: (1/4)^L
    axin=ax.inset_axes([0.60,0.42,0.36,0.42])
    Ls=np.arange(1,13)
    axin.semilogy(Ls,(0.25)**Ls,'-o',color=TEAL,ms=3,lw=1.6)
    axin.set_title(r'$(1/4)^L$',fontsize=10,color=TEAL)
    axin.set_xlabel('depth L',fontsize=8); axin.tick_params(labelsize=7)
    axin.set_yticks([1e0,1e-3,1e-6])
    save(fig,'sigmoid_saturation')

# ── 4 — variance compounding c^L for c in {0.5, 1, 2} ──
def f_variance_compound():
    L=np.arange(0,41)
    fig,ax=plt.subplots(figsize=(6.6,3.2))
    for c,lbl,col in [(0.5,'c = 0.5  (shrinks)',RED),(1.0,'c = 1  (preserved)',GREEN),(2.0,'c = 2  (explodes)',ACC)]:
        ax.semilogy(L,c**L,'-o',color=col,ms=2.6,lw=2.0,label=lbl)
    ax.axhline(1.0,color=MUTED,lw=.8,ls=':')
    ax.set_xlabel(r'depth  $L$'); ax.set_ylabel(r'$\mathrm{Var}(h^{(L)})/\mathrm{Var}(x)=c^{L}$')
    ax.set_ylim(1e-7,1e7); ax.set_yticks([1e-6,1e-3,1,1e3,1e6])
    ax.legend(frameon=False,fontsize=11,loc='upper left')
    ax.set_title(r'$c=n\,\mathrm{Var}(w)$ — only $c=1$ survives depth',fontsize=12)
    save(fig,'variance_compound')

# ── 5 — signal flow: activation RMS vs depth for tiny / Xavier / He (ReLU net) ──
def f_signal_flow():
    def run(std_val, L=40, n=256, B=1024, seed=0):
        rng=np.random.default_rng(seed)
        h=rng.standard_normal((B,n))            # x ~ N(0,1)
        rms=[np.sqrt((h**2).mean())]
        for _ in range(L):
            W=rng.standard_normal((n,n))*std_val(n)
            h=np.maximum(0, h@W.T)              # ReLU
            rms.append(np.sqrt((h**2).mean()))
        return np.array(rms)
    L=np.arange(0,41)
    tiny =run(lambda n:0.02)
    xav  =run(lambda n:np.sqrt(1.0/n))
    he   =run(lambda n:np.sqrt(2.0/n))
    fig,ax=plt.subplots(figsize=(6.8,3.3))
    ax.semilogy(L,tiny,'-o',color=RED, ms=2.6,lw=2.0,label=r'tiny  $\sigma=0.02$  (collapses)')
    ax.semilogy(L,xav, '-o',color=BLUE,ms=2.6,lw=2.0,label=r'Xavier  $1/n$  (drifts under ReLU)')
    ax.semilogy(L,he,  '-o',color=GREEN,ms=2.6,lw=2.0,label=r'He  $2/n$  (preserved)')
    ax.axhline(1.0,color=MUTED,lw=.8,ls=':')
    ax.set_xlabel('layer  ℓ'); ax.set_ylabel(r'activation RMS  $\sqrt{E[h^2]}$')
    ax.set_ylim(1e-6,3); ax.legend(frameon=False,fontsize=10.5,loc='lower left')
    save(fig,'signal_flow')

# ── 6 — gradient norm vs depth: sigmoid (vanishes) vs ReLU+He (stable) ──
def f_grad_norm_depth():
    def run(act,dact,std_val,L=25,n=128,B=512,seed=1):
        rng=np.random.default_rng(seed)
        h=rng.standard_normal((B,n)); zs=[]; Ws=[]; hs=[h]
        for _ in range(L):
            W=rng.standard_normal((n,n))*std_val(n); Ws.append(W)
            z=hs[-1]@W.T; zs.append(z); hs.append(act(z))
        g=rng.standard_normal(hs[-1].shape)          # dL/dh_L
        norms=[np.sqrt((g**2).mean())]
        for l in reversed(range(L)):
            g=g*dact(zs[l]); g=g@Ws[l]
            norms.append(np.sqrt((g**2).mean()))
        return np.array(norms[::-1])                 # layer 0..L
    L=np.arange(0,26)
    sg=run(sig, lambda z:sig(z)*(1-sig(z)), lambda n:np.sqrt(1.0/n))
    rl=run(lambda z:np.maximum(0,z), lambda z:(z>0).astype(float), lambda n:np.sqrt(2.0/n))
    sg=sg/sg[-1]; rl=rl/rl[-1]                        # normalise at output layer
    fig,ax=plt.subplots(figsize=(6.8,3.3))
    ax.semilogy(L,sg,'-o',color=RED, ms=2.6,lw=2.0,label='sigmoid + Xavier  (vanishes)')
    ax.semilogy(L,rl,'-o',color=GREEN,ms=2.6,lw=2.0,label='ReLU + He  (stable)')
    ax.set_xlabel('layer  ℓ  (0 = input side)')
    ax.set_ylabel(r'grad norm  $\Vert\partial L/\partial h^{(\ell)}\Vert$')
    ax.set_ylim(1e-8,1e1); ax.legend(frameon=False,fontsize=10.5,loc='lower right')
    ax.set_title('gradient reaching each layer during backprop',fontsize=12)
    save(fig,'grad_norm_depth')

# ── 7 — ablation: train loss vs steps for six configs ──
def f_ablation_curves():
    t=np.arange(0,300)
    rng=np.random.default_rng(4)
    def curve(start,floor,tau,noise=0.01):
        y=floor+(start-floor)*np.exp(-t/tau)
        return y+rng.normal(0,noise,t.shape)*np.sqrt(np.maximum(y,1e-3))
    cfgs=[
      ('sigmoid + N(0,1)',            curve(2.35,2.15,120,0.006), RED),
      ('sigmoid + Xavier',            curve(2.30,1.15,90),        ACC),
      ('tanh + Xavier',               curve(2.30,0.62,55),        BLUE),
      ('ReLU + He',                   curve(2.30,0.34,38),        TEAL),
      ('ReLU + He + BatchNorm',       curve(2.30,0.17,24),        GREEN),
      ('ReLU + He + BN + residual',   curve(2.30,0.08,16),        INK),
    ]
    fig,ax=plt.subplots(figsize=(7.2,3.4))
    for lbl,y,c in cfgs:
        ax.plot(t,y,color=c,lw=2.2,label=lbl)
    ax.set_xlabel('training step'); ax.set_ylabel('training loss')
    ax.set_ylim(0,2.5); ax.legend(frameon=False,fontsize=9.5,loc='upper right')
    ax.set_title('50-layer net — same data, six recipes',fontsize=12)
    save(fig,'ablation_curves')

# ── 8 — residual vs plain: gradient norm across layers ──
def f_residual_vs_plain():
    L=40; rng=np.random.default_rng(2)
    layers=np.arange(0,L+1)
    plain=0.88**layers*np.exp(rng.normal(0,0.04,L+1))          # geometric decay
    steps=1+rng.normal(0,0.05,L)                               # near-identity Jacobians
    resid=np.concatenate([[1.0],np.cumprod(steps)])[::-1]      # random walk near 1
    fig,ax=plt.subplots(figsize=(6.8,3.3))
    ax.semilogy(layers,plain,'-o',color=RED, ms=2.6,lw=2.0,label='plain net  (decays)')
    ax.semilogy(layers,resid,'-o',color=GREEN,ms=2.6,lw=2.0,label='residual net  (flat)')
    ax.axhline(1.0,color=MUTED,lw=.8,ls=':')
    ax.set_xlabel('layer  ℓ  (0 = input side)')
    ax.set_ylabel(r'grad norm  $\Vert\partial L/\partial x^{(\ell)}\Vert$')
    ax.set_ylim(1e-3,3); ax.legend(frameon=False,fontsize=10.5,loc='lower right')
    ax.set_title('identity path keeps gradients alive to depth',fontsize=12)
    save(fig,'residual_vs_plain')

for f in [f_failure_modes,f_activation_zoo,f_sigmoid_saturation,f_variance_compound,
          f_signal_flow,f_grad_norm_depth,f_ablation_curves,f_residual_vs_plain]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

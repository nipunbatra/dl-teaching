"""Metropolis-style figures for Lecture 3 (Backpropagation & Autodiff).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Only two plots — everything else in the deck is fletcher / table / math.
Run from repo root:  python3 lecture3/diagrams/l3_figs.py"""
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
OUT='lecture3/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# 1 — sigmoid and its derivative, with the 1/4 cap
def f_sigmoid_deriv():
    z=np.linspace(-8,8,400)
    s=1/(1+np.exp(-z)); ds=s*(1-s)
    fig,ax=plt.subplots(figsize=(4.8,3.1))
    ax.plot(z,s,color=INK,lw=2.6,label=r'$\sigma(z)$')
    ax.plot(z,ds,color=ACC,lw=2.6,label=r"$\sigma'(z)$")
    ax.axhline(0.25,ls='--',color=TEAL,lw=1.5)
    ax.text(4.4,0.285,r"max $\sigma'=\frac{1}{4}$",color=TEAL,fontsize=12)
    ax.axhline(0,color=MUTED,lw=.6)
    ax.set_xlabel('z'); ax.set_yticks([0,0.25,0.5,1.0])
    ax.legend(frameon=False,fontsize=13,loc='center left')
    ax.set_title(r"$\sigma'$ is tiny in the saturated tails",fontsize=13)
    save(fig,'sigmoid_deriv')

# 2 — gradient magnitude vs depth for sigmoid / tanh / relu
def f_grad_flow():
    L=20; layers=np.arange(L+1)
    rng=np.random.default_rng(0)
    # product of typical local derivatives per layer (rough illustration)
    def chain(scale):
        g=np.ones(L+1)
        for k in range(1,L+1):
            g[k]=g[k-1]*scale
        return g
    sig=chain(0.25)        # sigmoid: |sigma'| <= 1/4  -> strong decay
    tanh=chain(0.6)        # tanh: derivative <= 1, typically < 1 -> milder decay
    relu=chain(1.0)        # relu: derivative 0/1, ~flat when active
    fig,ax=plt.subplots(figsize=(5.2,3.1))
    ax.semilogy(layers,sig,'-o',color=RED,ms=3,lw=2.0,label='sigmoid  (×0.25)')
    ax.semilogy(layers,tanh,'-o',color=ACC,ms=3,lw=2.0,label='tanh  (×0.6)')
    ax.semilogy(layers,relu,'-o',color=TEAL,ms=3,lw=2.0,label='ReLU  (×1.0)')
    ax.set_xlabel('layers back-propagated'); ax.set_ylabel(r'$\|\partial\mathcal{L}/\partial h_\ell\|$  (rel.)')
    ax.legend(frameon=False,fontsize=11,loc='lower left')
    ax.set_title('gradient magnitude through a deep chain',fontsize=13)
    save(fig,'grad_flow')

for f in [f_sigmoid_deriv,f_grad_flow]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

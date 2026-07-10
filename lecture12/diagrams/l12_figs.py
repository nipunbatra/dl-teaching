"""Metropolis-style figures for Lecture 12 (Sequence Models I: RNNs, BPTT, LSTMs, GRUs).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Schematic / synthetic only, on-palette. Run from repo root:
    python3 lecture12/diagrams/l12_figs.py
Every structural diagram (RNN cell, unrolled RNN, BPTT, LSTM cell, GRU cell,
fixed-vs-recurrent) is NATIVE fletcher in the deck. These are the quantitative
plots only.
"""
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
OUT='lecture12/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── 1 — scalar-RNN forward: state trajectory h_1,h_2,h_3 for x=(1,2,-1) ──
def f_rnn_forward():
    wx,wh,b=1.0,0.5,0.0
    xs=[1,2,-1]
    hs=[0.0]; as_=[]
    h=0.0
    for x in xs:
        a=wx*x+wh*h+b; h=np.tanh(a)
        as_.append(a); hs.append(h)
    fig,ax=plt.subplots(figsize=(6.6,3.2))
    t=np.arange(0,len(xs)+1)
    ax.axhline(0,color=MUTED,lw=.7,alpha=.6)
    ax.axhline(1,color=MUTED,lw=.6,ls=':',alpha=.5)
    ax.axhline(-1,color=MUTED,lw=.6,ls=':',alpha=.5)
    ax.text(3.08,1.0,'+1',fontsize=10,color=MUTED,va='center')
    ax.text(3.08,-1.0,'−1',fontsize=10,color=MUTED,va='center')
    # hidden-state trajectory
    ax.plot(t,hs,'-o',color=TEAL,ms=8,lw=2.6,zorder=4,label=r'hidden state $h_t$')
    # input stems
    for i,x in enumerate(xs,1):
        ax.plot([i,i],[0,x*0.32],color=ACC,lw=2.2,alpha=.8,zorder=2)
        ax.plot(i,x*0.32,'o',color=ACC,ms=5,zorder=3)
    ax.plot([],[],color=ACC,lw=2.2,label=r'input $x_t$ (scaled)')
    labs=[r'$h_0=0$',r'$h_1\approx0.76$',r'$h_2\approx0.98$',r'$h_3\approx-0.47$']
    offs=[(0.06,-0.22),(0.02,0.20),(-0.05,0.20),(0.05,-0.26)]
    for ti,hi,lb,(dx,dy) in zip(t,hs,labs,offs):
        ax.annotate(lb,(ti,hi),(ti+dx,hi+dy),fontsize=11.5,color=INK,fontweight='bold')
    ax.set_xlim(-0.25,3.55); ax.set_ylim(-1.25,1.32)
    ax.set_xticks([0,1,2,3]); ax.set_xticklabels([r'$t{=}0$',r'$t{=}1$',r'$t{=}2$',r'$t{=}3$'])
    ax.set_yticks([-1,0,1])
    ax.set_xlabel('time step',fontsize=12)
    ax.set_title(r'scalar RNN  $h_t=\tanh(x_t+0.5\,h_{t-1})$,   $x=(1,2,-1)$',fontsize=12.5)
    ax.legend(frameon=False,fontsize=11,loc='lower left')
    save(fig,'rnn_forward')

# ── 2 — gradient norm vs time-lag: vanish / explode / gated (log y) ──
def f_grad_vs_lag():
    T=np.arange(0,51)
    vanish=0.9**T
    explode=1.1**T
    gated=np.full_like(T,1.0,dtype=float)*0.85  # roughly flat, f_t ~ 1
    fig,ax=plt.subplots(figsize=(6.8,3.3))
    ax.semilogy(T,explode,'-',color=RED,lw=2.8,label=r'$w=1.1$  exploding')
    ax.semilogy(T,gated,'-',color=GREEN,lw=2.8,label=r'gated (LSTM, $f_t\!\approx\!1$)')
    ax.semilogy(T,vanish,'-',color=BLUE,lw=2.8,label=r'$w=0.9$  vanishing')
    ax.axhline(1.0,color=MUTED,lw=.7,ls='--',alpha=.6)
    # end annotations
    ax.annotate(r'$\approx117$',(50,117),(41,220),fontsize=11,color=RED,fontweight='bold')
    ax.annotate(r'$\approx0.005$',(50,0.005),(37,0.0016),fontsize=11,color=BLUE,fontweight='bold')
    ax.set_xlim(0,52); ax.set_ylim(1e-3,5e2)
    ax.set_xlabel('time-lag  $T-k$  (steps back)',fontsize=12)
    ax.set_ylabel(r'gradient magnitude  $\|\partial h_T/\partial h_k\|$',fontsize=11.5)
    ax.set_title(r'why long-range learning fails — product of $T$ factors',fontsize=12.5)
    leg=ax.legend(frameon=True,fontsize=10.5,loc='lower left',
                  facecolor='white',edgecolor=MUTED,framealpha=.95)
    leg.get_frame().set_linewidth(.8)
    save(fig,'grad_vs_lag')

# ── 3 — tanh and its derivative: saturation kills the gradient ──
def f_tanh_sat():
    z=np.linspace(-4,4,400)
    t=np.tanh(z); dt=1-t**2
    fig,ax=plt.subplots(figsize=(6.4,3.1))
    ax.axhline(0,color=MUTED,lw=.7,alpha=.6); ax.axvline(0,color=MUTED,lw=.7,alpha=.6)
    ax.plot(z,t,color=TEAL,lw=2.8,label=r'$\tanh(z)$')
    ax.plot(z,dt,color=ACC,lw=2.8,label=r"$\tanh'(z)=1-\tanh^2 z$")
    # saturation shading
    ax.axvspan(2.2,4,color=MUTED,alpha=.10); ax.axvspan(-4,-2.2,color=MUTED,alpha=.10)
    ax.text(3.1,0.55,'saturated\n$\\tanh\'\\to 0$',fontsize=10,color=MUTED,ha='center',va='center')
    ax.annotate(r"$\tanh'(0)=1$",(0,1),(0.35,0.62),fontsize=11,color=ACC,fontweight='bold',
                arrowprops=dict(arrowstyle='-|>',color=ACC,lw=1.5))
    ax.set_xlim(-4,4); ax.set_ylim(-1.15,1.25)
    ax.set_xticks([-4,-2,0,2,4]); ax.set_yticks([-1,0,1])
    ax.set_xlabel('pre-activation  $z$',fontsize=12)
    ax.set_title('tanh saturates — the local slope shrinks toward 0',fontsize=12.5)
    ax.legend(frameon=False,fontsize=11,loc='lower right')
    save(fig,'tanh_sat')

for f in [f_rnn_forward,f_grad_vs_lag,f_tanh_sat]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

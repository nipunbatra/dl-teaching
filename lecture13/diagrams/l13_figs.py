"""Metropolis-style figures for Lecture 13 (Convolutional Sequence Models).
Schematic / synthetic ONLY — no real data. Transparent bg, ink + orange/teal.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Every structural diagram (causal connectivity, dilation fan-ins, dependency
trees, TCN block, WaveNet gated residual+skip block, RNN-vs-TCN path length)
is NATIVE fletcher in the deck. These are the three quantitative / schematic
plots only.
Run from repo root:  python3 lecture13/diagrams/l13_figs.py
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture13/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def cell(ax, x, y, w, h, val, fc, ec=INK, fs=15, tc=INK, lw=1.4):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
    if val is not None:
        ax.text(x+w/2, y+h/2, val, ha='center', va='center', fontsize=fs, color=tc, weight=600, zorder=3)

# ── 1 — temporal 1D convolution: x=[1,2,3,4,5] * w=[2,-1] -> y=[0,1,2,3] ──
def f_temporal_conv():
    x=[1,2,3,4,5]; w=[2,-1]; y=[0,1,2,3]
    fig,ax=plt.subplots(figsize=(9.4,3.9))
    ax.set_xlim(-0.6,5.6); ax.set_ylim(-2.35,3.15); ax.set_aspect('equal'); ax.axis('off')
    # input row (top), positions t=1..5
    ax.text(-0.5,2.5+0.5,'input  $x$',ha='right',va='center',fontsize=13,color=INK)
    for i,v in enumerate(x):
        cell(ax, i, 2.5, 1, 1, f'{v}', TEAL if i<2 else '#EAF3F3')
        ax.text(i+0.5, 3.72, f'$x_{i+1}$', ha='center', va='center', fontsize=11, color=MUTED)
    # highlight the first window (x1,x2) producing y1
    ax.add_patch(Rectangle((0,2.5), 2, 1, fill=False, edgecolor=ACC, lw=3.2, zorder=6))
    # kernel, floating between
    ax.text(-0.5, 1.1+0.4, 'kernel  $w$', ha='right', va='center', fontsize=13, color=INK)
    cell(ax, 0, 1.05, 1, 0.8, '2', '#FDECD6', ec=ACC, fs=15)
    cell(ax, 1, 1.05, 1, 0.8, '−1', '#FDECD6', ec=ACC, fs=15)
    ax.annotate('', xy=(0.98,2.45), xytext=(0.98,1.9), arrowprops=dict(arrowstyle='-|>',color=ACC,lw=1.8))
    # output row (bottom), positions t=1..4
    ax.text(-0.5, -0.5+0.5, 'output  $y$', ha='right', va='center', fontsize=13, color=INK)
    for i,v in enumerate(y):
        cell(ax, i, -0.5, 1, 1, f'{v}', GREEN if i==0 else '#EAF6EC', ec=INK)
        ax.text(i+0.5, -0.72, f'$y_{i+1}$', ha='center', va='center', fontsize=11, color=MUTED)
    # arrow window -> y1
    ax.annotate('', xy=(0.5,0.55), xytext=(0.5,1.0), arrowprops=dict(arrowstyle='-|>',color=GREEN,lw=1.8))
    # the rule + worked arithmetic, cleanly below everything
    ax.text(2.9, -1.35, r'$y_t \;=\; 2\,x_t \;-\; x_{t+1}$', ha='center', va='center', fontsize=15, color=INK, weight=600)
    ax.text(2.9, -2.05,
            r'$y_1{=}2{\cdot}1{-}2{=}0,\;\; y_2{=}2{\cdot}2{-}3{=}1,\;\; y_3{=}2{\cdot}3{-}4{=}2,\;\; y_4{=}2{\cdot}4{-}5{=}3$',
            ha='center', va='center', fontsize=12.5, color=MUTED)
    save(fig,'temporal_conv')

# ── 2 — receptive field vs #layers: ordinary (1+2L) vs dilated-exponential (2^L) ──
def f_rf_growth():
    L=np.arange(1,11)
    ordn=1+2*L               # ordinary causal, k=3
    dil=2.0**L               # dilated x2, k=2  -> R = 2^L
    fig,ax=plt.subplots(figsize=(6.9,3.5))
    ax.semilogy(L,dil,'-o',color=ACC,lw=2.8,ms=6,label=r'dilated $\times 2$:  $R=2^{L}$')
    ax.semilogy(L,ordn,'-o',color=TEAL,lw=2.8,ms=6,label=r'ordinary:  $R=1{+}2L$')
    # annotate the gap at L=10
    ax.annotate(r'$1024$',(10,1024),(8.1,1500),fontsize=11,color=ACC,fontweight='bold')
    ax.annotate(r'$21$',(10,21),(9.2,7.5),fontsize=11,color=TEAL,fontweight='bold')
    ax.set_xlim(0.6,10.6); ax.set_ylim(2,3000)
    ax.set_xticks(range(1,11))
    ax.set_xlabel('number of layers  $L$',fontsize=12)
    ax.set_ylabel('receptive field  $R_L$  (log)',fontsize=12)
    ax.set_title('depth buys context: linear vs exponential growth',fontsize=12.5)
    ax.legend(frameon=False,fontsize=11.5,loc='upper left')
    save(fig,'rf_growth')

# ── 3 — coverage: which past positions the top output reaches ──
def f_dilation_coverage():
    N=31  # positions t-30 .. t
    fig,ax=plt.subplots(figsize=(10.2,3.0))
    ax.set_xlim(-1.4,N+3.6); ax.set_ylim(-1.8,2.9); ax.set_aspect('equal'); ax.axis('off')
    w=1.0
    # top row: dilated stack d=[1,2,4,8] -> all 31 dense
    y1=1.4
    for i in range(N):
        cell(ax, i, y1, w, 0.9, None, TEAL, ec='white', lw=0.8)
    ax.text(-0.7, y1+0.45, 'dilated', ha='right', va='center', fontsize=12.5, color=INK, weight=600)
    ax.annotate('', xy=(0.05,y1+1.15), xytext=(N-0.05,y1+1.15),
                arrowprops=dict(arrowstyle='<->',color=INK,lw=1.4))
    ax.text(N/2, y1+1.5, r'$k{=}3,\ d{=}[1,2,4,8]\ \Rightarrow\ 31$ positions (dense)',
            ha='center', va='center', fontsize=12.5, color=TEAL, weight=600)
    # bottom row: ordinary 4 layers k=3 -> only rightmost 9
    y0=0.0
    for i in range(N):
        covered = i >= N-9
        cell(ax, i, y0, w, 0.9, None, ACC if covered else '#EFEEEB',
             ec='white' if covered else MUTED, lw=0.8)
    ax.text(-0.7, y0+0.45, 'ordinary', ha='right', va='center', fontsize=12.5, color=INK, weight=600)
    ax.annotate('', xy=(N-9+0.05,y0-0.35), xytext=(N-0.05,y0-0.35),
                arrowprops=dict(arrowstyle='<->',color=INK,lw=1.4))
    ax.text(N-4.5, y0-0.95, r'$4$ layers $\Rightarrow\ 9$ positions',
            ha='center', va='center', fontsize=12.5, color=ACC, weight=600)
    # output marker at position t (rightmost) — label to the RIGHT of both strips
    ax.text(N+1.9, 0.85, 'output\nat $t$', ha='center', va='center', fontsize=11.5, color=INK)
    ax.annotate('', xy=(N-0.05,y1+0.45), xytext=(N+1.0,0.95), arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.4))
    ax.annotate('', xy=(N-0.05,y0+0.45), xytext=(N+1.0,0.75), arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.4))
    # past/now axis labels
    ax.text(0.5, -1.35, r'$t{-}30$', ha='center', va='center', fontsize=11, color=MUTED)
    ax.text(N-0.5, -1.35, r'$t$', ha='center', va='center', fontsize=11, color=MUTED)
    ax.annotate('', xy=(N-0.5,-1.35), xytext=(1.6,-1.35), arrowprops=dict(arrowstyle='-|>',color=MUTED,lw=1.0))
    ax.text((N)/2, -1.72, 'time  (past $\\rightarrow$ present)', ha='center', va='center', fontsize=10.5, color=MUTED)
    save(fig,'dilation_coverage')

for f in [f_temporal_conv,f_rf_growth,f_dilation_coverage]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

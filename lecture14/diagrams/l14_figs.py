"""Metropolis-style figures for Lecture 14 (Seq2Seq, Teacher Forcing, Attention).
Schematic / synthetic ONLY — no real data. Transparent bg, ink + orange/teal.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Every STRUCTURAL diagram (encoder-decoder, attention mechanism, beam tree,
full seq2seq+attention, bottleneck, differentiable lookup) is NATIVE fletcher
in the deck. These three quantitative / schematic plots only:
  1. attention_heatmap    — U×T alignment matrix (I like cats -> j'aime les chats)
  2. attention_weights    — the worked bar chart alpha=[.0024,.0179,.9796] over h=[1,2,4]
  3. temperature_attention— same scores at tau=0.5/1/2, sharp -> diffuse
Run from repo root:  python3 lecture14/diagrams/l14_figs.py
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
CMAP=LinearSegmentedColormap.from_list('metro',['#2C7A7B','#EFEEEB','#EB811B'])
OUT='lecture14/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── 1 — attention alignment heatmap: A in R^{U x T} for the toy translation ──
def f_attention_heatmap():
    # cols = source (English) tokens, rows = target (French) tokens
    src=['I','like','cats','<EOS>']
    tgt=["j'aime",'les','chats','<EOS>']
    # each ROW (a target step) is a distribution over source positions -> sums to 1
    A=np.array([
        [0.34,0.56,0.06,0.04],   # j'aime  <- mostly "like" (+ a bit of "I")
        [0.06,0.12,0.72,0.10],   # les     <- "cats" (plural determiner)
        [0.05,0.06,0.85,0.04],   # chats   <- "cats"
        [0.04,0.05,0.11,0.80],   # <EOS>   <- "<EOS>"
    ])
    fig,ax=plt.subplots(figsize=(5.2,4.6))
    im=ax.imshow(A,cmap=CMAP,vmin=0,vmax=1,aspect='equal')
    ax.set_xticks(range(len(src))); ax.set_xticklabels(src,fontsize=13)
    ax.set_yticks(range(len(tgt))); ax.set_yticklabels(tgt,fontsize=13)
    ax.set_xlabel('source  $x_{1:T}$  (English)',fontsize=13)
    ax.set_ylabel('target  $y_{1:U}$  (French)',fontsize=13)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    # annotate each cell with the weight
    for u in range(A.shape[0]):
        for t in range(A.shape[1]):
            v=A[u,t]
            ax.text(t,u,f'{v:.2f}',ha='center',va='center',fontsize=11,
                    color='white' if v>0.55 else INK,weight=600)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,len(src),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(tgt),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2.5)
    ax.tick_params(which='minor',length=0)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cb.set_label(r'attention weight  $\alpha_{ut}$',fontsize=11)
    cb.outline.set_visible(False)
    save(fig,'attention_heatmap')

# ── 2 — the worked example: alpha over h=[1,2,4], spike on h_3 ──
def f_attention_weights():
    h=[1,2,4]; e=np.array([2.,4.,8.]); a=np.exp(e)/np.exp(e).sum()
    c=(a*np.array(h)).sum()
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8.8,3.4),gridspec_kw={'width_ratios':[1,1.05]})
    x=np.arange(3)
    # left: the encoder-state values h_t
    b1=ax1.bar(x,h,color=[TEAL,TEAL,ACC],width=0.62,edgecolor=INK,linewidth=1.2)
    ax1.set_xticks(x); ax1.set_xticklabels([r'$h_1$',r'$h_2$',r'$h_3$'],fontsize=14)
    ax1.set_title('encoder states  $h_t$   (query $q=2$)',fontsize=12.5)
    ax1.set_ylim(0,4.7)
    for xi,v in zip(x,h): ax1.text(xi,v+0.12,f'{v}',ha='center',fontsize=12,color=INK,weight=600)
    ax1.set_yticks([])
    # right: attention weights alpha_t
    ax2.bar(x,a,color=[TEAL,TEAL,ACC],width=0.62,edgecolor=INK,linewidth=1.2)
    ax2.set_xticks(x); ax2.set_xticklabels([r'$\alpha_1$',r'$\alpha_2$',r'$\alpha_3$'],fontsize=14)
    ax2.set_title(r'weights  $\alpha=\mathrm{softmax}(q\,h)$',fontsize=12.5)
    ax2.set_ylim(0,1.12)
    for xi,v in zip(x,a): ax2.text(xi,v+0.03,f'{v:.3f}',ha='center',fontsize=11.5,color=INK,weight=600)
    ax2.set_yticks([])
    ax2.text(0.02,0.60,f'$c=\\sum_t \\alpha_t h_t$\n$\\approx {c:.2f}\\;(\\approx h_3)$',
             transform=ax2.transAxes,fontsize=12.5,color=ACC,weight=600,va='center')
    save(fig,'attention_weights')

# ── 3 — temperature: same scores at tau=0.5/1/2, sharp -> diffuse ──
def f_temperature_attention():
    e=np.array([2.,4.,8.]); x=np.arange(3)
    taus=[(0.5,'sharp  $\\tau{=}0.5$',ACC),(1.0,'default  $\\tau{=}1$',TEAL),(2.0,'diffuse  $\\tau{=}2$',BLUE)]
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.0),sharey=True)
    for ax,(tau,ttl,c) in zip(axes,taus):
        a=np.exp(e/tau)/np.exp(e/tau).sum()
        ax.bar(x,a,color=c,width=0.62,edgecolor=INK,linewidth=1.1)
        ax.set_xticks(x); ax.set_xticklabels([r'$h_1$',r'$h_2$',r'$h_3$'],fontsize=13)
        ax.set_title(ttl,fontsize=12.5,color=c)
        ax.set_ylim(0,1.08)
        for xi,v in zip(x,a): ax.text(xi,v+0.03,f'{v:.2f}',ha='center',fontsize=10.5,color=INK)
        ax.set_yticks([])
    axes[0].set_ylabel(r'$\alpha=\mathrm{softmax}(e/\tau)$',fontsize=12)
    save(fig,'temperature_attention')

for f in [f_attention_heatmap,f_attention_weights,f_temperature_attention]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

"""Metropolis-style figures for Lecture 11 (Next-Token Prediction).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Schematic / synthetic only, on-palette. Run from repo root:
    python3 lecture11/diagrams/l11_figs.py
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture11/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def token_box(ax,x,y,w,h,label,fc,ec,tc='white',fs=12.5,alpha=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.03',
        linewidth=1.4,edgecolor=ec,facecolor=fc,alpha=alpha))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold')

# ── 1 — task taxonomy: four sequence tasks as input->output strips ──
def f_task_taxonomy():
    fig,axes=plt.subplots(2,2,figsize=(9.6,4.4))
    bw,bh,gap=0.9,0.62,0.14
    def strip(ax,x0,y0,tokens,fc,ec,tc='white',fs=12.5):
        for i,t in enumerate(tokens):
            token_box(ax,x0+i*(bw+gap),y0,bw,bh,t,fc,ec,tc,fs)
        return x0+len(tokens)*(bw+gap)-gap
    def arrow(ax,x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,
            lw=1.8,color=MUTED))
    cfgs=[
      ('classification   $x_(1:T) -> y$', ['$x_1$','$x_2$','$x_3$','$x_4$'], ['$y$'], BLUE, 'single label'),
      ('token classification   $-> y_(1:T)$', ['$x_1$','$x_2$','$x_3$','$x_4$'], ['$y_1$','$y_2$','$y_3$','$y_4$'], TEAL, 'one label / token'),
      ('seq2seq   $-> y_(1:U)$', ['$x_1$','$x_2$','$x_3$'], ['$y_1$','$y_2$'], GREEN, 'new length U'),
      ('next-token   $x_(1:t) -> x_(t+1)$', ['$x_1$','$x_2$','$x_3$'], ['$x_4$'], ACC, 'predict the next'),
    ]
    for ax,(title,xin,yout,col,note) in zip(axes.ravel(),cfgs):
        ax.set_xlim(0,7.2); ax.set_ylim(0,2.4); ax.axis('off')
        title=title.replace('$x_(1:T) -> y$',r'$x_{1:T}\to y$').replace('$-> y_(1:T)$',r'$\to y_{1:T}$')
        title=title.replace('$-> y_(1:U)$',r'$\to y_{1:U}$').replace('$x_(1:t) -> x_(t+1)$',r'$x_{1:t}\to x_{t+1}$')
        ax.set_title(title,fontsize=12.5,color=INK,pad=2)
        xin=[t.replace('$x_1$',r'$x_1$').replace('$x_2$',r'$x_2$').replace('$x_3$',r'$x_3$').replace('$x_4$',r'$x_4$') for t in xin]
        xr=strip(ax,0.1,1.35,xin,INK,INK)
        arrow(ax,xr+0.15,1.66,xr+0.85,1.66)
        highlight = col if 'next' in title.lower() else col
        strip(ax,xr+1.05,1.35,yout,highlight,highlight)
        ax.text(0.1,0.5,note,fontsize=11,color=MUTED,style='italic')
    fig.tight_layout(pad=0.6)
    save(fig,'task_taxonomy')

# ── 2 — one-hot (sparse) vs dense embedding (bars) ──
def f_one_hot_vs_embed():
    V=27; d=8; idx=3
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.4,3.0))
    oh=np.zeros(V); oh[idx]=1
    ax1.bar(range(V),oh,color=[ACC if i==idx else '#D9DCDC' for i in range(V)],
            edgecolor=INK,linewidth=.4,width=0.8)
    ax1.set_title(r'one-hot  $o_i\in\mathbb{R}^{V}$  (V=27)',fontsize=12.5)
    ax1.set_ylim(0,1.25); ax1.set_yticks([0,1]); ax1.set_xticks([idx]); ax1.set_xticklabels([f'id={idx}'])
    ax1.text(idx+1,1.05,'a single 1,\nrest zero',fontsize=10.5,color=MUTED)
    rng=np.random.default_rng(7); e=rng.normal(0,0.7,d)
    ax2.bar(range(d),e,color=[TEAL if v>=0 else ACC for v in e],edgecolor=INK,linewidth=.5,width=0.7)
    ax2.axhline(0,color=INK,lw=.8)
    ax2.set_title(r'dense embedding  $e_i\in\mathbb{R}^{d}$  (d=8)',fontsize=12.5)
    ax2.set_xticks(range(d)); ax2.set_yticks([])
    ax2.text(0.02,0.92,'few real numbers,\nall informative',transform=ax2.transAxes,fontsize=10.5,color=MUTED,va='top')
    fig.tight_layout(pad=0.8)
    save(fig,'one_hot_vs_embed')

# ── 3 — embedding table E with two looked-up rows highlighted ──
def f_embedding_table():
    V,d=8,4
    rng=np.random.default_rng(1); E=rng.normal(0,1,(V,d))
    hl=[1,4]  # rows looked up
    labels=['.', 'e','m','a','o','l','i','v']
    fig,ax=plt.subplots(figsize=(5.2,4.6))
    cw,ch=1.0,0.62
    vmax=np.abs(E).max()
    for r in range(V):
        for c in range(d):
            val=E[r,c]; t=(val/vmax+1)/2
            # teal (neg) -> cream -> orange (pos)
            col=np.array([44,122,123])/255*(1-t)+np.array([235,129,27])/255*t
            ax.add_patch(Rectangle((c*cw,(V-1-r)*ch),cw,ch,facecolor=col,edgecolor='white',linewidth=1.2))
            ax.text(c*cw+cw/2,(V-1-r)*ch+ch/2,f'{val:+.1f}',ha='center',va='center',fontsize=9,color='white')
        ax.text(-0.35,(V-1-r)*ch+ch/2,labels[r],ha='center',va='center',fontsize=12,
                color=(ACC if r in hl else INK),fontweight='bold')
    for r in hl:
        ax.add_patch(Rectangle((0,(V-1-r)*ch),d*cw,ch,facecolor='none',edgecolor=ACC,linewidth=3))
    ax.text(-0.9,V*ch/2,'token',rotation=90,ha='center',va='center',fontsize=11,color=MUTED)
    ax.text(d*cw/2,V*ch+0.12,r'$E\in\mathbb{R}^{V\times d}$   (rows = tokens, cols = features)',
            ha='center',fontsize=12,color=INK)
    ax.text(d*cw+0.15,(V-1-hl[0])*ch+ch/2,r'$\leftarrow e_{\mathrm{e}}$',fontsize=12,color=ACC,va='center')
    ax.text(d*cw+0.15,(V-1-hl[1])*ch+ch/2,r'$\leftarrow e_{\mathrm{o}}$',fontsize=12,color=ACC,va='center')
    ax.set_xlim(-1.1,d*cw+1.2); ax.set_ylim(-0.15,V*ch+0.4); ax.set_aspect('equal'); ax.axis('off')
    save(fig,'embedding_table')

# ── 4 — 2D embedding scatter: vowels vs consonants clustering + cos sim ──
def f_embedding_2d():
    # hand-placed so no labels overlap; two clusters that are clearly apart
    vowels={'a':(1.5,1.8),'e':(2.7,1.9),'i':(2.3,2.5),'o':(1.4,2.2),'u':(3.0,1.2)}
    cons={'b':(-1.1,-0.5),'d':(-1.9,-1.1),'g':(-2.4,-0.4),'k':(-1.4,-1.8),
          'm':(-2.6,-1.6),'n':(-0.9,-1.4),'r':(-2.0,-2.2),'t':(-1.2,-2.6),'s':(-2.9,-2.5)}
    fig,ax=plt.subplots(figsize=(6.0,4.4))
    for l,(x,y) in vowels.items():
        ax.text(x,y,l,fontsize=16,color=ACC,fontweight='bold',ha='center',va='center')
    for l,(x,y) in cons.items():
        ax.text(x,y,l,fontsize=15,color=TEAL,fontweight='bold',ha='center',va='center')
    # cosine similarity annotation between two nearby vowels a,e
    a=np.array(vowels['a']); b=np.array(vowels['e'])
    cos=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    ax.annotate('',xy=(b[0]-0.15,b[1]-0.1),xytext=(a[0]+0.15,a[1]+0.1),
                arrowprops=dict(arrowstyle='<->',color=MUTED,lw=1.4,alpha=.85))
    ax.text(2.75,0.75,f'cos sim $\\approx${cos:.2f}',fontsize=11,color=INK)
    ax.text(2.3,3.15,'vowels',fontsize=13,color=ACC,ha='center',fontweight='bold')
    ax.text(-1.9,0.15,'consonants',fontsize=13,color=TEAL,ha='center',fontweight='bold')
    ax.axhline(0,color=MUTED,lw=.5,alpha=.4); ax.axvline(0,color=MUTED,lw=.5,alpha=.4)
    ax.set_xlim(-3.7,3.9); ax.set_ylim(-3.4,3.7)
    ax.set_xlabel('embedding dim 1',fontsize=11); ax.set_ylabel('embedding dim 2',fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('learned 2-D character embeddings (schematic)',fontsize=12.5)
    save(fig,'embedding_2d')

# ── 5 — temperature: same logits -> softmax bars at tau=0.5,1,2 ──
def f_temperature():
    z=np.array([2.0,1.0,0.3,-0.5,-1.2])
    labels=['a','e','i','o','u']
    taus=[0.5,1.0,2.0]
    fig,axes=plt.subplots(1,3,figsize=(9.8,3.0),sharey=True)
    for ax,tau in zip(axes,taus):
        zz=z/tau; p=np.exp(zz-zz.max()); p/=p.sum()
        ax.bar(range(len(z)),p,color=[ACC if i==0 else TEAL for i in range(len(z))],
               edgecolor=INK,linewidth=.5,width=0.72)
        ax.set_xticks(range(len(z))); ax.set_xticklabels(labels)
        title={0.5:r'$\tau=0.5$  (sharper)',1.0:r'$\tau=1.0$  (base)',2.0:r'$\tau=2.0$  (flatter)'}[tau]
        ax.set_title(title,fontsize=12.5)
        ax.set_ylim(0,1.0)
    axes[0].set_ylabel('probability',fontsize=11)
    fig.suptitle(r'same logits $z=[2,\,1,\,0.3,\,-0.5,\,-1.2]$,  softmax$(z/\tau)$',fontsize=12,y=1.02)
    fig.tight_layout(pad=0.6)
    save(fig,'temperature')

# ── 6 — perplexity: confident vs uncertain distribution, PPL = exp(NLL) ──
def f_perplexity():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.4,3.0),sharey=True)
    V=10; xs=np.arange(V)
    # confident: peaked -> low perplexity
    pc=np.array([0.62,0.14,0.09,0.05,0.03,0.02,0.02,0.01,0.01,0.01]); pc/=pc.sum()
    # uncertain: near-uniform -> high perplexity
    pu=np.ones(V)/V
    for ax,p,ttl,c in [(ax1,pc,'confident model',TEAL),(ax2,pu,'uncertain model',ACC)]:
        nll=-np.sum(p*np.log(p)); ppl=np.exp(nll)
        ax.bar(xs,p,color=c,edgecolor=INK,linewidth=.5,width=0.75)
        ax.set_title(ttl,fontsize=12.5)
        ax.set_ylim(0,0.72); ax.set_xticks([])
        ax.text(0.5,0.9,f'NLL $=$ {nll:.2f}\nPPL $=e^{{\\mathrm{{NLL}}}}\\approx$ {ppl:.1f}',
                transform=ax.transAxes,fontsize=12,color=INK,ha='center',va='top',
                bbox=dict(boxstyle='round,pad=0.4',fc='white',ec=c,lw=1.5))
    ax1.set_ylabel('next-token prob',fontsize=11)
    fig.suptitle(r'perplexity = "effective number of plausible next tokens" $=\exp(\mathrm{NLL})$',fontsize=12,y=1.02)
    fig.tight_layout(pad=0.6)
    save(fig,'perplexity')

for f in [f_task_taxonomy,f_one_hot_vs_embed,f_embedding_table,f_embedding_2d,
          f_temperature,f_perplexity]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

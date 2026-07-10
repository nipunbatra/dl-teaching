"""Metropolis-style figures for Lecture 18 (Self-Supervised & Representation Learning).
Schematic / synthetic ONLY — NO real images. "Images" are colored patch grids.
Transparent bg, ink + orange/teal palette. Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Every STRUCTURAL diagram (SimCLR pipeline, contrastive batch, MAE asymmetric encoder/decoder,
BYOL teacher-student) is NATIVE fletcher in the deck. These schematic / matrix plots only:
  1. temperature_contrast — the board example: softmax over sims [0.9, 0.2, -0.1] at tau=1 vs tau=0.1
  2. sim_matrix          — a 2N x 2N contrastive similarity matrix, positive pairs highlighted
  3. mae_mask            — a 14x14 patch grid with 75% masked (muted) and 25% visible
Run from repo root:  python3 lecture18/diagrams/l18_figs.py
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
OUT='lecture18/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def softmax(x):
    e=np.exp(x-np.max(x)); return e/e.sum()

# ── 1 — temperature is a contrast amplifier: softmax([0.9, 0.2, -0.1]) at tau=1 vs tau=0.1 ──
def f_temperature_contrast():
    sims=np.array([0.9,0.2,-0.1])
    labels=['positive\n(s = 0.9)','neg\n(0.2)','neg\n(-0.1)']
    colors=[ACC,TEAL,BLUE]
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.3))
    for ax,tau in zip(axes,[1.0,0.1]):
        p=softmax(sims/tau)
        b=ax.bar(range(3),p,color=colors,edgecolor='white',linewidth=1.4,width=0.66)
        for bar,v in zip(b,p):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.03, f'{v:.3f}',
                    ha='center',fontsize=12,color=INK,weight=600)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels,fontsize=11)
        ax.set_ylim(0,1.15); ax.set_yticks([0,0.5,1.0])
        ax.set_title(rf'$\tau={tau}$   $\to$   $p(\mathrm{{pos}})\approx{p[0]:.3f}$',
                     fontsize=13,color=INK)
        ax.spines['left'].set_color(MUTED)
    axes[0].set_ylabel('softmax probability',fontsize=12)
    fig.suptitle(r'same similarities, two temperatures — smaller $\tau$ sharpens the competition',
                 fontsize=12.5,color=MUTED,y=1.02)
    fig.tight_layout()
    save(fig,'temperature_contrast')

# ── 2 — contrastive similarity matrix: 2N x 2N, positive pairs highlighted ──
def f_sim_matrix():
    # N=4 sources A,B,C,D -> 2N=8 views, ordered A1,A2,B1,B2,C1,C2,D1,D2.
    # positive partner of view v is its sibling: 2k <-> 2k+1.
    N=4; M=2*N; rng=np.random.default_rng(5)
    S=rng.uniform(0.05,0.35,(M,M))
    for k in range(N):                      # positive pairs get high similarity
        a,b=2*k,2*k+1
        val=rng.uniform(0.80,0.95)
        S[a,b]=val; S[b,a]=val
    S=(S+S.T)/2
    np.fill_diagonal(S,1.0)                  # self-similarity (ignored)
    labels=['A1','A2','B1','B2','C1','C2','D1','D2']
    fig,ax=plt.subplots(figsize=(5.2,4.7))
    Sm=np.ma.array(S,mask=np.eye(M,dtype=bool))   # mask diagonal so it reads as "ignored"
    im=ax.imshow(Sm,cmap=CMAP,vmin=0,vmax=1,aspect='equal')
    # grey out the ignored diagonal
    for i in range(M):
        ax.add_patch(plt.Rectangle((i-0.5,i-0.5),1,1,facecolor='#D9D6D0',edgecolor='white',
                     hatch='///',linewidth=0.8))
    # highlight the positive pairs (off-diagonal siblings)
    for k in range(N):
        a,b=2*k,2*k+1
        for (i,j) in [(a,b),(b,a)]:
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor=INK,linewidth=3.0))
    ax.set_xticks(range(M)); ax.set_xticklabels(labels,fontsize=10.5)
    ax.set_yticks(range(M)); ax.set_yticklabels(labels,fontsize=10.5)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    ax.set_xlabel('view  $z_j$',fontsize=12); ax.set_ylabel('anchor  $z_i$',fontsize=12)
    ax.set_xticks(np.arange(-.5,M,1),minor=True); ax.set_yticks(np.arange(-.5,M,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.6); ax.tick_params(which='minor',length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title('one positive per row (boxed); diagonal ignored;\nall else are negatives',
                 fontsize=11,color=INK,pad=10)
    save(fig,'sim_matrix')

# ── 3 — MAE masking: 14x14 patch grid, 75% masked (muted) / 25% visible ──
def f_mae_mask():
    rng=np.random.default_rng(3)
    G=14; ntot=G*G                          # 196 patches
    nvis=ntot//4                            # 49 visible (25%)
    idx=rng.permutation(ntot)
    visible=set(idx[:nvis].tolist())
    # give visible patches a soft colored value; masked = muted grey
    fig,ax=plt.subplots(figsize=(4.6,4.6))
    for i in range(G):
        for j in range(G):
            p=i*G+j
            if p in visible:
                fc=CMAP(rng.uniform(0.15,0.9))
                ax.add_patch(plt.Rectangle((j,G-1-i),1,1,facecolor=fc,edgecolor='white',linewidth=0.7))
            else:
                ax.add_patch(plt.Rectangle((j,G-1-i),1,1,facecolor='#DCD9D3',edgecolor='white',linewidth=0.7))
    ax.set_xlim(-0.15,G+0.15); ax.set_ylim(-0.15,G+0.15); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(r'$14\times14 = 196$ patches · 75% masked $\to$ 49 visible',
                 fontsize=12,color=INK,pad=8)
    # legend chips
    ax.add_patch(plt.Rectangle((1.0,-1.35),0.9,0.9,facecolor=CMAP(0.75),edgecolor='white',clip_on=False))
    ax.text(2.1,-0.9,'visible',fontsize=10.5,color=INK,va='center')
    ax.add_patch(plt.Rectangle((7.3,-1.35),0.9,0.9,facecolor='#DCD9D3',edgecolor='white',clip_on=False))
    ax.text(8.4,-0.9,'masked',fontsize=10.5,color=MUTED,va='center')
    save(fig,'mae_mask')

for f in [f_temperature_contrast,f_sim_matrix,f_mae_mask]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

"""Metropolis-style figures for Lecture 16 (Transformers II: BERT, GPT, Pretraining, Finetuning, Generation).
Schematic / synthetic ONLY — no real data. Transparent bg, ink + orange/teal.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Every STRUCTURAL diagram (three-families, cross-attention, pretrain->finetune,
generation loop) is NATIVE fletcher in the deck. These matrix / schematic plots only:
  1. attention_masks — three masks side by side: bidirectional / causal / cross-attention
  2. mlm_vs_clm      — schematic: MLM sees both sides (mask in middle) vs CLM sees left only
Run from repo root:  python3 lecture16/diagrams/l16_figs.py
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
OUT='lecture16/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

tealrgb=np.array([44,122,123])/255.; mutedrgb=np.array([110,127,130])/255.

def _draw_mask(ax, allowed, xlabels, ylabels, xlab, ylab, title):
    """allowed: 2D 0/1 array. teal tint = attend, muted tint = masked."""
    R,C=allowed.shape
    img=np.zeros((R,C,4))
    for i in range(R):
        for j in range(C):
            img[i,j]=[*tealrgb,0.85] if allowed[i,j] else [*mutedrgb,0.28]
    ax.imshow(img,aspect='equal')
    ax.set_xticks(range(C)); ax.set_xticklabels(xlabels,fontsize=11)
    ax.set_yticks(range(R)); ax.set_yticklabels(ylabels,fontsize=11)
    ax.set_xlabel(xlab,fontsize=12); ax.set_ylabel(ylab,fontsize=12)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    for i in range(R):
        for j in range(C):
            if allowed[i,j]:
                ax.text(j,i,r'$\checkmark$',ha='center',va='center',fontsize=13,color='white',weight=700)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,C,1),minor=True); ax.set_yticks(np.arange(-.5,R,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2.4); ax.tick_params(which='minor',length=0)
    ax.set_title(title,fontsize=12.5,pad=22)

# ── 1 — three masks side by side: bidirectional / causal / cross-attention ──
def f_attention_masks():
    T=5
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.4))
    # bidirectional (encoder): every token attends every token
    full=np.ones((T,T))
    _draw_mask(axes[0],full,[f'{j+1}' for j in range(T)],[f'{i+1}' for i in range(T)],
               r'key $j$',r'query $i$','bidirectional (encoder)\nBERT — no mask')
    # causal (decoder): lower-triangular
    causal=np.tril(np.ones((T,T)))
    _draw_mask(axes[1],causal,[f'{j+1}' for j in range(T)],[f'{i+1}' for i in range(T)],
               r'key $j$',r'query $i$','causal (decoder)\nGPT — mask future')
    # cross-attention: decoder targets (rows) x encoder source (cols), full (T_dec x T_enc)
    U,Tenc=4,5
    cross=np.ones((U,Tenc))
    _draw_mask(axes[2],cross,[f'{j+1}' for j in range(Tenc)],[f'{u+1}' for u in range(U)],
               r'source (encoder) $j$',r'target (decoder) $u$','cross-attention\ndecoder $\\to$ encoder')
    save(fig,'attention_masks')

# ── 2 — MLM vs CLM schematic: which positions each objective may read ──
def f_mlm_vs_clm():
    from matplotlib.path import Path
    import matplotlib.patches as mpatches
    toks=['deep','learning','is','very','useful']
    T=len(toks)
    fig,axes=plt.subplots(1,2,figsize=(11.0,3.4))

    def bow(ax,x0,x1,c):
        """quadratic bezier from top of box x0 up over to top of box x1 — always bows upward."""
        peak=1.05
        verts=[(x0,0.40),((x0+x1)/2.0,peak),(x1,0.46)]
        codes=[Path.MOVETO,Path.CURVE3,Path.CURVE3]
        ax.add_patch(mpatches.FancyArrowPatch(path=Path(verts,codes),
            arrowstyle='-|>',mutation_scale=13,color=c,lw=1.3,alpha=0.75))

    def panel(ax,title,target_idx,source_idx,masked_idx=None):
        ax.set_xlim(-0.65,T-0.35); ax.set_ylim(-0.9,1.5)
        for t,w in enumerate(toks):
            is_masked = (masked_idx is not None and t==masked_idx)
            is_target = (t==target_idx)
            is_src = (t in source_idx)
            if is_masked:
                fc=RED; txt='[MASK]'; tc='white'
            elif is_target:
                fc=ACC; txt=w; tc='white'
            elif is_src:
                fc=TEAL; txt=w; tc='white'
            else:
                fc='none'; txt=w; tc=MUTED
            ec = MUTED if fc=='none' else fc
            ax.add_patch(mpatches.FancyBboxPatch((t-0.42,-0.34),0.84,0.68,
                boxstyle='round,pad=0.02,rounding_size=0.08',
                linewidth=1.4,edgecolor=ec,facecolor=fc if fc!='none' else 'none',
                alpha=0.92 if fc!='none' else 1.0))
            ax.text(t,0,txt,ha='center',va='center',fontsize=10.5,color=tc,weight=600)
        # bows from every source token up over to the prediction target
        for t in source_idx:
            bow(ax,t,target_idx,TEAL)
        ax.set_title(title,fontsize=12.5,pad=8)
        ax.axis('off')

    # MLM: predict masked token in the MIDDLE, see BOTH sides
    panel(axes[0],'MLM (BERT): predict [MASK] from both sides',target_idx=1,
          source_idx=[0,2,3,4],masked_idx=1)
    # CLM: predict NEXT token (position 3), see LEFT only
    panel(axes[1],'CLM (GPT): predict next token from the left only',target_idx=3,
          source_idx=[0,1,2])
    save(fig,'mlm_vs_clm')

for f in [f_attention_masks,f_mlm_vs_clm]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

"""Metropolis-style figures for Lecture 15 (Transformers I: Self-Attention, PE, Blocks).
Schematic / synthetic ONLY — no real data. Transparent bg, ink + orange/teal.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Every STRUCTURAL diagram (Q/K/V lookup, RNN-vs-attention, multi-head block,
encoder block, decoder-only pipeline) is NATIVE fletcher in the deck.
These quantitative / schematic matrix plots only:
  1. attn_matrix   — worked self-attention QK^T = [[1,0,1],[0,1,1],[1,1,2]]
  2. causal_mask   — T=5 lower-triangular causal mask (allowed vs -inf)
  3. attn_heatmap  — T x T self-attention weights (softmaxed, rows sum to 1)
  4. pe_waves      — sinusoidal PE curves + PP^T position-similarity matrix
Run from repo root:  python3 lecture15/diagrams/l15_figs.py
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
OUT='lecture15/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def softmax(x):
    x=np.asarray(x,float); e=np.exp(x-x.max()); return e/e.sum()

# ── 1 — worked self-attention scores QK^T (unscaled), X=[[1,0],[0,1],[1,1]] ──
def f_attn_matrix():
    S=np.array([[1,0,1],[0,1,1],[1,1,2]],float)
    labels=[r'$x_1$',r'$x_2$',r'$x_3$']
    fig,ax=plt.subplots(figsize=(4.4,4.0))
    im=ax.imshow(S,cmap=CMAP,vmin=0,vmax=2,aspect='equal')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels,fontsize=14)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels,fontsize=14)
    ax.set_xlabel(r'key  $k_j$',fontsize=13); ax.set_ylabel(r'query  $q_i$',fontsize=13)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    for i in range(3):
        for j in range(3):
            v=S[i,j]
            ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=15,
                    color='white' if v>1.3 else INK,weight=700)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,3,1),minor=True); ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2.5); ax.tick_params(which='minor',length=0)
    save(fig,'attn_matrix')

# ── 2 — causal mask: T=5 lower-triangular (allowed vs masked -inf) ──
def f_causal_mask():
    T=5
    allowed=np.tril(np.ones((T,T)))          # 1 where j<=i
    # RGBA image: allowed -> teal tint, masked -> muted tint
    img=np.zeros((T,T,4))
    tealrgb=np.array([44,122,123])/255.; mutedrgb=np.array([110,127,130])/255.
    for i in range(T):
        for j in range(T):
            if allowed[i,j]: img[i,j]=[*tealrgb,0.85]
            else:            img[i,j]=[*mutedrgb,0.30]
    fig,ax=plt.subplots(figsize=(4.6,4.2))
    ax.imshow(img,aspect='equal')
    ax.set_xticks(range(T)); ax.set_xticklabels([f'{j+1}' for j in range(T)],fontsize=12)
    ax.set_yticks(range(T)); ax.set_yticklabels([f'{i+1}' for i in range(T)],fontsize=12)
    ax.set_xlabel(r'key position  $j$',fontsize=13); ax.set_ylabel(r'query position  $i$',fontsize=13)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    for i in range(T):
        for j in range(T):
            if allowed[i,j]:
                ax.text(j,i,r'$\checkmark$',ha='center',va='center',fontsize=15,color='white',weight=700)
            else:
                ax.text(j,i,r'$-\infty$',ha='center',va='center',fontsize=12,color=RED,weight=700)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,T,1),minor=True); ax.set_yticks(np.arange(-.5,T,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2.5); ax.tick_params(which='minor',length=0)
    save(fig,'causal_mask')

# ── 3 — T x T self-attention weights (softmaxed rows, sums to 1) ──
def f_attn_heatmap():
    toks=['the','cat','sat','on','the','mat']
    T=len(toks)
    # synthetic score matrix with structure: strong self + a few content links
    rng=np.random.default_rng(7)
    base=rng.normal(0,0.25,(T,T))
    base+=np.eye(T)*1.4                       # self-attention ridge
    base[1,0]+=1.8; base[1,4]+=0.6            # cat -> the
    base[2,1]+=2.0                            # sat -> cat (subject)
    base[3,2]+=1.5                            # on  -> sat
    base[5,3]+=1.6; base[5,4]+=1.2            # mat -> on / the
    base[4,0]+=1.0                            # the -> the
    A=np.array([softmax(r) for r in base])
    fig,ax=plt.subplots(figsize=(5.2,4.8))
    im=ax.imshow(A,cmap=CMAP,vmin=0,vmax=A.max(),aspect='equal')
    ax.set_xticks(range(T)); ax.set_xticklabels(toks,fontsize=12,rotation=0)
    ax.set_yticks(range(T)); ax.set_yticklabels(toks,fontsize=12)
    ax.set_xlabel(r'attends to  (key $j$)',fontsize=12.5); ax.set_ylabel(r'token  (query $i$)',fontsize=12.5)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    for i in range(T):
        for j in range(T):
            v=A[i,j]
            if v>0.12:
                ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8.5,
                        color='white' if v>0.55*A.max() else INK,weight=600)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,T,1),minor=True); ax.set_yticks(np.arange(-.5,T,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2.2); ax.tick_params(which='minor',length=0)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cb.set_label(r'$\alpha_{ij}$  (each row sums to 1)',fontsize=10.5); cb.outline.set_visible(False)
    save(fig,'attn_heatmap')

# ── 4 — sinusoidal PE: curves at several frequencies + PP^T similarity ──
def f_pe_waves():
    Tmax=48; d=64
    pos=np.arange(Tmax)
    def pe(pos,d):
        P=np.zeros((len(pos),d))
        for i in range(d//2):
            w=1.0/(10000**(2*i/d))
            P[:,2*i]=np.sin(pos*w); P[:,2*i+1]=np.cos(pos*w)
        return P
    P=pe(pos,d)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,3.6),gridspec_kw={'width_ratios':[1.15,1.0]})
    # left: a few dimensions (frequencies), fast -> slow, on a fine grid (smooth curves)
    posf=np.linspace(0,Tmax-1,600); Pf=pe(posf,d)
    dims=[(0,'sin, dim 0  (fast)',ACC),(8,'sin, dim 8',TEAL),
          (16,'sin, dim 16',BLUE),(24,'sin, dim 24  (slow)',GREEN)]
    for k,lbl,c in dims:
        ax1.plot(posf,Pf[:,k],color=c,lw=2.4,label=lbl)
    ax1.axhline(0,color=MUTED,lw=.6,alpha=.5)
    ax1.set_xlim(0,Tmax-1); ax1.set_ylim(-1.25,1.35)
    ax1.set_xlabel('position  $pos$',fontsize=12); ax1.set_ylabel(r'$PE_{(pos,\cdot)}$',fontsize=12)
    ax1.set_yticks([-1,0,1])
    ax1.set_title('each dimension = a different frequency',fontsize=12.5)
    ax1.legend(frameon=False,fontsize=9.5,loc='lower left',ncol=1)
    # right: PP^T position-similarity (normalized), banded -> nearby positions similar
    G=P@P.T
    G=G/np.max(np.abs(G))
    im=ax2.imshow(G,cmap=CMAP,vmin=-1,vmax=1,aspect='equal')
    ax2.set_xlabel('position  $j$',fontsize=12); ax2.set_ylabel('position  $i$',fontsize=12)
    ax2.set_xticks([0,Tmax//2,Tmax-1]); ax2.set_yticks([0,Tmax//2,Tmax-1])
    ax2.set_title(r'similarity  $P P^\top$  (banded)',fontsize=12.5)
    for s in ax2.spines.values(): s.set_visible(False)
    cb=fig.colorbar(im,ax=ax2,fraction=0.046,pad=0.04); cb.outline.set_visible(False)
    save(fig,'pe_waves')

for f in [f_attn_matrix,f_causal_mask,f_attn_heatmap,f_pe_waves]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

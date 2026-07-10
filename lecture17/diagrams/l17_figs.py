"""Metropolis-style figures for Lecture 17 (Vision Transformers & Multimodal Models).
Schematic / synthetic ONLY — NO real images. "Images" are colored patch grids.
Transparent bg, ink + orange/teal palette. Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Every STRUCTURAL diagram (image->token pipeline, patchification flow, ViT encoder block,
dual-encoder, VLM pipeline, BLIP-2 Q-Former) is NATIVE fletcher in the deck. These
schematic / matrix plots only:
  1. patchify      — colored image grid split into patches, one highlighted -> flatten -> project token
  2. patch_as_conv — strided P x P convolution IS the patch embedding (kernel = stride = P)
  3. attn_cost     — N and N^2 for P=16 vs P=8 (log y, ~16x gap in the attention matrix)
  4. clip_matrix   — B x B image-text similarity matrix, diagonal (matched pairs) highlighted
  5. windows       — standard vs shifted window partition on a patch grid (Swin)
Run from repo root:  python3 lecture17/diagrams/l17_figs.py
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
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
OUT='lecture17/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def cellgrid(ax, ox, oy, vals, cell, ec='white', lw=0.6):
    """Draw a grid of colored cells; vals is (rows, cols) in [0,1]. Top row at oy (grows down)."""
    R,C=vals.shape
    for i in range(R):
        for j in range(C):
            ax.add_patch(plt.Rectangle((ox+j*cell, oy-(i+1)*cell), cell, cell,
                         facecolor=CMAP(vals[i,j]), edgecolor=ec, linewidth=lw))

def arrow(ax,a,b,c=INK,lw=2.4):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=c,lw=lw))

# ── 1 — patchify: image grid -> highlight one patch -> flatten -> project to a token ──
def f_patchify():
    rng=np.random.default_rng(7)
    G,P=8,2                       # 8x8 "image", 2x2 patches -> 4x4 = 16 patches
    vals=rng.random((G,G))
    cell=0.5; top=4.0
    fig,ax=plt.subplots(figsize=(10.2,3.4))
    # image
    cellgrid(ax,0,top,vals,cell)
    for k in range(0,G+1,P):
        ax.plot([0,G*cell],[top-k*cell,top-k*cell],color=INK,lw=1.5)
        ax.plot([k*cell,k*cell],[top,top-G*cell],color=INK,lw=1.5)
    # highlight top-left patch (rows 0-1, cols 0-1)
    ax.add_patch(plt.Rectangle((0,top-P*cell),P*cell,P*cell,fill=False,edgecolor=ACC,linewidth=3.4))
    ax.text(G*cell/2,top+0.34,'image  =  grid of patches',ha='center',fontsize=12.5,color=INK,weight=600)
    ax.text(G*cell/2,-0.55,r'$H\times W\times C$',ha='center',fontsize=12,color=MUTED)
    # the highlighted patch's pixels (2x2 = 4 values) flattened into a column
    pv=np.array([vals[0,0],vals[0,1],vals[1,0],vals[1,1]]).reshape(4,1)
    fx,ftop=5.6,3.4
    cellgrid(ax,fx,ftop,pv,cell)
    ax.add_patch(plt.Rectangle((fx,ftop-4*cell),cell,4*cell,fill=False,edgecolor=ACC,linewidth=1.8))
    ax.text(fx+cell/2,ftop+0.32,'flatten',ha='center',fontsize=12,color=INK,weight=600)
    ax.text(fx+cell/2,ftop-4*cell-0.34,r'$x_i\in\mathbb{R}^{P^2C}$',ha='center',fontsize=11.5,color=MUTED)
    arrow(ax,(4.15,top-cell),(fx-0.28,ftop-2*cell),ACC)
    # projected token z_i (3-dim)
    tv=np.array([0.85,0.25,0.62]).reshape(3,1)
    tx,ttop=7.9,3.0
    cellgrid(ax,tx,ttop,tv,cell)
    ax.add_patch(plt.Rectangle((tx,ttop-3*cell),cell,3*cell,fill=False,edgecolor=TEAL,linewidth=2.2))
    ax.text(tx+cell/2,ttop+0.32,r'$z_i = x_i E$',ha='center',fontsize=12,color=INK,weight=600)
    ax.text(tx+cell/2,ttop-3*cell-0.34,r'$z_i\in\mathbb{R}^{d}$',ha='center',fontsize=11.5,color=MUTED)
    arrow(ax,(fx+cell+0.22,ftop-2*cell),(tx-0.28,ttop-1.5*cell),TEAL)
    # patch sequence strip
    sy=0.15
    for k in range(6):
        ax.add_patch(plt.Rectangle((5.6+k*0.55,sy),0.42,0.42,facecolor=TEAL,
                     edgecolor='white',linewidth=1.0,alpha=0.85))
    ax.text(5.6+3*0.55,sy-0.42,r'patch token sequence  $z_1\,z_2\,\dots\,z_N$',ha='center',fontsize=11,color=MUTED)
    ax.set_xlim(-0.4,9.1); ax.set_ylim(-1.1,4.9); ax.set_aspect('equal'); ax.axis('off')
    save(fig,'patchify')

# ── 2 — patch embedding IS a strided convolution: P x P kernel, stride P, C_out = d ──
def f_patch_as_conv():
    rng=np.random.default_rng(2)
    G,P=8,2
    vals=rng.random((G,G))
    cell=0.5; top=4.0
    fig,ax=plt.subplots(figsize=(9.4,3.5))
    cellgrid(ax,0,top,vals,cell)
    for k in range(0,G+1,P):
        ax.plot([0,G*cell],[top-k*cell,top-k*cell],color=INK,lw=1.4)
        ax.plot([k*cell,k*cell],[top,top-G*cell],color=INK,lw=1.4)
    # the kernel window sitting on the top-left patch, no overlap (stride = P)
    ax.add_patch(plt.Rectangle((0,top-P*cell),P*cell,P*cell,fill=False,edgecolor=RED,linewidth=3.2))
    ax.text(G*cell/2,top+0.34,r'input  $224\times224\times3$',ha='center',fontsize=12,color=INK,weight=600)
    ax.text(G*cell/2,-0.5,r'kernel $=P\times P$, stride $=P$  (non-overlapping)',ha='center',fontsize=11,color=RED)
    # output feature map: 4x4 x d  (one d-vector per patch position)
    fvals=rng.random((4,4)); fx,ftop=6.4,3.2; fc=0.42
    cellgrid(ax,fx,ftop,fvals,fc)
    for k in range(5):
        ax.plot([fx,fx+4*fc],[ftop-k*fc,ftop-k*fc],color=INK,lw=0.8)
        ax.plot([fx+k*fc,fx+k*fc],[ftop,ftop-4*fc],color=INK,lw=0.8)
    ax.add_patch(plt.Rectangle((fx,ftop-fc),fc,fc,fill=False,edgecolor=RED,linewidth=2.6))
    ax.text(fx+2*fc,ftop+0.34,r'$14\times14\times d$',ha='center',fontsize=12,color=INK,weight=600)
    ax.text(fx+2*fc,ftop-4*fc-0.42,r'flatten $\to$ 196 tokens',ha='center',fontsize=11,color=MUTED)
    arrow(ax,(4.15,top-cell),(fx-0.3,ftop-2*fc),INK)
    ax.text((4.15+fx)/2,top-cell+0.5,'strided conv',ha='center',fontsize=11.5,color=INK,weight=600)
    ax.set_xlim(-0.4,8.6); ax.set_ylim(-1.0,4.9); ax.set_aspect('equal'); ax.axis('off')
    save(fig,'patch_as_conv')

# ── 3 — attention cost: N and N^2 for P=16 vs P=8 (log y, ~16x gap) ──
def f_attn_cost():
    labels=['N\n(tokens)',r'$N^2$'+'\n(attention entries)']
    p16=[196,38416]; p8=[784,614656]
    x=np.arange(2); w=0.34
    fig,ax=plt.subplots(figsize=(7.2,3.4))
    b1=ax.bar(x-w/2,p16,w,color=TEAL,label='P = 16  (N = 196)',edgecolor='white',linewidth=1.2)
    b2=ax.bar(x+w/2,p8,w,color=ACC,label='P = 8  (N = 784)',edgecolor='white',linewidth=1.2)
    ax.set_yscale('log'); ax.set_ylim(50,3e6)
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=12)
    ax.set_ylabel('count (log scale)',fontsize=12)
    for b,v in zip(b1,p16): ax.text(b.get_x()+b.get_width()/2,v*1.25,f'{v:,}',ha='center',fontsize=10,color=TEAL,weight=600)
    for b,v in zip(b2,p8):  ax.text(b.get_x()+b.get_width()/2,v*1.25,f'{v:,}',ha='center',fontsize=10,color=ACC,weight=600)
    # annotate the ~16x gap on N^2
    ax.annotate('',xy=(1+w/2,614656),xytext=(1+w/2,38416),
                arrowprops=dict(arrowstyle='<->',color=INK,lw=1.8))
    ax.text(1.34,150000,r'$\approx 16\times$',fontsize=13,color=INK,weight=700)
    ax.legend(frameon=False,fontsize=11,loc='upper left')
    ax.set_title(r'halving the patch size $P$ quadruples $N$ and $16\times$es $N^2$',fontsize=12)
    save(fig,'attn_cost')

# ── 4 — CLIP similarity matrix: B x B, diagonal (matched pairs) is the positive set ──
def f_clip_matrix():
    B=6; rng=np.random.default_rng(11)
    S=rng.uniform(0.02,0.34,(B,B))
    np.fill_diagonal(S,rng.uniform(0.82,0.98,B))
    fig,ax=plt.subplots(figsize=(5.0,4.4))
    im=ax.imshow(S,cmap=CMAP,vmin=0,vmax=1,aspect='equal')
    ax.set_xticks(range(B)); ax.set_xticklabels([f'$T_{j+1}$' for j in range(B)],fontsize=11)
    ax.set_yticks(range(B)); ax.set_yticklabels([f'$I_{i+1}$' for i in range(B)],fontsize=11)
    ax.set_xlabel('text captions',fontsize=12); ax.set_ylabel('images',fontsize=12)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    # highlight the diagonal (matched positives)
    for i in range(B):
        ax.add_patch(plt.Rectangle((i-0.5,i-0.5),1,1,fill=False,edgecolor=INK,linewidth=2.8))
    ax.set_xticks(np.arange(-.5,B,1),minor=True); ax.set_yticks(np.arange(-.5,B,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.6); ax.tick_params(which='minor',length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(r'$S_{ij}=\hat v_i^\top \hat t_j/\tau$  —  maximize the diagonal',fontsize=11.5,pad=26)
    save(fig,'clip_matrix')

# ── 5 — standard vs shifted windows (Swin) on a patch grid ──
def f_windows():
    G=8; M=4                     # 8x8 patches, 4x4 windows
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.2))
    wcolors=[TEAL,ACC,BLUE,GREEN]
    def patchgrid(ax):
        for i in range(G):
            for j in range(G):
                ax.add_patch(plt.Rectangle((j,G-1-i),1,1,facecolor='#EFEEEB',edgecolor='white',linewidth=0.8))
    def window(ax,x0,y0,w,h,c):
        ax.add_patch(plt.Rectangle((x0,y0),w,h,facecolor=c,edgecolor=INK,linewidth=2.6,alpha=0.30))
        ax.add_patch(plt.Rectangle((x0,y0),w,h,fill=False,edgecolor=INK,linewidth=2.6))
    # ---- standard: 4 clean MxM windows ----
    ax=axes[0]; patchgrid(ax)
    ci=0
    for by in (0,M):
        for bx in (0,M):
            window(ax,bx,by,M,M,wcolors[ci%4]); ci+=1
    ax.set_title('standard windows\nattention inside each window',fontsize=12,color=INK)
    # ---- shifted: partition shifted by M/2 -> 9 unequal windows, neighbors now mix ----
    ax=axes[1]; patchgrid(ax)
    sh=M//2
    xs=[0,sh,sh+M]; widths=[sh,M,sh]
    ys=[0,sh,sh+M]; heights=[sh,M,sh]
    ci=0
    for (y0,h) in zip(ys,heights):
        for (x0,w) in zip(xs,widths):
            window(ax,x0,y0,w,h,wcolors[ci%4]); ci+=1
    # dashed guides showing the shift
    for k in (sh,sh+M):
        ax.plot([k,k],[0,G],color=RED,lw=1.4,ls='--',alpha=0.8)
        ax.plot([0,G],[k,k],color=RED,lw=1.4,ls='--',alpha=0.8)
    ax.set_title('shifted windows (next block)\nwindows straddle old borders $\\to$ mix',fontsize=12,color=INK)
    for ax in axes:
        ax.set_xlim(-0.2,G+0.2); ax.set_ylim(-0.2,G+0.2); ax.set_aspect('equal'); ax.axis('off')
    save(fig,'windows')

for f in [f_patchify,f_patch_as_conv,f_attn_cost,f_clip_matrix,f_windows]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

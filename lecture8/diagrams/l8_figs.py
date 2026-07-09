"""Metropolis-style figures for Lecture 8 (Convolutional Neural Networks).
Schematic / synthetic ONLY — no real images. Transparent bg, ink + orange/teal.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Run from repo root:  python3 lecture8/diagrams/l8_figs.py"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyArrow, Circle, Polygon
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
OUT='lecture8/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def draw_grid(ax, M, x0=0, y0=0, cmap=CMAP, vmin=None, vmax=None, fs=14,
              fmt='{:g}', textcol=None, ec=INK, lw=1.0, alpha=1.0):
    """Draw a numeric grid of matrix M with top-left at (x0,y0). Row 0 at top."""
    M=np.asarray(M,dtype=float); nr,nc=M.shape
    if vmin is None: vmin=M.min()
    if vmax is None: vmax=M.max()
    norm=Normalize(vmin=vmin, vmax=vmax)
    for i in range(nr):
        for j in range(nc):
            yy=y0-i  # row 0 at top
            ax.add_patch(Rectangle((x0+j, yy-1), 1, 1, facecolor=cmap(norm(M[i,j])),
                                   edgecolor=ec, lw=lw, alpha=alpha))
            tc=textcol if textcol is not None else INK
            ax.text(x0+j+0.5, yy-0.5, fmt.format(M[i,j]), ha='center', va='center',
                    fontsize=fs, color=tc, weight=600)
    return nr,nc

# ── 1 — convolution operation: input window · kernel -> one output cell ──
def f_conv_op():
    inp=np.array([[1,2,0,1,2],
                  [0,1,3,2,0],
                  [2,1,0,1,3],
                  [1,0,2,0,1],
                  [0,2,1,3,2]])
    ker=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])   # horizontal-edge detector
    patch=inp[:3,:3]
    val=int((patch*ker).sum())                    # = 0
    fig,ax=plt.subplots(figsize=(9.2,3.7))
    ax.set_xlim(-0.4,15.6); ax.set_ylim(-1.9,5.6); ax.set_aspect('equal'); ax.axis('off')
    # input grid, highlight the 3x3 window (rows 0-2, cols 0-2)
    draw_grid(ax, inp, x0=0, y0=5, vmin=-1, vmax=3, fs=13)
    ax.add_patch(Rectangle((0,2), 3, 3, fill=False, edgecolor=ACC, lw=3.2, zorder=6))
    ax.text(2.5, 5.55, 'input  $X$', ha='center', fontsize=13, color=INK)
    # kernel
    draw_grid(ax, ker, x0=6, y0=4, vmin=-1, vmax=1, fs=14)
    ax.text(7.5, 4.55, 'kernel  $K$', ha='center', fontsize=13, color=INK)
    ax.text(5.4, 2.5, r'$\ast$', ha='center', va='center', fontsize=26, color=INK)
    # arrow to output cell
    ax.annotate('', xy=(12.9,2.5), xytext=(9.3,2.5),
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.4))
    ax.add_patch(Rectangle((13,2), 1, 1, facecolor=CMAP(0.5), edgecolor=ACC, lw=3.0))
    ax.text(13.5, 2.5, f'{val}', ha='center', va='center', fontsize=17, color=INK, weight=700)
    ax.text(13.5, 3.35, 'output', ha='center', fontsize=12, color=INK)
    # the multiply-add annotation (below everything, clear of the input grid)
    ax.text(8.4, -0.8, r'$(1{+}2{+}0)\cdot 1 \;+\;(0{+}1{+}3)\cdot 0\;+\;(2{+}1{+}0)\cdot(-1)$',
            ha='center', va='center', fontsize=13, color=MUTED)
    ax.text(8.4, -1.6, r'$=\; 3 - 3 \;=\; 0$', ha='center', va='center', fontsize=14.5, color=ACC, weight=700)
    save(fig,'conv_op')

# ── 2 — feature hierarchy: edges -> textures/parts -> objects ──
def f_feature_hierarchy():
    fig,axes=plt.subplots(1,3,figsize=(9.6,3.2))
    titles=['early: edges','middle: textures / parts','late: objects']
    for ax in axes:
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
        ax.add_patch(Rectangle((0.02,0.02),0.96,0.86,fill=False,edgecolor=MUTED,lw=1.2))
    # panel 1: oriented edges (short line segments at varied angles)
    rng=np.random.default_rng(7)
    for _ in range(22):
        cx,cy=rng.uniform(0.12,0.88),rng.uniform(0.12,0.78)
        ang=rng.choice([0,45,90,135])*np.pi/180; L=0.09
        dx,dy=L*np.cos(ang),L*np.sin(ang)
        c=[ACC,TEAL,INK][rng.integers(3)]
        axes[0].plot([cx-dx,cx+dx],[cy-dy,cy+dy],color=c,lw=2.6,solid_capstyle='round')
    # panel 2: textures / parts (blobs, curves, corners)
    for _ in range(7):
        cx,cy=rng.uniform(0.2,0.8),rng.uniform(0.2,0.7)
        r=rng.uniform(0.05,0.1)
        axes[1].add_patch(Circle((cx,cy),r,fill=False,edgecolor=TEAL,lw=2.4))
    for _ in range(5):
        cx,cy=rng.uniform(0.18,0.82),rng.uniform(0.2,0.72)
        t=np.linspace(0,np.pi,30); r=0.09
        axes[1].plot(cx+r*np.cos(t),cy+0.5*r*np.sin(t),color=ACC,lw=2.2)
    # corner marks
    for _ in range(4):
        cx,cy=rng.uniform(0.2,0.8),rng.uniform(0.2,0.7)
        axes[1].plot([cx,cx,cx+0.08],[cy+0.08,cy,cy],color=INK,lw=2.4)
    # panel 3: object (simple house = square + triangle roof + door + window)
    ax=axes[2]
    ax.add_patch(Rectangle((0.3,0.2),0.4,0.34,facecolor='none',edgecolor=INK,lw=2.8))
    ax.add_patch(Polygon([[0.25,0.54],[0.5,0.74],[0.75,0.54]],closed=True,fill=False,edgecolor=ACC,lw=2.8))
    ax.add_patch(Rectangle((0.44,0.2),0.12,0.18,facecolor='none',edgecolor=TEAL,lw=2.4))
    ax.add_patch(Rectangle((0.34,0.4),0.08,0.08,facecolor='none',edgecolor=TEAL,lw=2.2))
    for ax,t in zip(axes,titles):
        ax.text(0.5,-0.03,t,ha='center',va='top',fontsize=12.5,color=INK)
    # arrows between panels
    fig.text(0.365,0.52,'→',fontsize=26,color=MUTED,ha='center',va='center')
    fig.text(0.635,0.52,'→',fontsize=26,color=MUTED,ha='center',va='center')
    save(fig,'feature_hierarchy')

# ── 3 — padding + stride: padded grid, stride-2 window stepping ──
def f_padding_stride():
    fig,ax=plt.subplots(figsize=(7.6,3.6))
    ax.set_xlim(-0.4,7.4); ax.set_ylim(-0.6,7.4); ax.set_aspect('equal'); ax.axis('off')
    n=5; p=1  # 5x5 input, pad 1 -> 7x7
    # padding ring (muted) first
    for i in range(n+2*p):
        for j in range(n+2*p):
            yy=(n+2*p)-i
            is_pad = (i<p or i>=n+p or j<p or j>=n+p)
            fc = MUTED if is_pad else '#EFEEEB'
            al = 0.28 if is_pad else 1.0
            ax.add_patch(Rectangle((j,yy-1),1,1,facecolor=fc,edgecolor=INK,lw=0.8,alpha=al))
    ax.text(3.5,7.15,'padded input  (p = 1)',ha='center',fontsize=13,color=INK)
    ax.text(6.15,6.5,'zero-pad ring',ha='left',fontsize=11,color=MUTED)
    # two stride-2 window positions (3x3), top-left then stepped right+down
    def win(x0,y0,col):
        ax.add_patch(Rectangle((x0,y0),3,3,fill=False,edgecolor=col,lw=3.0,zorder=6))
    win(0,4,ACC)          # position (0,0)
    win(2,2,TEAL)         # stepped by stride 2
    ax.annotate('',xy=(2.4,3.5),xytext=(1.4,4.5),
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.2,connectionstyle='arc3,rad=-0.3'))
    ax.text(1.15,5.35,'stride 2',ha='center',fontsize=11.5,color=ACC)
    save(fig,'padding_stride')

# ── 4 — pooling: 4x4 grid -> 2x2 max-pool and avg-pool ──
def f_pooling():
    M=np.array([[1,3,2,4],
                [5,6,1,2],
                [7,2,3,0],
                [1,0,4,8]])
    # 2x2 non-overlapping regions, stride 2
    mx=np.array([[max(1,3,5,6),max(2,4,1,2)],[max(7,2,1,0),max(3,0,4,8)]])
    av=np.array([[np.mean([1,3,5,6]),np.mean([2,4,1,2])],[np.mean([7,2,1,0]),np.mean([3,0,4,8])]])
    fig,ax=plt.subplots(figsize=(9.2,3.4))
    ax.set_xlim(-0.4,14.4); ax.set_ylim(-0.6,4.8); ax.set_aspect('equal'); ax.axis('off')
    # input 4x4, tint the four 2x2 quadrants faintly
    quadcol=['#FDECD6','#E3EDED','#FDECD6','#E3EDED']
    for qi,(ri,ci) in enumerate([(0,0),(0,2),(2,0),(2,2)]):
        yy=4-ri
        ax.add_patch(Rectangle((ci,yy-2),2,2,facecolor=quadcol[qi],edgecolor='none',zorder=0))
    draw_grid(ax, M, x0=0, y0=4, vmin=0, vmax=8, fs=14, ec=INK, lw=1.0)
    # separators between quadrants
    ax.add_patch(Rectangle((0,0),4,4,fill=False,edgecolor=INK,lw=2.4))
    ax.plot([2,2],[0,4],color=INK,lw=2.4); ax.plot([0,4],[2,2],color=INK,lw=2.4)
    ax.text(2,4.35,r'input $4{\times}4$',ha='center',fontsize=13,color=INK)
    # max pool
    ax.annotate('',xy=(8.4,3.4),xytext=(4.4,3.4),arrowprops=dict(arrowstyle='-|>',color=ACC,lw=2.4))
    ax.text(6.4,3.75,'max pool',ha='center',fontsize=12,color=ACC)
    draw_grid(ax, mx, x0=9, y0=4, vmin=0, vmax=8, fs=15)
    ax.text(10,4.35,r'$2{\times}2$ out',ha='center',fontsize=11.5,color=MUTED)
    # avg pool
    ax.annotate('',xy=(8.4,1.0),xytext=(4.4,1.0),arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=2.4))
    ax.text(6.4,1.35,'avg pool',ha='center',fontsize=12,color=TEAL)
    draw_grid(ax, av, x0=9, y0=1.5, vmin=0, vmax=8, fs=14, fmt='{:.2g}')
    save(fig,'pooling')

# ── 5 — receptive field: nested squares growing 3 -> 5 -> 7 ──
def f_receptive_field():
    fig,ax=plt.subplots(figsize=(6.6,3.4))
    N=7
    ax.set_xlim(-0.4,N+2.6); ax.set_ylim(-0.9,N+0.4); ax.set_aspect('equal'); ax.axis('off')
    # base 7x7 grid faint
    for i in range(N):
        for j in range(N):
            ax.add_patch(Rectangle((j,N-1-i),1,1,facecolor='#EFEEEB',edgecolor=MUTED,lw=0.6,alpha=0.5))
    c=N/2.0  # center = 3.5
    # nested receptive fields centered: 1 (output), 3, 5, 7
    boxes=[(1,INK,'output cell'),(3,GREEN,'after conv 1: 3×3'),
           (5,TEAL,'after conv 2: 5×5'),(7,ACC,'after conv 3: 7×7')]
    for k,col,lbl in boxes:
        x0=c-k/2.0; y0=c-k/2.0
        ax.add_patch(Rectangle((x0,y0),k,k,fill=False,edgecolor=col,lw=3.0))
    # legend
    for idx,(k,col,lbl) in enumerate(boxes):
        yy=N-0.6-idx*0.95
        ax.add_patch(Rectangle((N+0.4,yy),0.5,0.5,fill=False,edgecolor=col,lw=2.6))
        ax.text(N+1.05,yy+0.25,lbl,ha='left',va='center',fontsize=11,color=INK)
    ax.text(c,-0.6,r'stacking 3×3 convs: $r_\ell = r_{\ell-1} + (k-1)$',
            ha='center',fontsize=12,color=INK)
    save(fig,'receptive_field')

# ── 6 — output-size: H_out vs input H for three (k,p,s) configs ──
def f_output_size():
    H=np.arange(4,65)
    def out(H,k,p,s): return (H+2*p-k)//s + 1
    cfgs=[(3,1,1,'k=3, p=1, s=1  (same)',TEAL),
          (3,0,1,'k=3, p=0, s=1  (valid)',BLUE),
          (3,1,2,'k=3, p=1, s=2  (half)',ACC)]
    fig,ax=plt.subplots(figsize=(7.2,3.2))
    for k,p,s,lbl,c in cfgs:
        ax.plot(H,out(H,k,p,s),color=c,lw=2.6,label=lbl)
    ax.plot(H,H,color=MUTED,lw=1.2,ls='--',alpha=0.7,label='H (no change)')
    # mark the two worked points from the deck
    ax.scatter([32],[out(32,5,2,1)],color=RED,s=70,zorder=5)
    ax.annotate('H=32, k=5, p=2, s=1 → 32',(32,32),textcoords='offset points',
                xytext=(-6,10),fontsize=10,color=RED,ha='right')
    ax.scatter([32],[out(32,3,1,2)],color=RED,s=70,zorder=5)
    ax.annotate('k=3, p=1, s=2 → 16',(32,16),textcoords='offset points',
                xytext=(8,-4),fontsize=10,color=RED,ha='left')
    ax.set_xlabel('input size  H'); ax.set_ylabel(r'output size  $H_{\rm out}$')
    ax.legend(frameon=False,fontsize=10,loc='upper left')
    ax.grid(True,alpha=0.15)
    save(fig,'output_size')

for f in [f_conv_op,f_feature_hierarchy,f_padding_stride,f_pooling,
          f_receptive_field,f_output_size]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

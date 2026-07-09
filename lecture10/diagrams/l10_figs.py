"""Metropolis-style figures for Lecture 10 (Semantic and Instance Segmentation).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
SCHEMATIC only — no real images. Fletcher (native typst) draws the
encoder-decoder / U-Net / Mask R-CNN diagrams; matplotlib handles the toy
mask + label-grid figures below.
Run from repo root:  python3 lecture10/diagrams/l10_figs.py"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture10/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def stamp_ellipse(arr, cy, cx, ry, rx, val):
    Y,X = np.ogrid[:arr.shape[0], :arr.shape[1]]
    m = ((Y-cy)/ry)**2 + ((X-cx)/rx)**2 <= 1.0
    arr[m] = val

# ── 1 — task taxonomy: semantic mask (by class) vs instance masks (by object) ──
def f_seg_taxonomy():
    H,W = 44,64
    SKY='#B7D3E0'; GRASS='#A9CBB0'; CREAM='#EFEEEB'
    # semantic: 0 sky, 1 grass, 2 animal-CLASS (both animals share one colour)
    sem = np.zeros((H,W),int)
    sem[int(H*0.55):,:] = 1
    stamp_ellipse(sem, 30, 20, 8, 10, 2)
    stamp_ellipse(sem, 32, 45, 7, 9, 2)
    sem_cmap = ListedColormap([SKY, GRASS, ACC])
    # instance: 0 background (amorphous), 1 animal#1, 2 animal#2 (distinct colours)
    ins = np.zeros((H,W),int)
    stamp_ellipse(ins, 30, 20, 8, 10, 1)
    stamp_ellipse(ins, 32, 45, 7, 9, 2)
    ins_cmap = ListedColormap([CREAM, TEAL, ACC])

    fig,axes = plt.subplots(1,2,figsize=(9.2,3.1))
    axes[0].imshow(sem, cmap=sem_cmap, vmin=0, vmax=2, interpolation='nearest')
    axes[0].set_title('semantic  —  colour = class', fontsize=12.5, color=INK)
    axes[0].text(20,30,'animal',ha='center',va='center',color='white',fontsize=10,weight='bold')
    axes[0].text(45,32,'animal',ha='center',va='center',color='white',fontsize=10,weight='bold')
    axes[0].text(32,7,'both animals share one class colour',ha='center',fontsize=9.5,color=INK)

    axes[1].imshow(ins, cmap=ins_cmap, vmin=0, vmax=2, interpolation='nearest')
    axes[1].set_title('instance  —  colour = object', fontsize=12.5, color=INK)
    # draw boxes around each instance
    for (cx,cy,rx,ry,lbl) in [(20,30,10,8,'#1'),(45,32,9,7,'#2')]:
        axes[1].add_patch(Rectangle((cx-rx-1,cy-ry-1),2*rx+2,2*ry+2,fill=False,
                                    edgecolor=INK,lw=1.4,ls='--'))
        axes[1].text(cx,cy,lbl,ha='center',va='center',color='white',fontsize=10,weight='bold')
    axes[1].text(32,7,'each object = its own mask + id',ha='center',fontsize=9.5,color=INK)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor(MUTED)
    save(fig,'seg_taxonomy')

# ── 2 — Dice worked example: G, P, overlap on a 4x4 grid  ──
def f_dice():
    n = 4
    G = np.zeros((n,n),bool); P = np.zeros((n,n),bool)
    Gcells = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    Pcells = [(1,1),(1,2),(1,3),(2,1),(2,2),(0,1),(0,2),(3,1)]
    for r,c in Gcells: G[r,c]=True
    for r,c in Pcells: P[r,c]=True
    inter = G & P
    sizeP,sizeG,ov = P.sum(), G.sum(), inter.sum()
    dice = 2*ov/(sizeP+sizeG)
    print(f'DICE  |P|={sizeP} |G|={sizeG} |P∩G|={ov}  ->  2*{ov}/({sizeP}+{sizeG}) = {2*ov}/{sizeP+sizeG} = {dice:.4f}')

    # overlap code: 0 bg, 1 only-G (miss/FN), 2 only-P (FP), 3 both (TP)
    code = np.zeros((n,n),int)
    code[G & ~P] = 1; code[P & ~G] = 2; code[inter] = 3
    ov_cmap = ListedColormap(['#EFEEEB', TEAL, ACC, GREEN])

    fig,axes = plt.subplots(1,3,figsize=(9.6,3.0))
    def draw(ax,arr,cmap,vmax,title):
        ax.imshow(arr,cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
        ax.set_xticks(np.arange(-.5,n,1)); ax.set_yticks(np.arange(-.5,n,1))
        ax.grid(color=MUTED,lw=1.0); ax.tick_params(length=0,labelbottom=False,labelleft=False)
        for s in ax.spines.values(): s.set_edgecolor(MUTED)
        ax.set_title(title,fontsize=12,color=INK)
    draw(axes[0],G.astype(int),ListedColormap(['#EFEEEB',TEAL]),1,f'ground truth  $|G|={sizeG}$')
    draw(axes[1],P.astype(int),ListedColormap(['#EFEEEB',ACC]),1,f'prediction  $|P|={sizeP}$')
    draw(axes[2],code,ov_cmap,3,f'overlap  $|P\\cap G|={ov}$')
    axes[2].text(1.5,-0.95,f'Dice = 2·{ov}/({sizeP}+{sizeG}) = {2*ov}/{sizeP+sizeG} ≈ {dice:.3f}',
                 ha='center',fontsize=11.5,color=INK,weight='bold')
    # tiny legend for overlap panel
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(facecolor=GREEN,label='both (TP)'),
                            Patch(facecolor=TEAL,label='miss (FN)'),
                            Patch(facecolor=ACC,label='extra (FP)')],
                   fontsize=8.5,frameon=False,loc='upper left',bbox_to_anchor=(1.02,1.0))
    save(fig,'dice')

# ── 3 — CE vs Dice under class imbalance (5% foreground) ──
def f_ce_vs_dice():
    groups = ['trivial\n(all background)','good\nsegmenter']
    acc  = [0.95, 0.99]     # pixel accuracy — both look "fine"
    dice = [0.00, 0.90]     # Dice — only this exposes the failure
    x = np.arange(len(groups)); w = 0.36
    fig,ax = plt.subplots(figsize=(7.2,3.2))
    b1 = ax.bar(x-w/2, acc,  w, color=MUTED, label='pixel accuracy')
    b2 = ax.bar(x+w/2, dice, w, color=ACC,   label='Dice score')
    for b in (*b1,*b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{b.get_height():.2f}',
                ha='center',fontsize=10.5,color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups,fontsize=11)
    ax.set_ylim(0,1.15); ax.set_yticks([0,0.5,1.0]); ax.set_ylabel('score (higher = better)')
    ax.set_title('5% foreground: accuracy hides the failure, Dice reveals it',fontsize=11.5,pad=10)
    ax.legend(frameon=False,fontsize=10.5,loc='center',ncol=1,bbox_to_anchor=(0.5,0.66))
    ax.annotate('all-background\nscores 0 Dice',xy=(-w/2,0.02),xytext=(-0.05,0.42),
                fontsize=9.5,color=RED,ha='center',
                arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6))
    save(fig,'ce_vs_dice')

# ── 4 — pixel grid: toy H×W label map, every pixel carries a class ──
def f_pixel_grid():
    sky,bld,road,car = 0,1,2,3
    grid = np.array([
        [sky, sky, sky, sky, sky, sky, sky, sky],
        [sky, sky, bld, bld, bld, sky, sky, sky],
        [bld, bld, bld, bld, bld, bld, bld, bld],
        [road,road,road,road,road,road,road,road],
        [road,car, car, road,road,car, car, road],
        [road,road,road,road,road,road,road,road],
    ])
    names = {0:'sky',1:'bldg',2:'road',3:'car'}
    cmap = ListedColormap(['#B7D3E0','#C9B79C','#5B6E72',ACC])
    H,W = grid.shape
    fig,ax = plt.subplots(figsize=(7.2,3.0))
    ax.imshow(grid,cmap=cmap,vmin=0,vmax=3,interpolation='nearest')
    for r in range(H):
        for c in range(W):
            v = grid[r,c]
            tc = 'white' if v in (2,) else INK
            ax.text(c,r,names[v],ha='center',va='center',fontsize=7.5,color=tc)
    ax.set_xticks(np.arange(-.5,W,1)); ax.set_yticks(np.arange(-.5,H,1))
    ax.grid(color='white',lw=1.2); ax.tick_params(length=0,labelbottom=False,labelleft=False)
    for s in ax.spines.values(): s.set_edgecolor(MUTED)
    ax.set_title(r'a toy $6\times8$ label map  —  one class per pixel',fontsize=12)
    save(fig,'pixel_grid')

for f in [f_seg_taxonomy, f_dice, f_ce_vs_dice, f_pixel_grid]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

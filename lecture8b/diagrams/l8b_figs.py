"""Metropolis-style figures for Lecture 8B (Modern CNN Pipelines & Transfer Learning).
Schematic / synthetic ONLY — no real images. Transparent bg, ink + orange/teal.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Run from repo root:  python3 lecture8b/diagrams/l8b_figs.py"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch
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
OUT='lecture8b/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def fbox(ax,x,y,w,h,fc,ec,lw=1.6,rad=0.06,alpha=1.0,zorder=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rad}',
                 facecolor=fc,edgecolor=ec,lw=lw,alpha=alpha,zorder=zorder,
                 mutation_aspect=1.0))

# ── 1 — VGG stage: conv3x3 -> conv3x3 -> pool, channels doubling ──
def f_vgg_stage():
    fig,ax=plt.subplots(figsize=(9.6,3.4))
    ax.set_xlim(0,16); ax.set_ylim(-1.2,4.6); ax.set_aspect('equal'); ax.axis('off')
    # a "stage" = two 3x3 convs then a 2x2 pool; channels shown as stacked slabs
    # draw three stages with channel count doubling: 64 -> 128 -> 256
    stages=[(0.4, 64, 3.0, TEAL),(5.6, 128, 2.3, ACC),(10.8, 256, 1.7, BLUE)]
    for x0,ch,ht,col in stages:
        # feature-map slab (height ~ spatial size shrinks, depth ~ channels grows via width of stack)
        nslab=3
        for s in range(nslab):
            ax.add_patch(Rectangle((x0+0.18*s, 1.5-ht/2+0.16*s), 1.5, ht,
                         facecolor=col if s==nslab-1 else 'white', edgecolor=INK, lw=1.4,
                         alpha=1.0 if s==nslab-1 else 1.0, zorder=3+s))
        ax.text(x0+0.75+0.18, 1.5+ht/2+0.16*nslab+0.15, f'{ch} ch', ha='center',
                fontsize=12.5, color=INK, weight=600)
        # the two 3x3 conv labels + pool label under the slab
        ax.text(x0+0.9, -0.35, r'conv $3{\times}3$', ha='center', fontsize=11, color=col, weight=600)
        ax.text(x0+0.9, -0.85, r'$\times 2$ + pool', ha='center', fontsize=10.5, color=MUTED)
    # arrows between stages
    for xa,xb in [(2.9,5.5),(8.1,10.7)]:
        ax.annotate('',xy=(xb,1.5),xytext=(xa,1.5),
                    arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.2))
    ax.text(4.2,2.35,'pool',ha='center',fontsize=10.5,color=MUTED)
    ax.text(9.4,2.35,'pool',ha='center',fontsize=10.5,color=MUTED)
    ax.text(8.0,4.2,'spatial size halves · channels double',ha='center',fontsize=12.5,color=INK)
    save(fig,'vgg_stage')

# ── 2 — 1x1 conv: channel mixing at one fixed pixel location ──
def f_one_by_one():
    fig,ax=plt.subplots(figsize=(9.4,4.0))
    ax.set_xlim(-0.4,14.4); ax.set_ylim(-1.6,9.0); ax.set_aspect('equal'); ax.axis('off')
    # input feature map (grid) with one pixel highlighted
    for i in range(4):
        for j in range(4):
            hot = (i==1 and j==2)
            ax.add_patch(Rectangle((j,4.4-i),1,1,facecolor=ACC if hot else '#EFEEEB',
                         edgecolor=INK,lw=0.9,alpha=1.0 if hot else 0.9,zorder=2))
    ax.text(2,5.8,r'feature map $H{\times}W{\times}C_{\rm in}$',ha='center',fontsize=12,color=INK)
    ax.text(2.9,0.9,'one spatial location',ha='center',fontsize=11,color=ACC,weight=600)
    # input channel vector (C_in) at that pixel
    cin=6; x_in=6.0
    for k in range(cin):
        ax.add_patch(Rectangle((x_in, 6.9-k*0.7),0.7,0.7,facecolor=TEAL,edgecolor=INK,lw=1.0,
                     alpha=0.35+0.1*k,zorder=3))
    ax.text(x_in+0.35,8.4,r'$x_{hw}\in\mathbb{R}^{C_{\rm in}}$',ha='center',fontsize=12,color=INK)
    # arrow from hot pixel to vector
    ax.annotate('',xy=(x_in-0.15,4.2),xytext=(3.15,3.9),
                arrowprops=dict(arrowstyle='-|>',color=ACC,lw=2.0,connectionstyle='arc3,rad=-0.2'))
    # 1x1 weight box (matrix W)
    ax.add_patch(FancyBboxPatch((8.3,2.9),1.5,2.0,boxstyle='round,pad=0,rounding_size=0.1',
                 facecolor='#FDECD6',edgecolor=ACC,lw=2.0,zorder=3))
    ax.text(9.05,3.9,r'$W$',ha='center',va='center',fontsize=17,color=INK,weight=700)
    ax.text(9.05,2.4,r'$1{\times}1$',ha='center',fontsize=10.5,color=ACC)
    ax.annotate('',xy=(8.25,3.9),xytext=(6.95,3.9),
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.0))
    # output channel vector (C_out)
    cout=3; x_out=11.0
    for k in range(cout):
        ax.add_patch(Rectangle((x_out, 5.0-k*0.7),0.7,0.7,facecolor=BLUE,edgecolor=INK,lw=1.0,
                     alpha=0.5+0.15*k,zorder=3))
    ax.text(x_out+0.35,5.9,r'$y_{hw}\in\mathbb{R}^{C_{\rm out}}$',ha='center',fontsize=12,color=INK)
    ax.annotate('',xy=(x_out-0.05,3.9),xytext=(9.85,3.9),
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.0))
    ax.text(7.0,-1.0,r'$y_{hw}=W\,x_{hw}+b$  — mixes channels, not neighbours',
            ha='center',fontsize=12,color=MUTED)
    save(fig,'one_by_one')

# ── 3 — param bars: 1x1 vs 3x3 params ; standard vs depthwise-sep cost ──
def f_param_bars():
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.4))
    # left: params of a channel-projection 256->64
    ax=axes[0]
    names=[r'$1{\times}1$',r'$3{\times}3$']; vals=[16448,147520]; cols=[TEAL,ACC]
    bars=ax.bar(names,vals,color=cols,width=0.6,edgecolor=INK,lw=1.2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+4000, f'{v:,}',ha='center',fontsize=11.5,color=INK,weight=600)
    ax.set_title(r'params: $256\!\to\!64$ projection',fontsize=12.5,color=INK)
    ax.set_ylim(0,170000); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.text(0.5,-0.28,r'$9\times$ fewer params',transform=ax.transAxes,ha='center',
            fontsize=11,color=TEAL,weight=600)
    # right: FLOP cost of a 3x3, 128->128
    ax=axes[1]
    names=['standard','depthwise\nseparable']; vals=[147456,17536]; cols=[ACC,GREEN]
    bars=ax.bar(names,vals,color=cols,width=0.6,edgecolor=INK,lw=1.2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+4000, f'{v:,}',ha='center',fontsize=11.5,color=INK,weight=600)
    ax.set_title(r'cost: $3{\times}3$, $128\!\to\!128$',fontsize=12.5,color=INK)
    ax.set_ylim(0,170000); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.text(0.5,-0.28,r'$\approx 8.4\times$ cheaper',transform=ax.transAxes,ha='center',
            fontsize=11,color=GREEN,weight=600)
    save(fig,'param_bars')

# ── 4 — depthwise separable: depthwise (per-channel spatial) + pointwise (1x1) ──
def f_depthwise_sep():
    fig,ax=plt.subplots(figsize=(10.0,3.6))
    ax.set_xlim(0,17); ax.set_ylim(-1.0,5.4); ax.set_aspect('equal'); ax.axis('off')
    def slab(x0,y0,col,n=4,w=1.2,h=1.2,dz=0.16):
        for s in range(n):
            ax.add_patch(Rectangle((x0+dz*s,y0+dz*s),w,h,
                         facecolor=col if s==n-1 else 'white',edgecolor=INK,lw=1.2,zorder=3+s))
    # input
    slab(0.3,1.6,'#EFEEEB'); ax.text(1.2,4.0,r'in  $C$ ch',ha='center',fontsize=11.5,color=INK)
    # depthwise: one k x k spatial filter PER channel (no channel mixing)
    ax.annotate('',xy=(4.4,2.3),xytext=(2.9,2.3),arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.0))
    ax.text(3.65,3.1,'depthwise',ha='center',fontsize=11.5,color=TEAL,weight=600)
    ax.text(3.65,-0.5,r'$k{\times}k$ per channel',ha='center',fontsize=10,color=MUTED)
    # depthwise output: same channel count, spatial mixing only
    slab(4.6,1.6,TEAL); ax.text(5.5,4.0,r'$C$ ch',ha='center',fontsize=11.5,color=INK)
    # small per-channel filter icons
    for kk in range(3):
        ax.add_patch(Rectangle((3.35+0.0, 1.95+kk*0.02),0.0,0.0))
    # pointwise: 1x1 mixes channels C -> C_out
    ax.annotate('',xy=(9.0,2.3),xytext=(7.2,2.3),arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.0))
    ax.text(8.1,3.1,'pointwise',ha='center',fontsize=11.5,color=ACC,weight=600)
    ax.text(8.1,-0.5,r'$1{\times}1$ mix channels',ha='center',fontsize=10,color=MUTED)
    slab(9.2,1.6,ACC); ax.text(10.1,4.0,r'out $C_{\rm out}$',ha='center',fontsize=11.5,color=INK)
    # equals annotation
    ax.text(13.4,3.2,'standard conv',ha='center',fontsize=11.5,color=INK,weight=600)
    ax.text(13.4,2.5,'= spatial + channel',ha='center',fontsize=10.5,color=MUTED)
    ax.text(13.4,1.7,'in one step',ha='center',fontsize=10.5,color=MUTED)
    ax.annotate('',xy=(11.6,2.3),xytext=(11.0,2.3),arrowprops=dict(arrowstyle='-',color=MUTED,lw=0))
    ax.text(13.4,0.7,'separable = split it',ha='center',fontsize=10.5,color=GREEN,weight=600)
    save(fig,'depthwise_sep')

# ── 5 — transfer modes: feature-extractor (frozen) vs finetuning ──
def f_transfer_modes():
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8))
    for ax in axes:
        ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect('equal'); ax.axis('off')
    FROZEN='#8FA3A6'; TRAIN=ACC
    def draw_net(ax,backbone_col,backbone_lbl,head_col,head_lbl,title,sub):
        # backbone = three stacked blocks, head = one block on top
        xs=2.2
        for i,yy in enumerate([1.2,3.0,4.8]):
            ax.add_patch(FancyBboxPatch((xs,yy),3.6,1.5,boxstyle='round,pad=0,rounding_size=0.12',
                         facecolor=backbone_col,edgecolor=INK,lw=1.6,zorder=2))
        ax.text(xs+1.8,3.75,'pretrained\nbackbone',ha='center',va='center',fontsize=12,
                color='white',weight=600,zorder=3)
        # head
        ax.add_patch(FancyBboxPatch((xs,7.0),3.6,1.5,boxstyle='round,pad=0,rounding_size=0.12',
                     facecolor=head_col,edgecolor=INK,lw=1.6,zorder=2))
        ax.text(xs+1.8,7.75,'head',ha='center',va='center',fontsize=12.5,color='white',weight=700,zorder=3)
        ax.annotate('',xy=(xs+1.8,7.0),xytext=(xs+1.8,6.3),arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.8))
        ax.text(5,9.5,title,ha='center',fontsize=13.5,color=INK,weight=700)
        # labels for frozen/trainable
        ax.text(xs+4.0,3.75,backbone_lbl,ha='left',va='center',fontsize=10.5,
                color=MUTED if 'frozen' in backbone_lbl else ACC,rotation=90,weight=600)
        ax.text(xs+4.0,7.75,head_lbl,ha='left',va='center',fontsize=10.5,color=ACC,rotation=90,weight=600)
        ax.text(5,0.3,sub,ha='center',fontsize=10.5,color=MUTED)
    draw_net(axes[0],FROZEN,'frozen',TRAIN,'trained',
             'feature extractor','freeze backbone · train head only')
    draw_net(axes[1],ACC,'trained\n(small LR)',TRAIN,'trained',
             'finetuning',r'train all · $\eta_{\rm backbone}<\eta_{\rm head}$')
    # legend
    fig.text(0.5,0.01,'grey = frozen (no gradient update)     orange = trainable',
             ha='center',fontsize=11,color=INK)
    save(fig,'transfer_modes')

for f in [f_vgg_stage,f_one_by_one,f_param_bars,f_depthwise_sep,f_transfer_modes]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

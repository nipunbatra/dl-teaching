"""Metropolis-style SCHEMATIC figures for Lecture 9 (Localization & Object Detection).
No real images — colored rectangles / patches on plain axes, on-palette.
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Run from repo root:  python3 lecture9/diagrams/l9_figs.py"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture9/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def canvas(ax, w=1.0, h=1.0):
    """A plain gray 'image' canvas [0,w]x[0,h]."""
    ax.add_patch(Rectangle((0,0),w,h,facecolor='#E9E7E2',edgecolor=MUTED,lw=1.0,zorder=0))
    ax.set_xlim(-0.03*w,1.03*w); ax.set_ylim(-0.03*h,1.06*h)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def blob(ax, cx, cy, rw, rh, color, alpha=0.85):
    """A soft schematic 'object' (rounded rectangle-ish ellipse)."""
    ax.add_patch(mpatches.Ellipse((cx,cy),rw,rh,facecolor=color,edgecolor='none',alpha=alpha,zorder=1))

# ── 1 — task taxonomy: label / one box / many boxes across 3 mini panels ──
def f_task_taxonomy():
    fig,axes=plt.subplots(1,3,figsize=(9.6,3.1))
    titles=['classification','localization','detection']
    subs=[r'$x\to y$', r'$x\to(y,b)$', r'$x\to\{(y_i,b_i)\}$']
    for k,(ax,ttl,sub) in enumerate(zip(axes,titles,subs)):
        canvas(ax)
        # same schematic scene: a big 'cat' blob and two small 'bird' blobs
        blob(ax,0.42,0.42,0.44,0.40,TEAL)
        blob(ax,0.78,0.74,0.20,0.16,ACC)
        blob(ax,0.18,0.80,0.16,0.13,BLUE)
        if k==0:  # one label only (whole-image class, drawn on the canvas)
            ax.text(0.42,0.42,'cat',ha='center',va='center',fontsize=17,color='white',weight='bold',
                    zorder=4)
        if k==1:  # one dominant box
            ax.add_patch(Rectangle((0.20,0.22),0.44,0.40,fill=False,edgecolor=INK,lw=2.4,zorder=3))
            ax.text(0.20,0.66,'cat',ha='left',va='bottom',fontsize=11,color=INK,weight='bold')
        if k==2:  # many boxes
            for (x0,y0,w,h,c,lab) in [(0.20,0.22,0.44,0.40,TEAL,'cat'),
                                      (0.68,0.66,0.20,0.16,ACC,'bird'),
                                      (0.10,0.73,0.16,0.14,BLUE,'bird')]:
                ax.add_patch(Rectangle((x0,y0),w,h,fill=False,edgecolor=c,lw=2.2,zorder=3))
                ax.text(x0,y0+h,lab,ha='left',va='bottom',fontsize=9,color=c,weight='bold')
        ax.set_title(f'{ttl}\n{sub}',fontsize=12)
    save(fig,'task_taxonomy')

# ── 2 — box parameterizations: corners AND center/size ──
def f_box_params():
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.3))
    for ax in axes: canvas(ax)
    x0,y0,x1,y1=0.28,0.24,0.74,0.68
    xc,yc=(x0+x1)/2,(y0+y1)/2; w,h=x1-x0,y1-y0
    # left: corners
    ax=axes[0]
    ax.add_patch(Rectangle((x0,y0),w,h,fill=False,edgecolor=ACC,lw=2.6,zorder=3))
    ax.scatter([x0,x1],[y0,y1],color=INK,s=42,zorder=4)
    ax.text(x0-0.02,y0-0.02,r'$(x_{\min},y_{\min})$',ha='right',va='top',fontsize=11.5,color=INK)
    ax.text(x1+0.02,y1+0.02,r'$(x_{\max},y_{\max})$',ha='left',va='bottom',fontsize=11.5,color=INK)
    ax.set_title('corner form',fontsize=12.5)
    # right: center + width/height
    ax=axes[1]
    ax.add_patch(Rectangle((x0,y0),w,h,fill=False,edgecolor=TEAL,lw=2.6,zorder=3))
    ax.scatter([xc],[yc],color=RED,s=52,zorder=4,marker='+',linewidths=2.6)
    ax.text(xc+0.015,yc+0.02,r'$(x_c,y_c)$',ha='left',va='bottom',fontsize=11.5,color=RED)
    ax.annotate('',xy=(x1,y0-0.07),xytext=(x0,y0-0.07),arrowprops=dict(arrowstyle='<|-|>',color=INK,lw=1.8))
    ax.text(xc,y0-0.12,r'$w$',ha='center',va='top',fontsize=12,color=INK)
    ax.annotate('',xy=(x0-0.07,y1),xytext=(x0-0.07,y0),arrowprops=dict(arrowstyle='<|-|>',color=INK,lw=1.8))
    ax.text(x0-0.10,yc,r'$h$',ha='right',va='center',fontsize=12,color=INK)
    ax.set_title('center + size form',fontsize=12.5)
    save(fig,'box_params')

# ── 3 — smooth-L1 vs |r| (L1) vs r^2 (L2) ──
def f_smooth_l1():
    r=np.linspace(-3,3,400)
    l1=np.abs(r); l2=r**2
    sl1=np.where(np.abs(r)<1, 0.5*r**2, np.abs(r)-0.5)
    fig,ax=plt.subplots(figsize=(6.6,3.4))
    ax.plot(r,l2,color=RED,lw=2.4,label=r'$L_2:\ r^2$')
    ax.plot(r,l1,color=BLUE,lw=2.4,label=r'$L_1:\ |r|$')
    ax.plot(r,sl1,color=ACC,lw=3.0,label='smooth $L_1$')
    ax.axvspan(-1,1,color=MUTED,alpha=.10)
    ax.text(0,0.62,r'$|r|<1$',ha='center',va='bottom',fontsize=10.5,color=MUTED)
    ax.set_xlabel(r'residual $r=\hat b-b$'); ax.set_ylabel('loss')
    ax.set_ylim(-0.2,4.2); ax.set_xlim(-3,3)
    ax.legend(frameon=False,fontsize=11.5,loc='upper center')
    save(fig,'smooth_l1')

# ── 4 — IoU: two overlapping boxes, intersection shaded + numbers ──
def f_iou():
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    ax.set_xlim(0,12); ax.set_ylim(0,10); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    # A: 10x10 area=100 -> use 5 wide x ... keep area labels schematic
    A=Rectangle((1.5,2.0),5.0,5.0,fill=False,edgecolor=TEAL,lw=2.8,zorder=3)
    B=Rectangle((4.0,4.5),5.0,5.0,fill=False,edgecolor=ACC,lw=2.8,zorder=3)
    ax.add_patch(A); ax.add_patch(B)
    # intersection [4.0,6.5]x[4.5,7.0]
    ix0,iy0,ix1,iy1=4.0,4.5,6.5,7.0
    ax.add_patch(Rectangle((ix0,iy0),ix1-ix0,iy1-iy0,facecolor=RED,alpha=.30,edgecolor=RED,lw=1.4,zorder=2))
    ax.text(1.6,7.1,r'$A$',color=TEAL,fontsize=15,weight='bold')
    ax.text(8.9,9.6,r'$B$',color=ACC,fontsize=15,weight='bold',ha='right')
    ax.text((ix0+ix1)/2,(iy0+iy1)/2,r'$A\cap B$',color=RED,fontsize=11,ha='center',va='center',weight='bold')
    ax.text(6.0,0.7,r'$|A|=100,\ \ |B|=100,\ \ |A\cap B|=25$',ha='center',fontsize=12.5,color=INK)
    ax.text(6.0,-0.4,r'$|A\cup B|=175,\quad \mathrm{IoU}=25/175\approx0.143$',
            ha='center',fontsize=13,color=INK,weight='bold')
    ax.set_ylim(-1.2,10)
    save(fig,'iou')

# ── 5 — precision-recall curve with AP shaded ──
def f_pr_curve():
    rec=np.linspace(0,1,200)
    # a plausible descending PR curve
    prec=np.clip(1.0-0.55*rec**2 - 0.15*rec, 0, 1)
    prec[rec>0.92]=np.linspace(prec[rec>0.92][0],0.15,np.sum(rec>0.92))
    fig,ax=plt.subplots(figsize=(6.2,3.4))
    ax.fill_between(rec,0,prec,color=ACC,alpha=.18,zorder=1)
    ax.plot(rec,prec,color=ACC,lw=2.8,zorder=3)
    ax.text(0.42,0.35,'AP = area\nunder curve',ha='center',fontsize=12,color=INK)
    ax.set_xlabel('recall'); ax.set_ylabel('precision')
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xticks([0,0.5,1.0]); ax.set_yticks([0,0.5,1.0])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig,'pr_curve')

# ── 6 — anchors: feature grid with reference anchor boxes at one cell ──
def f_anchors():
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    canvas(ax)
    S=5
    for i in range(S+1):
        ax.plot([i/S,i/S],[0,1],color=MUTED,lw=0.7,alpha=.6,zorder=1)
        ax.plot([0,1],[i/S,i/S],color=MUTED,lw=0.7,alpha=.6,zorder=1)
    # one cell highlighted
    cx,cy=2.5/S,2.5/S
    ax.add_patch(Rectangle((2/S,2/S),1/S,1/S,facecolor=ACC,alpha=.16,zorder=1))
    ax.scatter([cx],[cy],color=INK,s=34,zorder=5)
    # a few anchors of different aspect ratios / scales centered on that cell
    anchors=[(0.30,0.30,TEAL),(0.44,0.22,ACC),(0.22,0.44,BLUE)]
    for (w,h,c) in anchors:
        ax.add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fill=False,edgecolor=c,lw=2.2,zorder=4))
    ax.set_ylim(-0.03,1.05)
    save(fig,'anchors')

# ── 7 — focal loss: FL(pt) for gamma in {0,1,2,5} ──
def f_focal_loss():
    pt=np.linspace(0.01,1.0,300)
    fig,ax=plt.subplots(figsize=(6.6,3.5))
    cols={0:INK,1:BLUE,2:ACC,5:RED}
    for g in [0,1,2,5]:
        fl=-(1-pt)**g*np.log(pt)
        lbl=r'$\gamma=0$ (CE)' if g==0 else rf'$\gamma={g}$'
        ax.plot(pt,fl,color=cols[g],lw=2.6,label=lbl)
    ax.axvspan(0.6,1.0,color=GREEN,alpha=.08)
    ax.text(0.8,1.9,'easy examples\n(down-weighted)',ha='center',fontsize=10,color=MUTED)
    ax.set_xlabel(r'$p_t$  (prob. of true class)'); ax.set_ylabel(r'loss  $FL(p_t)$')
    ax.set_xlim(0,1); ax.set_ylim(0,5)
    ax.legend(frameon=False,fontsize=11,loc='upper right')
    save(fig,'focal_loss')

# ── 8 — YOLO grid: S x S image grid with a couple predicted boxes ──
def f_yolo_grid():
    fig,ax=plt.subplots(figsize=(6.0,3.6))
    canvas(ax)
    S=6
    for i in range(S+1):
        ax.plot([i/S,i/S],[0,1],color=MUTED,lw=0.7,alpha=.55,zorder=1)
        ax.plot([0,1],[i/S,i/S],color=MUTED,lw=0.7,alpha=.55,zorder=1)
    # two objects, each 'assigned' to the cell containing its center
    blob(ax,0.40,0.44,0.40,0.36,TEAL,alpha=.55)
    blob(ax,0.76,0.74,0.22,0.18,ACC,alpha=.55)
    ax.add_patch(Rectangle((0.20,0.26),0.40,0.36,fill=False,edgecolor=TEAL,lw=2.4,zorder=4))
    ax.add_patch(Rectangle((0.65,0.65),0.22,0.18,fill=False,edgecolor=ACC,lw=2.4,zorder=4))
    # mark responsible cells (centers)
    for (cx,cy,c) in [(0.40,0.44,TEAL),(0.76,0.74,ACC)]:
        gi,gj=int(cx*S),int(cy*S)
        ax.add_patch(Rectangle((gi/S,gj/S),1/S,1/S,facecolor=c,alpha=.18,zorder=1))
        ax.scatter([cx],[cy],color=c,s=30,zorder=5)
    ax.text(0.5,1.10,r'$S\times S$ grid: each cell predicts boxes + classes',
            ha='center',fontsize=11,color=INK)
    ax.set_ylim(-0.03,1.18)
    save(fig,'yolo_grid')

# ── 9 — NMS: before (many overlapping) vs after (one) ──
def f_nms():
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.4))
    for ax in axes: canvas(ax)
    base=(0.30,0.28,0.42,0.44)  # x0,y0,w,h
    rng=np.random.default_rng(2)
    # before: many jittered boxes with scores
    ax=axes[0]
    blob(ax,0.51,0.50,0.44,0.44,TEAL,alpha=.35)
    for k in range(6):
        dx,dy=rng.normal(0,0.035,2); dw,dh=rng.normal(0,0.03,2)
        ax.add_patch(Rectangle((base[0]+dx,base[1]+dy),base[2]+dw,base[3]+dh,
                     fill=False,edgecolor=ACC,lw=1.6,alpha=.8,zorder=3))
    ax.set_title('before NMS: many boxes / object',fontsize=12)
    # after: single best box
    ax=axes[1]
    blob(ax,0.51,0.50,0.44,0.44,TEAL,alpha=.35)
    ax.add_patch(Rectangle(base[:2],base[2],base[3],fill=False,edgecolor=GREEN,lw=2.8,zorder=3))
    ax.text(base[0],base[1]+base[3],'score 0.95',ha='left',va='bottom',fontsize=10,color=GREEN,weight='bold')
    ax.set_title('after NMS: keep the best',fontsize=12)
    save(fig,'nms')

# ── 10 — anchor offsets: anchor box --(t_x,t_y,t_w,t_h)--> predicted box ──
def f_anchor_offset():
    fig,ax=plt.subplots(figsize=(6.8,3.7))
    ax.set_xlim(0,10); ax.set_ylim(0,8); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    # anchor box (dashed, teal) — the fixed reference
    axc,ayc,aw,ah = 3.5,3.9,2.8,2.2
    ax.add_patch(Rectangle((axc-aw/2,ayc-ah/2),aw,ah,fill=False,edgecolor=TEAL,lw=2.4,ls=(0,(5,3)),zorder=2))
    ax.scatter([axc],[ayc],color=TEAL,s=42,zorder=5)
    ax.text(axc-aw/2,ayc-ah/2-0.2,'anchor $a$',ha='left',va='top',fontsize=12,color=TEAL,weight='bold')
    # predicted box (solid, orange) — shifted + resized
    pxc,pyc,pw,ph = 6.1,5.0,3.7,2.9
    ax.add_patch(Rectangle((pxc-pw/2,pyc-ph/2),pw,ph,fill=False,edgecolor=ACC,lw=2.8,zorder=3))
    ax.scatter([pxc],[pyc],color=ACC,s=46,zorder=5)
    ax.text(pxc+pw/2,pyc+ph/2+0.15,'predicted box',ha='right',va='bottom',fontsize=12,color=ACC,weight='bold')
    # center-shift arrow (t_x, t_y)
    ax.annotate('',xy=(pxc,pyc),xytext=(axc,ayc),
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.2,shrinkA=4,shrinkB=4))
    ax.text((axc+pxc)/2-0.15,(ayc+pyc)/2+0.15,r'$(t_x,t_y)$',ha='right',va='bottom',fontsize=12,color=INK)
    # width brackets: w_a (anchor) and w (predicted), stacked below
    ya1,ya2 = 1.9,1.15
    ax.annotate('',xy=(axc+aw/2,ya1),xytext=(axc-aw/2,ya1),arrowprops=dict(arrowstyle='<|-|>',color=TEAL,lw=1.6))
    ax.text(axc,ya1-0.12,r'$w_a$',ha='center',va='top',fontsize=11.5,color=TEAL)
    ax.annotate('',xy=(pxc+pw/2,ya2),xytext=(pxc-pw/2,ya2),arrowprops=dict(arrowstyle='<|-|>',color=ACC,lw=1.6))
    ax.text(pxc,ya2-0.12,r'$w$',ha='center',va='top',fontsize=11.5,color=ACC)
    ax.text(9.7,3.4,r'$t_w=\log\dfrac{w}{w_a}$'+'\n'+r'$t_h=\log\dfrac{h}{h_a}$',
            ha='right',va='center',fontsize=11.5,color=INK)
    ax.set_ylim(0.3,8)
    save(fig,'anchor_offset')

for f in [f_task_taxonomy,f_box_params,f_smooth_l1,f_iou,f_pr_curve,
          f_anchors,f_focal_loss,f_yolo_grid,f_nms,f_anchor_offset]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

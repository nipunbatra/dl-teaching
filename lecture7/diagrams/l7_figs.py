"""Metropolis-style figures for Lecture 7 (Generalization and Regularization).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Run from repo root:  python3 lecture7/diagrams/l7_figs.py"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture7/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── 1 — train loss (down) + val loss (U), early-stop vertical line ──
def f_train_val():
    t=np.linspace(0,1,200)
    train=1.15*np.exp(-3.2*t)+0.12
    val=0.9*np.exp(-3.6*t)+0.30+0.55*t**2
    star=t[np.argmin(val)]
    fig,ax=plt.subplots(figsize=(6.6,3.0))
    ax.plot(t,train,color=TEAL,lw=2.6,label='training loss')
    ax.plot(t,val,color=ACC,lw=2.6,label='validation loss')
    ax.axvline(star,color=RED,lw=1.8,ls='--')
    ax.scatter([star],[val.min()],color=RED,s=55,zorder=6)
    ax.annotate('early stop  $t^\\star$',xy=(star,val.min()),xytext=(star+0.06,val.min()+0.42),
                color=RED,fontsize=12,arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6))
    ax.text(0.72,0.30,'overfitting\n(val rises)',color=MUTED,fontsize=11,ha='center')
    ax.set_xlabel('training epochs'); ax.set_ylabel('loss')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0,1.5)
    ax.legend(frameon=False,fontsize=11,loc='upper right')
    save(fig,'train_val')

# ── 2 — bias / variance: sin(2*pi*x)+noise fit with degree 1 / 5 / 20 ──
def f_bias_variance_poly():
    rng=np.random.default_rng(1)
    x=np.sort(rng.uniform(0.02,0.98,16)); y=np.sin(2*np.pi*x)+rng.normal(0,0.18,x.size)
    xx=np.linspace(0,1,400); truth=np.sin(2*np.pi*xx)
    fig,axes=plt.subplots(1,3,figsize=(9.6,2.9))
    for ax,(d,ttl,c) in zip(axes,[(1,'degree 1 — underfit',BLUE),
                                   (5,'degree 5 — good',GREEN),
                                   (15,'degree 15 — overfit',RED)]):
        co=np.polyfit(x,y,d); fit=np.polyval(co,xx)
        fit=np.where(np.abs(fit)>1.85,np.nan,fit)  # hide off-plot spikes
        ax.plot(xx,truth,color=MUTED,lw=1.6,ls='--',alpha=.8)
        ax.scatter(x,y,color=INK,s=22,zorder=4)
        ax.plot(xx,fit,color=c,lw=2.6)
        ax.set_ylim(-1.9,1.9); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl,fontsize=12,color=c)
    save(fig,'bias_variance_poly')

# ── 3 — weight-decay: weight-norm vs step for several lambda ──
def f_weight_decay():
    steps=np.arange(0,60)
    fig,ax=plt.subplots(figsize=(6.4,3.0))
    for lam,c in [(0.0,INK),(0.01,BLUE),(0.05,TEAL),(0.15,ACC)]:
        # norm grows toward an unregularized plateau, damped by (1-eta*lam)
        norm=3.0*(1-np.exp(-0.10*steps))*(1/(1+8*lam))+0.15
        ax.plot(steps,norm,color=c,lw=2.5,label=f'$\\lambda$={lam}')
    ax.set_xlabel('training step'); ax.set_ylabel(r'weight norm  $\|\theta\|$')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False,fontsize=11,loc='center right',title='')
    ax.set_title('larger $\\lambda$  →  smaller weights',fontsize=12)
    save(fig,'weight_decay')

# ── 4 — dropout mask: hidden layer bars, ~half zeroed (dropped) ──
def f_dropout_mask():
    rng=np.random.default_rng(4)
    n=10; h=rng.uniform(0.4,1.0,n)
    mask=np.array([1,0,1,1,0,1,0,1,0,1])
    fig,ax=plt.subplots(figsize=(6.8,2.8))
    for i in range(n):
        kept=mask[i]==1
        ax.bar(i,h[i] if kept else 0.06,width=0.7,
               color=TEAL if kept else MUTED,alpha=1.0 if kept else .45)
        if not kept:
            ax.text(i,0.10,'✕',ha='center',va='bottom',color=MUTED,fontsize=14)
    ax.set_ylim(0,1.15); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('hidden units'); ax.set_ylabel('activation')
    ax.set_title('dropout: kept (teal)  ·  dropped (grey ✕)',fontsize=12)
    save(fig,'dropout_mask')

# ── 5 — mixup: two 2-class clusters + a mixed point on the interp line ──
def f_mixup():
    rng=np.random.default_rng(5)
    A=rng.normal([-1.1,-0.4],0.35,(28,2)); B=rng.normal([1.1,0.5],0.35,(28,2))
    a=np.array([-1.0,-0.5]); b=np.array([1.2,0.6]); lam=0.7
    mixed=lam*a+(1-lam)*b
    fig,ax=plt.subplots(figsize=(6.0,3.1))
    ax.scatter(A[:,0],A[:,1],color=TEAL,s=26,label='class A',alpha=.85)
    ax.scatter(B[:,0],B[:,1],color=ACC,s=26,label='class B',alpha=.85)
    ax.plot([a[0],b[0]],[a[1],b[1]],color=MUTED,lw=1.5,ls='--')
    ax.scatter([a[0]],[a[1]],color=TEAL,edgecolor=INK,s=90,zorder=5)
    ax.scatter([b[0]],[b[1]],color=ACC,edgecolor=INK,s=90,zorder=5)
    ax.scatter([mixed[0]],[mixed[1]],color=RED,edgecolor=INK,s=110,zorder=6,marker='D')
    ax.annotate(r'$\tilde{x}=0.7x_A+0.3x_B$',xy=(mixed[0],mixed[1]),
                xytext=(mixed[0]-0.4,mixed[1]-1.05),color=RED,fontsize=11,
                arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False,fontsize=11,loc='upper left')
    save(fig,'mixup')

# ── 6 — label smoothing: hard one-hot vs smoothed target (K=5) ──
def f_label_smoothing():
    K=5; eps=0.1; y=2
    hard=np.zeros(K); hard[y]=1.0
    soft=np.full(K,eps/(K-1)); soft[y]=1-eps
    fig,axes=plt.subplots(1,2,figsize=(8.2,2.8))
    xs=np.arange(K)
    for ax,(vals,ttl,c) in zip(axes,[(hard,'hard one-hot',INK),
                                      ('','',''),]):
        pass
    ax=axes[0]; ax.bar(xs,hard,color=INK,width=0.6)
    ax.set_title('hard one-hot',fontsize=12); ax.set_ylim(0,1.08)
    ax=axes[1]; bars=ax.bar(xs,soft,color=[TEAL if i==y else ACC for i in xs],width=0.6)
    ax.set_title(r'smoothed  ($\epsilon=0.1$)',fontsize=12); ax.set_ylim(0,1.08)
    for a in axes:
        a.set_xticks(xs); a.set_xticklabels([f'{i}' for i in xs])
        a.set_yticks([]); a.set_xlabel('class')
    axes[1].text(y,0.92,'0.9',ha='center',fontsize=10,color=TEAL)
    for i in xs:
        if i!=y: axes[1].text(i,soft[i]+0.03,'0.025',ha='center',fontsize=8,color=ACC)
    save(fig,'label_smoothing')

# ── 7 — decision boundary: wiggly (overfit) vs smooth (regularized) ──
def f_decision_boundary_reg():
    rng=np.random.default_rng(7)
    n=40
    A=rng.normal([-0.8,-0.2],0.55,(n,2)); B=rng.normal([0.8,0.4],0.55,(n,2))
    # flip a few labels to create noise the wiggly boundary chases
    X=np.vstack([A,B]); yb=np.r_[np.zeros(n),np.ones(n)]
    gx=np.linspace(-2.6,2.6,220); gy=np.linspace(-2.2,2.2,220); GX,GY=np.meshgrid(gx,gy)
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.1))
    # smooth: straight-ish linear boundary
    def draw(ax,wig,ttl):
        ax.scatter(A[:,0],A[:,1],color=TEAL,s=24,alpha=.85)
        ax.scatter(B[:,0],B[:,1],color=ACC,s=24,alpha=.85)
        if wig:
            bx=0.35*np.sin(3.0*gy)+0.30*np.sin(6*gy)  # wiggly curve x = f(y)
        else:
            bx=0.05*gy
        ax.plot(bx,gy,color=INK,lw=2.4)
        ax.set_xlim(-2.6,2.6); ax.set_ylim(-2.2,2.2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=12)
    draw(axes[0],True,'overfit — wiggly boundary')
    draw(axes[1],False,'regularized — smooth boundary')
    save(fig,'decision_boundary_reg')

# ── 8 — double descent: test error vs model size (classical U + 2nd descent) ──
def f_double_descent():
    c=np.linspace(0.05,3.0,300)
    interp=1.0  # interpolation threshold
    # classical U on the underparam side, spike at threshold, 2nd descent after
    train=np.clip(0.9*np.exp(-2.2*c),0,1)
    bias=0.35*np.exp(-2.5*c)
    var=0.28/ (np.abs(c-interp)+0.14)*np.where(c<interp,1.0,0.55)
    test=bias+var*0.35+0.12
    test=np.clip(test,0,1.4)
    fig,ax=plt.subplots(figsize=(6.8,3.0))
    ax.plot(c,test,color=ACC,lw=2.8,label='test error')
    ax.plot(c,train,color=TEAL,lw=2.2,label='train error',alpha=.9)
    ax.axvline(interp,color=MUTED,lw=1.4,ls='--')
    ax.text(interp,1.28,'interpolation\nthreshold',ha='center',fontsize=10,color=MUTED)
    ax.text(0.45,0.55,'classical\nU-curve',ha='center',fontsize=10,color=INK)
    ax.text(2.35,0.34,'modern\n2nd descent',ha='center',fontsize=10,color=INK)
    ax.set_xlabel('model size / capacity'); ax.set_ylabel('error')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0,1.45)
    ax.legend(frameon=False,fontsize=11,loc='upper right')
    save(fig,'double_descent')

# ── 9 — augmentation schematic: a shape under label-preserving transforms ──
def f_augmentation():
    # an "L" point-cloud shape, transformed 4 ways (identity/flip-h/rotate/jitter)
    base=np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[2,0]],float)  # an L
    base=base-base.mean(0)
    def rot(P,th):
        R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); return P@R.T
    rng=np.random.default_rng(9)
    variants=[('original',base,TEAL),
              ('flip',base*[-1,1],ACC),
              ('rotate',rot(base,0.5),BLUE),
              ('jitter',base+rng.normal(0,0.18,base.shape),GREEN)]
    fig,axes=plt.subplots(1,4,figsize=(9.6,2.6))
    for ax,(ttl,P,c) in zip(axes,variants):
        ax.scatter(P[:,0],P[:,1],color=c,s=60)
        ax.plot(P[:,0],P[:,1],color=c,lw=1.0,alpha=.4)
        ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=12,color=c)
    fig.suptitle('same label under label-preserving transforms',fontsize=12,y=1.04,color=INK)
    save(fig,'augmentation')

for f in [f_train_val,f_bias_variance_poly,f_weight_decay,f_dropout_mask,f_mixup,
          f_label_smoothing,f_decision_boundary_reg,f_double_descent,f_augmentation]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

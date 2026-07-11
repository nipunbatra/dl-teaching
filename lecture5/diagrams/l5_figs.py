"""Metropolis-style figures for Lecture 5 (Optimization for Deep Learning).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Run from repo root:  python3 lecture5/diagrams/l5_figs.py"""
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
OUT='lecture5/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)
def bare3d(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.set_pane_color((1,1,1,0)); a.line.set_color(MUTED)
    ax.grid(False)

# ── 1 — loss landscape: 3D surface + contour with gradient arrow ──
def f_landscape():
    fig=plt.figure(figsize=(7.4,3.0))
    X,Y=np.meshgrid(np.linspace(-2,2,60),np.linspace(-2,2,60)); Z=X**2+2.2*Y**2
    ax=fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(X,Y,Z,cmap=CMAP,alpha=.92,linewidth=0,antialiased=True,rstride=2,cstride=2)
    ax.set_title(r'loss surface  $\mathcal{L}(\theta_1,\theta_2)$',fontsize=12)
    bare3d(ax); ax.view_init(30,-52)
    ax2=fig.add_subplot(1,2,2)
    g=np.linspace(-2,2,220); Xc,Yc=np.meshgrid(g,g); Zc=Xc**2+2.2*Yc**2
    ax2.contour(Xc,Yc,Zc,levels=10,colors=[TEAL],linewidths=.8,alpha=.8)
    p=np.array([-1.5,1.3]); grad=np.array([2*p[0],4.4*p[1]])
    ax2.scatter([p[0]],[p[1]],color=INK,s=40,zorder=4)
    ax2.annotate('',xy=(p[0]-0.28*grad[0],p[1]-0.28*grad[1]),xytext=(p[0],p[1]),
                 arrowprops=dict(arrowstyle='-|>',color=ACC,lw=2.6))
    ax2.text(p[0]-0.9,p[1]-0.9,r'$-\nabla L$',color=ACC,fontsize=13)
    ax2.scatter([0],[0],marker='*',s=170,color=RED,zorder=5)
    ax2.set_aspect('equal'); ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title('contours + gradient step',fontsize=12)
    save(fig,'landscape')

# ── 2 — lr_trajectories — RETIRED.
# Now native ml-plot: four panels, each the parabola L=(θ-3)² plus the ACTUAL GD
# iterates at that η (markers-per-series), computed in-deck — so the diverging
# case genuinely climbs out. See lecture5/L5-optimization.typ ("learning rate matters").

# ── 3 & 6 — batch_noise / optimizer_compare — RETIRED.
# Both are now native ml-field contours with real ml-optim trajectories computed
# in-deck: batch_noise = three seeded SGD runs (noise 0/2.2/6) on x²+3y²;
# optimizer_compare = gd / sgd / momentum / adam on x²+50y². The seeded PRNG
# (ml-random) makes the stochastic jitter reproducible.
# See lecture5/L5-optimization.typ ("Batch size shapes the noise" / "four optimizers").

# ── 7 — learning-rate schedules ──
def f_lr_schedules():
    T=100; t=np.arange(T); e0=1.0
    step=e0*(0.5**(t//30))
    expo=e0*np.exp(-0.03*t)
    cosine=0.5*e0*(1+np.cos(np.pi*t/T))
    wu=10
    wc=np.where(t<wu, e0*t/wu, 0.5*e0*(1+np.cos(np.pi*(t-wu)/(T-wu))))
    # one-cycle: rise then fall (triangular-ish) with lower tail
    peak=25
    oc=np.where(t<peak, 0.25*e0+0.75*e0*t/peak,
                np.where(t<T-10, e0-(e0-0.15*e0)*(t-peak)/(T-10-peak), 0.15*e0*(1-(t-(T-10))/10)))
    oc=np.clip(oc,0,1.05)
    fig,ax=plt.subplots(figsize=(7.2,3.0))
    for y,lbl,c in [(step,'step',TEAL),(expo,'exponential',BLUE),(cosine,'cosine',GREEN),
                    (wc,'warmup + cosine',ACC),(oc,'one-cycle',RED)]:
        ax.plot(t,y,color=c,lw=2.2,label=lbl)
    ax.set_xlabel('training step'); ax.set_ylabel(r'learning rate $\eta_t$')
    ax.set_yticks([]); ax.legend(frameon=False,fontsize=10.5,loc='upper right',ncol=2)
    save(fig,'lr_schedules')

# ── 8b — momentum as a vector sum: beta*v_{t-1} + g_t = v_t ──
def f_momentum_vector():
    O=np.array([0.0,0.0])
    bv=np.array([2.4,0.0])     # beta * v_{t-1}: accumulated velocity, along the valley
    g =np.array([0.6,-1.0])    # this step's gradient: a cross-valley kick
    vt=bv+g                    # new velocity = the actual step direction
    fig,ax=plt.subplots(figsize=(6.0,3.2))
    def arr(a,b,c,lw=2.8):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=c,lw=lw))
    # parallelogram guide (faint): shows v_t as the diagonal
    ax.plot([bv[0],vt[0]],[bv[1],vt[1]],color=MUTED,lw=1.0,ls='--',alpha=.7)
    ax.plot([g[0],vt[0]],[g[1],vt[1]],color=MUTED,lw=1.0,ls='--',alpha=.7)
    arr(O,bv,TEAL); ax.text(1.0,0.14,r'$\beta\,v_{t-1}$',color=TEAL,fontsize=16,ha='center')
    arr(O,g,ACC);   ax.text(0.12,-0.62,r'$g_t$',color=ACC,fontsize=16,ha='right')
    arr(O,vt,INK,lw=3.4); ax.text(2.25,-0.46,r'$v_t$',color=INK,fontsize=16,ha='center')
    ax.scatter([O[0]],[O[1]],color=INK,s=34,zorder=5)
    ax.set_xlim(-0.8,3.4); ax.set_ylim(-1.4,0.55); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ['left','bottom']: ax.spines[s].set_visible(False)
    ax.set_title(r'$v_t=\beta\,v_{t-1}+g_t$  — momentum adds this step to the running velocity',fontsize=12)
    save(fig,'momentum_vector')

# ── 8 — sharp vs flat minimum (1D) ──
def f_sharp_flat():
    x=np.linspace(-2.5,2.5,300)
    sharp=6.0*x**2
    flat=0.7*x**2
    fig,ax=plt.subplots(figsize=(6.2,2.9))
    ax.plot(x,sharp,color=ACC,lw=2.6,label='sharp minimum')
    ax.plot(x,flat,color=TEAL,lw=2.6,label='flat minimum')
    # a small perturbation band
    ax.axvspan(-0.6,0.6,color=MUTED,alpha=.12)
    ax.scatter([0,0],[0,0],color=INK,s=30,zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-0.5,10)
    ax.set_xlabel(r'parameter shift $\Delta\theta$'); ax.set_ylabel('loss')
    ax.legend(frameon=False,fontsize=11,loc='upper center')
    ax.set_title('flat minima resist parameter / data shift',fontsize=12)
    save(fig,'sharp_flat')

for f in [f_landscape,f_lr_schedules,f_sharp_flat,f_momentum_vector]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

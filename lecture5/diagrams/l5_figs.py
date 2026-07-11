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

# ── 2 — learning-rate: four GD trajectories on L=(theta-3)^2 ──
def f_lr_trajectories():
    L=lambda t:(t-3)**2; grad=lambda t:2*(t-3)
    cfgs=[(0.05,'too small',TEAL),(0.4,'good',GREEN),(0.9,'large (oscillates)',ACC),(1.05,'diverges',RED)]
    fig,axes=plt.subplots(1,4,figsize=(9.4,2.5))
    for ax,(eta,ttl,c) in zip(axes,cfgs):
        th=np.linspace(-1.5,7.5,300); ax.plot(th,L(th),color=INK,lw=2.0,alpha=.8)
        t=0.0; pts=[t]
        for _ in range(12):
            t=t-eta*grad(t); pts.append(t)
            if abs(t)>1e3: break
        pts=np.array(pts); pts=np.clip(pts,-1.5,7.5)
        ax.plot(pts,L(np.clip(pts,-1.5,7.5)),'-o',color=c,ms=4,lw=1.4)
        ax.scatter([3],[0],marker='*',s=130,color=RED,zorder=5)
        ax.set_title(f'$\\eta$={eta}\n{ttl}',fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-3,26)
    save(fig,'lr_trajectories')

# ── 3 — batch-size noise: full / mini / tiny batch paths on a bowl ──
def f_batch_noise():
    rng=np.random.default_rng(3)
    g=np.linspace(-2.6,2.6,220); X,Y=np.meshgrid(g,g); Z=X**2+3*Y**2
    def run(sigma,eta=0.10,steps=40):
        p=np.array([-2.2,2.0]); pts=[p.copy()]
        for _ in range(steps):
            noise=rng.normal(0,sigma,2)
            p=p-eta*(np.array([2*p[0],6*p[1]])+noise); pts.append(p.copy())
        return np.array(pts)
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.0))
    cfgs=[(0.0,'full batch',TEAL),(2.2,'minibatch',ACC),(6.0,'tiny batch',RED)]
    for ax,(s,ttl,c) in zip(axes,cfgs):
        ax.contour(X,Y,Z,levels=9,colors=[MUTED],linewidths=.6,alpha=.6)
        pts=run(s); ax.plot(pts[:,0],pts[:,1],'-o',color=c,ms=2.4,lw=1.2,alpha=.9)
        ax.scatter([0],[0],marker='*',s=140,color=INK,zorder=5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=12)
    save(fig,'batch_noise')

# ── 4 & 5 — ravine / momentum_vs_gd — RETIRED.
# Both are now native ml-field: contour(L) with the descent() trajectory computed
# in-deck (GD zigzag, and GD-vs-momentum side by side). Same ravine L=x²+100y².
# See lecture5/L5-optimization.typ ("The problem: ravines" / "Why momentum helps").

# ── 6 — optimizer comparison on L = x^2 + 50 y^2 ──
def f_optimizer_compare():
    a,b=1.0,50.0
    def grad(p): return np.array([2*a*p[0],2*b*p[1]])
    p0=np.array([-2.4,0.85]); N=60
    rng=np.random.default_rng(0)
    def gd(eta=0.014):
        p=p0.copy(); P=[p.copy()]
        for _ in range(N): p=p-eta*grad(p); P.append(p.copy())
        return np.array(P)
    def sgd(eta=0.014,sig=6.0):
        p=p0.copy(); P=[p.copy()]
        for _ in range(N): p=p-eta*(grad(p)+rng.normal(0,sig,2)); P.append(p.copy())
        return np.array(P)
    def momentum(eta=0.006,beta=0.9):
        p=p0.copy(); v=np.zeros(2); P=[p.copy()]
        for _ in range(N):
            v=beta*v+grad(p); p=p-eta*v; P.append(p.copy())
        return np.array(P)
    def adam(eta=0.16,b1=0.9,b2=0.999,eps=1e-8):
        p=p0.copy(); m=np.zeros(2); v=np.zeros(2); P=[p.copy()]
        for t in range(1,N+1):
            g=grad(p); m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
            mh=m/(1-b1**t); vh=v/(1-b2**t)
            p=p-eta*mh/(np.sqrt(vh)+eps); P.append(p.copy())
        return np.array(P)
    gx=np.linspace(-2.7,2.7,260); gy=np.linspace(-1.0,1.0,260); X,Y=np.meshgrid(gx,gy); Z=a*X**2+b*Y**2
    fig,ax=plt.subplots(figsize=(7.6,3.2))
    ax.contour(X,Y,Z,levels=12,colors=[MUTED],linewidths=.55,alpha=.55)
    for pts,lbl,c in [(gd(),'GD',INK),(sgd(),'SGD',TEAL),(momentum(),'Momentum',BLUE),(adam(),'Adam',ACC)]:
        ax.plot(pts[:,0],pts[:,1],'-o',color=c,ms=2.2,lw=1.5,label=lbl,alpha=.92)
    ax.scatter([0],[0],marker='*',s=170,color=RED,zorder=6)
    ax.set_xlim(-2.7,2.7); ax.set_ylim(-1.0,1.0); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(r'$L=x^2+50\,y^2$',fontsize=12)
    ax.legend(frameon=False,fontsize=11,loc='upper right',ncol=2)
    save(fig,'optimizer_compare')

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

for f in [f_landscape,f_lr_trajectories,f_batch_noise,
          f_optimizer_compare,f_lr_schedules,f_sharp_flat,f_momentum_vector]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

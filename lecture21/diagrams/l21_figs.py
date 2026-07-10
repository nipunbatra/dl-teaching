"""Metropolis-style figures for Lecture 21 (Diffusion Models — Theory).
Schematic / synthetic ONLY — NO real images. "Data" is 2-D point clouds and 1-D mixtures.
Transparent bg, ink + orange/teal palette. Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).

The STRUCTURAL diagrams (forward/reverse Markov chains, training flow, sampling loop,
noise-prediction network, U-Net, reverse-mean arrows) are NATIVE fletcher in the deck.
These matplotlib plots are the quantitative ones:
  1. forward_filmstrip — a 2-D data cloud getting noisier as t grows  (q(x_t|x_0))
  2. noise_schedule    — the linear beta_t schedule and the cumulative abar_t
  3. signal_noise      — sqrt(abar_t) (signal) and sqrt(1-abar_t) (noise) vs t  (SNR)
  4. predict_noise     — x_t = sqrt(abar) x_0 + sqrt(1-abar) eps, the network predicts eps
  5. reverse_density   — 1-D marginal p_t(x): one Gaussian blob at t=T -> two modes at t=0
  6. denoise_traj      — honest 1-D DDPM reverse trajectories (analytic score) -> modes +/-2
Run from repo root:  python3 lecture21/diagrams/l21_figs.py
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
  'lines.linewidth':2.6,'lines.solid_capstyle':'round',
})
CMAP=LinearSegmentedColormap.from_list('metro',['#2C7A7B','#EFEEEB','#EB811B'])
OUT='lecture21/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── shared linear DDPM schedule (DDPM, Ho et al. 2020) ──
T=1000
BETA=np.linspace(1e-4,0.02,T)
ALPHA=1-BETA
ABAR=np.cumprod(ALPHA)
TS=np.arange(1,T+1)

def two_moons(n=380,seed=3):
    rng=np.random.default_rng(seed)
    th=np.pi*rng.random(n)
    x1=np.c_[np.cos(th),         np.sin(th)]
    x2=np.c_[1-np.cos(th), 0.4-np.sin(th)]
    X=np.r_[x1,x2]+0.05*rng.standard_normal((2*n,2))
    X=X-X.mean(0); X=X/X.std(0)/1.6           # roughly unit-ish scale
    return X

# ── 1 — forward filmstrip: a data cloud gets noisier as t increases ──
def f_forward_filmstrip():
    X=two_moons(); rng=np.random.default_rng(0)
    eps=rng.standard_normal(X.shape)
    picks=[0,150,400,700,999]
    fig,axes=plt.subplots(1,5,figsize=(12.4,2.9))
    col=np.where(X[:,0]<0,0.0,1.0)            # colour by original mode (left/right)
    for ax,t in zip(axes,picks):
        a=ABAR[t]
        xt=np.sqrt(a)*X+np.sqrt(1-a)*eps
        ax.scatter(xt[:,0],xt[:,1],c=col,cmap=CMAP,s=7,alpha=0.8,edgecolors='none')
        ax.set_xlim(-3.2,3.2); ax.set_ylim(-3.2,3.2); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        lab=(r'$x_0$  (data)' if t==0 else (r'$x_T\!\approx\mathcal{N}(0,I)$' if t==999 else fr'$x_{{{t+1}}}$'))
        ax.set_title(lab,fontsize=12.5,color=INK,pad=4)
        ax.text(0.5,-0.12,fr'$\bar\alpha_t={a:.2f}$',transform=ax.transAxes,ha='center',fontsize=10.5,color=MUTED)
    # a long right arrow under the strip
    fig.text(0.5,0.02,r'forward process  $q(x_t\mid x_{t-1})$  —  add a little Gaussian noise each step  $\longrightarrow$',
             ha='center',fontsize=12,color=ACC,weight=600)
    save(fig,'forward_filmstrip')

# ── 2 — the noise schedule: beta_t (linear) and the cumulative abar_t ──
def f_noise_schedule():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,3.5))
    ax1.plot(TS,BETA,color=RED)
    ax1.fill_between(TS,BETA,color=RED,alpha=0.12)
    ax1.set_title(r'variance schedule  $\beta_t$  (linear)',fontsize=12.5,color=INK)
    ax1.set_xlabel('timestep  $t$'); ax1.set_ylabel(r'$\beta_t$')
    ax1.annotate(r'$\beta_1=10^{-4}$',xy=(1,BETA[0]),xytext=(150,0.004),fontsize=11,color=RED,
                 arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.4))
    ax1.annotate(r'$\beta_T=0.02$',xy=(T,BETA[-1]),xytext=(430,0.017),fontsize=11,color=RED,
                 arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.4))
    ax2.plot(TS,ABAR,color=TEAL)
    ax2.fill_between(TS,ABAR,color=TEAL,alpha=0.12)
    ax2.set_title(r'signal retained  $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$',fontsize=12.5,color=INK)
    ax2.set_xlabel('timestep  $t$'); ax2.set_ylabel(r'$\bar\alpha_t$'); ax2.set_ylim(-0.02,1.03)
    ax2.text(70,0.86,'nearly all\nsignal',fontsize=10.5,color=TEAL)
    ax2.text(560,0.10,r'$\bar\alpha_T\!\approx\!0$   (pure noise)',fontsize=10.5,color=MUTED)
    save(fig,'noise_schedule')

# ── 3 — signal vs noise coefficients: sqrt(abar) falls, sqrt(1-abar) rises ──
def f_signal_noise():
    fig,ax=plt.subplots(figsize=(7.4,3.7))
    s=np.sqrt(ABAR); nn=np.sqrt(1-ABAR)
    ax.plot(TS,s,color=TEAL,label=r'$\sqrt{\bar\alpha_t}$   (signal weight)')
    ax.plot(TS,nn,color=ACC,label=r'$\sqrt{1-\bar\alpha_t}$   (noise weight)')
    # crossover
    k=int(np.argmin(np.abs(s-nn)))
    ax.axvline(TS[k],color=MUTED,ls='--',lw=1.3)
    ax.scatter([TS[k]],[s[k]],color=INK,zorder=5,s=28)
    ax.annotate('SNR = 1\n(signal = noise)',xy=(TS[k],s[k]),xytext=(TS[k]+90,0.72),
                fontsize=10.5,color=INK,arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.3))
    ax.text(30,0.05,'mostly signal',fontsize=10.5,color=TEAL)
    ax.text(720,0.05,'mostly noise',fontsize=10.5,color=ACC)
    ax.set_xlabel('timestep  $t$'); ax.set_ylabel('coefficient'); ax.set_ylim(-0.02,1.05)
    ax.set_title(r'$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$  —  a falling signal-to-noise ratio',
                 fontsize=11.5,color=INK)
    ax.legend(frameon=False,fontsize=11,loc='center right')
    save(fig,'signal_noise')

# ── 4 — predict the noise: x_t = signal + noise, network estimates eps ──
def f_predict_noise():
    X=two_moons(n=160,seed=9)
    t=430; a=ABAR[t]
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    # faint clean data cloud (the prior)
    ax.scatter(X[:,0],X[:,1],s=11,color=MUTED,alpha=0.30,edgecolors='none')
    ax.text(-1.4,1.55,'clean data\n$x_0$',fontsize=11,color=MUTED,ha='center')
    # one clean point -> signal, plus a known noise vector -> x_t
    x0=np.array([1.25,0.85]); eps=np.array([1.9,-1.7])
    sig=np.sqrt(a)*x0
    xt=sig+np.sqrt(1-a)*eps
    epsh=eps+np.array([0.55,0.30])                 # a slightly-off network estimate
    xt_h=sig+np.sqrt(1-a)*epsh
    # true noise arrow (signal -> x_t)
    ax.annotate('',xy=xt,xytext=sig,arrowprops=dict(arrowstyle='-|>',color=ACC,lw=3.0))
    # predicted noise arrow (dashed, ends slightly off)
    ax.annotate('',xy=xt_h,xytext=sig,arrowprops=dict(arrowstyle='-|>',color=GREEN,lw=2.4,ls=(0,(4,2))))
    ax.scatter(*sig,color=TEAL,s=110,zorder=6)
    ax.scatter(*xt,color=INK,s=130,zorder=6)
    ax.scatter(*xt_h,color=GREEN,s=70,zorder=6)
    ax.annotate(r'$\sqrt{\bar\alpha_t}\,x_0$  (signal)',sig,textcoords='offset points',
               xytext=(-118,-4),fontsize=12,color=TEAL,weight=600)
    ax.annotate(r'noisy input  $x_t$',xt,textcoords='offset points',
               xytext=(14,-2),fontsize=12,color=INK,weight=600)
    ax.annotate(r'true noise  $\epsilon$',(0.52*sig+0.48*xt),textcoords='offset points',
               xytext=(16,10),fontsize=12,color=ACC,weight=600)
    ax.annotate(r'$\epsilon_\theta(x_t,t)$',xt_h,textcoords='offset points',
               xytext=(12,8),fontsize=12,color=GREEN,weight=600)
    ax.set_xlim(-2.9,3.6); ax.set_ylim(-3.2,2.3); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sname in ax.spines.values(): sname.set_visible(False)
    ax.set_title(r'the supervised target is the injected noise;  train  $\epsilon_\theta(x_t,t)\approx\epsilon$',
                 fontsize=12.5,color=INK,pad=10)
    save(fig,'predict_noise')

# ── 1-D two-mode mixture, analytic forward marginal + score (modes +/-2) ──
MU, SIG0, W = 2.0, 0.30, 0.5                   # p_0 = 0.5 N(+2,SIG0^2) + 0.5 N(-2,SIG0^2)
def _norm(x,m,v): return np.exp(-0.5*(x-m)**2/v)/np.sqrt(2*np.pi*v)
def marg_pt(x,t):
    a=ABAR[t]; m=np.sqrt(a)*MU; v=a*SIG0**2+(1-a)
    return W*_norm(x,m,v)+W*_norm(x,-m,v)
def score_pt(x,t):
    a=ABAR[t]; m=np.sqrt(a)*MU; v=a*SIG0**2+(1-a)
    p1=W*_norm(x,m,v); p2=W*_norm(x,-m,v); p=p1+p2+1e-30
    return (p1*(-(x-m)/v)+p2*(-(x+m)/v))/p     # d/dx log p_t
def eps_star(x,t):                              # optimal noise prediction = -sqrt(1-abar) * score
    return -np.sqrt(1-ABAR[t])*score_pt(x,t)

# ── 5 — reverse density joyplot: N(0,1) at t=T melts into two modes at t=0 ──
def f_reverse_density():
    fig,ax=plt.subplots(figsize=(7.6,4.4))
    levels=[999,800,600,400,200,60,0]
    xs=np.linspace(-4.2,4.2,600)
    off=0.0; step=0.9
    for i,t in enumerate(levels):
        p=marg_pt(xs,t); p=p/p.max()*0.8
        y0=(len(levels)-1-i)*step
        c=CMAP(i/(len(levels)-1))
        ax.fill_between(xs,y0,y0+p,color=c,alpha=0.85,edgecolor=INK,linewidth=1.0)
        lab=(r'$t=T$  (noise)' if t==999 else (r'$t=0$  (data)' if t==0 else fr'$t={t}$'))
        ax.text(4.35,y0+0.05,lab,fontsize=10.5,color=INK,va='bottom')
    ax.annotate('reverse process\n(denoise)',xy=(-3.6,0.4),xytext=(-3.6,4.9),
                fontsize=11,color=GREEN,weight=600,ha='center',
                arrowprops=dict(arrowstyle='-|>',color=GREEN,lw=2.2))
    ax.set_xlim(-4.4,6.0); ax.set_ylim(-0.2,6.4)
    ax.set_xlabel('value  $x$'); ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(r'$p_t(x)$: one Gaussian blob at $t=T$  $\to$  two data modes at $t=0$',fontsize=12,color=INK)
    save(fig,'reverse_density')

# ── 6 — honest DDPM reverse trajectories (analytic score) converge to +/-2 ──
def f_denoise_traj():
    rng=np.random.default_rng(1)
    n=24
    x=rng.standard_normal(n)                    # x_T ~ N(0,I)
    rec=np.zeros((T+1,n)); rec[T]=x
    for t in range(T-1,-1,-1):                  # ancestral sampling t: T-1 ... 0  (index = timestep-1)
        a=ALPHA[t]; ab=ABAR[t]
        eps=eps_star(x,t)
        mean=(x-(1-a)/np.sqrt(1-ab)*eps)/np.sqrt(a)
        if t>0:
            sig=np.sqrt(BETA[t]); x=mean+sig*rng.standard_normal(n)
        else:
            x=mean
        rec[t]=x
    fig,ax=plt.subplots(figsize=(8.2,4.2))
    tt=np.arange(T+1)
    for j in range(n):
        c=ACC if rec[0,j]>0 else TEAL
        ax.plot(tt,rec[:,j],color=c,lw=1.3,alpha=0.75)
    ax.axhline(2,color=INK,ls='--',lw=1.2); ax.axhline(-2,color=INK,ls='--',lw=1.2)
    ax.text(1010,2,'mode $+2$',fontsize=11,color=ACC,va='center')
    ax.text(1010,-2,'mode $-2$',fontsize=11,color=TEAL,va='center')
    ax.set_xlim(0,1120); ax.invert_xaxis()      # start at t=T on the left
    ax.set_xlabel(r'timestep  $t$   (reverse: $T\to 0$)'); ax.set_ylabel('value  $x_t$')
    ax.set_title('reverse denoising trajectories: from $\\mathcal{N}(0,1)$ noise to the data modes',
                 fontsize=12,color=INK)
    save(fig,'denoise_traj')

if __name__=='__main__':
    for f in [f_forward_filmstrip,f_noise_schedule,f_signal_noise,
              f_predict_noise,f_reverse_density,f_denoise_traj]:
        f(); print('ok',f.__name__)
    # ---- arithmetic self-checks printed for the deck ----
    print('CHECK sqrt(0.64)=',np.sqrt(0.64),' sqrt(0.36)=',np.sqrt(0.36),
          ' x_t=',np.sqrt(0.64)*1+np.sqrt(0.36)*(-0.5))
    print('CHECK abar_T=',ABAR[-1],' sqrt(abar_T)=',np.sqrt(ABAR[-1]))
    print('done ->',OUT)

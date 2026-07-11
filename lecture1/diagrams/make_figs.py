"""Metropolis-style figures for Lecture 1 (Why These Losses?).
Transparent background, Fira-ish sans (DejaVu fallback), ink + orange/teal/green accents."""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['Fira Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.2,'lines.solid_capstyle':'round',
})
OUT='lecture1/figures'
os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    # SVG for the Marp deck (Chrome renders it); PNG for the Typst deck (resvg
    # mangles some path-text glyph advances, so we hand Typst a rasterized copy).
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)
def gauss(x,mu=0,s=1): return np.exp(-0.5*((x-mu)/s)**2)/(s*np.sqrt(2*np.pi))
def arrow(ax,x0,y0,x1,y1,c=MUTED,lw=2):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=16,color=c,lw=lw))
def box(ax,x,y,w,h,text,fc='#EFEEEB',ec=INK,fs=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.04',
                fc=fc,ec=ec,lw=1.4)); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK)

# 1 net_to_distribution
def f_net():
    fig,ax=plt.subplots(figsize=(4.2,2.6)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,6)
    layers=[(1.2,[1.5,3,4.5]),(3.3,[1,2.3,3.7,5]),(5.4,[2,3.5])]
    for lx,ys in layers:
        for y in ys: ax.add_patch(plt.Circle((lx,y),0.28,fc='#FFFFFF',ec=INK,lw=1.4,zorder=3))
    for (lx,ys),(nx,nys) in zip(layers[:-1],layers[1:]):
        for y in ys:
            for ny in nys: ax.plot([lx+.28,nx-.28],[y,ny],color=MUTED,lw=0.7,alpha=.6,zorder=1)
    xx=np.linspace(6.6,9.6,120); yy=gauss(xx,8.1,0.55); yy=yy/yy.max()*3+1.5
    ax.plot(xx,yy,color=ACC,lw=2.6); ax.fill_between(xx,1.5,yy,color=ACC,alpha=.12)
    arrow(ax,5.75,3,6.5,3.0,MUTED)
    ax.text(8.1,0.9,r'$p_\theta(y\mid x)$',ha='center',fontsize=13,color=INK)
    ax.text(3.3,5.7,'neural network',ha='center',fontsize=11,color=MUTED)
    save(fig,'net_to_distribution')

# 2 density_area
def f_density_area():
    fig,ax=plt.subplots(figsize=(5.2,2.9)); x=np.linspace(-4,4,400); y=gauss(x,0.2,1.1)
    ax.plot(x,y,color=INK,lw=2.4)
    a,b=-0.6,1.7; m=(x>=a)&(x<=b); ax.fill_between(x[m],0,y[m],color=ACC,alpha=.35)
    ax.annotate('P(a ≤ Y ≤ b) = area',(0.55,0.12),(1.9,0.30),color=INK,fontsize=13,
                arrowprops=dict(arrowstyle='-|>',color=MUTED))
    ax.set_yticks([]); ax.set_xticks([a,b]); ax.set_xticklabels(['a','b']); ax.set_ylim(0,0.45)
    ax.set_xlabel('y'); ax.set_ylabel('density  p(y)')
    save(fig,'density_area')

# 3 uniform
def f_uniform():
    fig,ax=plt.subplots(figsize=(5.0,2.8)); a,b=1,4; h=1/(b-a)
    ax.hlines(0,-1,a,color=INK,lw=2.4); ax.hlines(0,b,6,color=INK,lw=2.4)
    ax.hlines(h,a,b,color=INK,lw=2.4); ax.vlines(a,0,h,color=INK,lw=2.4,ls=':'); ax.vlines(b,0,h,color=INK,lw=2.4,ls=':')
    ax.fill_between([a,b],0,h,color=ACC,alpha=.25)
    ax.text((a+b)/2,h+0.04,r'$\frac{1}{b-a}$',ha='center',fontsize=15)
    ax.text(-0.3,0.06,r'$-\log 0=+\infty$',color=RED,fontsize=12)
    ax.set_xticks([a,b]); ax.set_xticklabels(['a','b']); ax.set_yticks([]); ax.set_ylim(-0.05,0.55); ax.set_xlabel('y')
    save(fig,'uniform')

# 4 gaussian_params
def f_gaussian_params():
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.7)); x=np.linspace(-6,6,400)
    for mu,c in [(-2,TEAL),(0,INK),(2,ACC)]: axes[0].plot(x,gauss(x,mu,1),color=c,lw=2.2)
    axes[0].set_title(r'changing $\mu$',fontsize=13);
    for s,c in [(0.6,ACC),(1.1,INK),(2.0,TEAL)]: axes[1].plot(x,gauss(x,0,s),color=c,lw=2.2)
    axes[1].set_title(r'changing $\sigma$',fontsize=13)
    for a in axes: a.set_yticks([]); a.set_xticks([]); a.set_xlabel('y')
    save(fig,'gaussian_params')

# 5 gaussian_to_square
def f_g2s():
    fig,axes=plt.subplots(1,4,figsize=(8.2,2.3)); x=np.linspace(-3,3,300)
    g=gauss(x,0,1)
    axes[0].plot(x,g,color=INK); axes[0].set_title('density',fontsize=12)
    axes[1].plot(x,np.log(g),color=TEAL); axes[1].set_title(r'$\log p$',fontsize=12)
    axes[2].plot(x,-np.log(g),color=ACC); axes[2].set_title(r'$-\log p$',fontsize=12)
    axes[3].plot(x,x**2,color=ACC); axes[3].set_title(r'$(y-\mu)^2$',fontsize=12)
    for a in axes: a.set_xticks([]); a.set_yticks([])
    fig.text(0.5,-0.02,'Gaussian  →  log  →  negative log  →  squared error',ha='center',color=MUTED,fontsize=11)
    save(fig,'gaussian_to_square')

# 6 mvn_triptych
def f_mvn():
    fig,axes=plt.subplots(1,3,figsize=(8.4,2.9))
    g=np.linspace(-3,3,120); X,Y=np.meshgrid(g,g)
    covs=[('$\\Sigma=\\tau^2 I$',np.array([[1,0],[0,1.]])),
          ('diagonal $\\Sigma$',np.array([[2.2,0],[0,0.55]])),
          ('full $\\Sigma$',np.array([[1.4,0.95],[0.95,1.0]]))]
    for ax,(t,C) in zip(axes,covs):
        Ci=np.linalg.inv(C); Z=np.exp(-0.5*(Ci[0,0]*X**2+2*Ci[0,1]*X*Y+Ci[1,1]*Y**2))
        ax.contour(X,Y,Z,levels=6,colors=ACC,linewidths=1.3)
        ax.set_title(t,fontsize=12); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0,color=MUTED,lw=.6); ax.axvline(0,color=MUTED,lw=.6)
    save(fig,'mvn_triptych')

# 7 regression_conditionals
def f_regconds():
    fig,ax=plt.subplots(figsize=(6.4,3.2)); x=np.linspace(0,10,200)
    f=lambda t: 2+1.6*np.sin(0.6*t)+0.12*t
    ax.plot(x,f(x),color=ACC,lw=2.6,zorder=3,label=r'$f_\theta(x)$')
    rng=np.random.default_rng(1)
    xs=rng.uniform(0.5,9.5,26); ax.scatter(xs,f(xs)+rng.normal(0,0.5,xs.size),s=16,color=INK,alpha=.55,zorder=2)
    for xc in [1.5,4,6.5,9]:
        yb=np.linspace(-1.6,1.6,80); bell=gauss(yb,0,0.55); bell=bell/bell.max()*0.9
        ax.plot(xc+bell,f(xc)+yb,color=TEAL,lw=1.6,zorder=4)
        ax.plot([xc,xc],[f(xc)-1.6,f(xc)+1.6],color=TEAL,lw=0.7,ls=':',zorder=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='upper left',frameon=False,fontsize=11)
    save(fig,'regression_conditionals')

# 8 residual_losses
def f_resloss():
    fig,ax=plt.subplots(figsize=(5.4,3.1)); r=np.linspace(-3,3,400)
    ax.plot(r,r**2,color=ACC,lw=2.4,label=r'Gaussian  $r^2$')
    ax.plot(r,np.abs(r)*1.6,color=TEAL,lw=2.2,label=r'Laplace  $|r|$')
    ax.plot(r,2.2*np.log(1+r**2/1.0),color=GREEN,lw=2.2,label=r'Student-$t$')
    c=2.2; ax.plot([-c,-c],[0,9],color=MUTED,lw=2,ls='--'); ax.plot([c,c],[0,9],color=MUTED,lw=2,ls='--',label='Uniform (barrier)')
    ax.set_ylim(0,9); ax.set_xlabel('residual  r'); ax.set_ylabel(r'$-\log p(r)$'); ax.set_yticks([])
    ax.legend(frameon=False,fontsize=10,loc='upper center')
    save(fig,'residual_losses')

# 9 softmax_pipeline
def f_softpipe():
    fig,ax=plt.subplots(figsize=(7.6,2.6)); ax.axis('off'); ax.set_xlim(0,15); ax.set_ylim(0,6)
    box(ax,0.3,2.2,1.6,1.6,'input\nx'); arrow(ax,2.0,3,2.9,3)
    box(ax,3.0,2.0,2.0,2.0,'neural\nnet')
    arrow(ax,5.1,3,6.0,3); z=[1.4,0.6,-0.8]
    for i,v in enumerate(z):
        ax.bar(6.4+i*0.5,v,0.4,bottom=3,color=INK if v>=0 else MUTED)
    ax.text(6.9,5.2,'logits z',ha='center',fontsize=11); ax.axhline(3,6.2/15,8.1/15,color=MUTED,lw=.8)
    arrow(ax,8.3,3,9.4,3,ACC); ax.text(8.85,3.5,'softmax',ha='center',fontsize=10,color=ACC)
    p=[0.55,0.30,0.15]
    for i,v in enumerate(p): ax.bar(9.9+i*0.55,v*3,0.44,bottom=1.8,color=ACC)
    ax.text(10.7,5.2,'probabilities',ha='center',fontsize=11); ax.text(10.7,1.4,'sum = 1',ha='center',fontsize=10,color=MUTED)
    save(fig,'softmax_pipeline')

# 10 sigmoid
def f_sigmoid():
    fig,ax=plt.subplots(figsize=(4.2,2.8)); z=np.linspace(-6,6,300); p=1/(1+np.exp(-z))
    ax.plot(z,p,color=ACC,lw=2.6); ax.axhline(1,color=MUTED,ls='--',lw=1); ax.axhline(0,color=MUTED,ls='--',lw=1)
    ax.plot(0,0.5,'o',color=INK,ms=7); ax.axhline(0.5,-6,0,color=MUTED,lw=.6,ls=':')
    ax.set_xlabel('z'); ax.set_ylabel(r'$\sigma(z)$'); ax.set_yticks([0,0.5,1]); ax.set_ylim(-0.05,1.1)
    save(fig,'sigmoid')

# 11 softmax_bars
def f_softbars():
    fig,axes=plt.subplots(1,2,figsize=(5.8,2.6),gridspec_kw={'wspace':0.5})
    axes[0].bar([0,1,2],[2,1,-0.5],color=[INK,MUTED,MUTED]); axes[0].set_title('logits z',fontsize=12,pad=8); axes[0].axhline(0,color=INK,lw=.8); axes[0].set_ylim(-0.9,2.4)
    axes[1].bar([0,1,2],[0.63,0.25,0.12],color=ACC); axes[1].set_title('probabilities',fontsize=12,pad=8); axes[1].set_ylim(0,0.85); axes[1].text(1,0.72,'sum = 1',ha='center',color=MUTED,fontsize=11)
    for a in axes: a.set_xticks([0,1,2]); a.set_xticklabels(['1','2','3']); a.set_yticks([])
    fig.text(0.5,0.5,'softmax →',ha='center',color=ACC,fontsize=12)
    save(fig,'softmax_bars')

# 12 neglog_curve
def f_neglog():
    fig,ax=plt.subplots(figsize=(4.6,3.0)); p=np.linspace(0.002,1,300)
    ax.plot(p,-np.log(p),color=ACC,lw=2.6)
    for pv,lab in [(0.9,'0.11'),(0.1,'2.30'),(0.001,'6.91')]:
        yv=-np.log(pv); ax.plot(pv,yv,'o',color=INK,ms=6); ax.plot([pv,pv],[0,yv],color=MUTED,ls=':',lw=1)
        ax.annotate(f'p={pv}\n$\\ell$≈{lab}',(pv,yv),(pv+0.06,yv+0.4),fontsize=9,color=INK)
    ax.set_xlabel('probability on the true class'); ax.set_ylabel(r'loss  $-\log p_y$'); ax.set_ylim(0,7.2)
    save(fig,'neglog_curve')

# 13 gradient_bars
def f_gradbars():
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.4),gridspec_kw={'wspace':0.45})
    p=[0.55,0.30,0.15]; y=[0,1,0]; d=[a-b for a,b in zip(p,y)]
    axes[0].bar([0,1,2],p,color=ACC); axes[0].set_title('p',fontsize=12)
    axes[1].bar([0,1,2],y,color=TEAL); axes[1].set_title('y  (one-hot)',fontsize=12)
    axes[2].bar([0,1,2],d,color=[MUTED if v<0 else INK for v in d]); axes[2].set_title(r'p $-$ y  = gradient',fontsize=12); axes[2].axhline(0,color=INK,lw=.8)
    for a in axes: a.set_xticks([0,1,2]); a.set_xticklabels(['1','2','3']); a.set_yticks([])
    save(fig,'gradient_bars')

# 14 bayes_classifier
def f_bayes():
    fig,ax=plt.subplots(figsize=(5.6,3.0)); x=np.linspace(-4,7,400)
    ax.plot(x,gauss(x,0,1),color=TEAL,lw=2.4,label='p(x | Y=0)')
    ax.plot(x,gauss(x,3,1.2),color=ACC,lw=2.4,label='p(x | Y=1)')
    xs=1.6; ax.axvline(xs,color=INK,ls='--',lw=1.4); ax.text(xs+0.1,0.38,r'$x^\star$',color=INK,fontsize=13)
    ax.plot(xs,gauss(xs,0,1),'o',color=TEAL,ms=7); ax.plot(xs,gauss(xs,3,1.2),'o',color=ACC,ms=7)
    ax.set_yticks([]); ax.set_xlabel('x'); ax.legend(frameon=False,fontsize=10,loc='upper right')
    save(fig,'bayes_classifier')

# 15 priors_shift_boundary
def f_priors():
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True); x=np.linspace(-4,7,400)
    for ax,(p1,tt) in zip(axes,[(0.5,'balanced prior'),(0.1,'rare positive (0.1)')]):
        d0=gauss(x,0,1)*(1-p1); d1=gauss(x,3,1)*p1
        ax.plot(x,gauss(x,0,1),color=TEAL,lw=1.6,alpha=.6); ax.plot(x,gauss(x,3,1),color=ACC,lw=1.6,alpha=.6)
        # boundary where posteriors equal
        idx=np.argmin(np.abs(d0-d1)[x>-1])+np.sum(x<=-1); xb=x[idx]
        ax.axvline(xb,color=INK,ls='--',lw=2); ax.text(xb,0.42,f'boundary',ha='center',fontsize=9,color=INK)
        ax.set_title(tt,fontsize=12); ax.set_yticks([]); ax.set_xlabel('x')
    save(fig,'priors_shift_boundary')

# 16 overfit_polynomial
def f_overfit():
    fig,ax=plt.subplots(figsize=(5.0,2.9)); rng=np.random.default_rng(3)
    xs=np.linspace(0,1,8); ys=np.sin(2*np.pi*xs)+rng.normal(0,0.12,xs.size)
    xf=np.linspace(0,1,300); c=np.polyfit(xs,ys,7); ax.plot(xf,np.polyval(c,xf),color=ACC,lw=2.4,label='overfit (deg 7)')
    ax.plot(xf,np.sin(2*np.pi*xf),color=MUTED,lw=1.6,ls='--',label='true trend')
    ax.scatter(xs,ys,s=32,color=INK,zorder=3)
    ax.set_ylim(-1.8,2.0); ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False,fontsize=9,loc='upper right')
    save(fig,'overfit_polynomial')

# 17 prior_likelihood_posterior / mle_map_contours — RETIRED.
# Now computed natively in the deck from real Gaussians via chalkdust ml-field:
#   contour((likelihood, prior, posterior), marks: (MLE, MAP, 0))
# See lecture1/L1-probabilistic-view.typ, "Bayes' rule over parameters".

# 18 posterior_samples
def f_postsamp():
    fig,ax=plt.subplots(figsize=(5.6,3.1)); rng=np.random.default_rng(5)
    xs=np.linspace(0,1,15); ys=np.sin(2*np.pi*xs)*0.9+rng.normal(0,0.12,xs.size)
    xf=np.linspace(0,1,200); base=np.sin(2*np.pi*xf)*0.9
    curves=[]
    for k in range(6):
        c=np.polyfit(xs,ys+rng.normal(0,0.14,xs.size),5); yv=np.polyval(c,xf); curves.append(yv)
        ax.plot(xf,yv,color=ACC,lw=1.1,alpha=.5)
    C=np.array(curves); m=C.mean(0); s=C.std(0)
    ax.fill_between(xf,m-2*s,m+2*s,color=ACC,alpha=.12)
    ax.scatter(xs,ys,s=30,color=INK,zorder=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_ylim(-1.8,1.8)
    save(fig,'posterior_samples')

for fn in [f_net,f_density_area,f_uniform,f_gaussian_params,f_g2s,f_mvn,f_regconds,f_resloss,
           f_softpipe,f_sigmoid,f_softbars,f_neglog,f_gradbars,f_bayes,f_priors,f_overfit,
           f_postsamp]:
    try: fn(); print('ok',fn.__name__)
    except Exception as e: print('FAIL',fn.__name__,e)
print('done')

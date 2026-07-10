"""Metropolis-style figures for Lecture 19 (Autoencoders & Variational Autoencoders).
Schematic / synthetic ONLY -- NO real images. "Generated images" are procedural shapes.
Transparent bg, ink + orange/teal palette. Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Every ARCHITECTURE / FLOW diagram (autoencoder, denoising AE, VAE + reparameterization,
graphical model, ELBO decomposition, train/gen loop, conditional VAE, latent diffusion) is
NATIVE fletcher in the deck. These schematic / plot figures only:
  1. latent_islands  -- AE latent: disconnected clusters + gaps; prior samples land in gaps
  2. latent_manifold -- VAE 2-D latent decoded on a grid: smooth procedural manifold
  3. gaussian_kl     -- KL(N(mu,sigma^2)||N(0,1)) vs sigma for several mu; min at (0,1)
  4. kl_recon        -- reconstruction-vs-KL Pareto frontier, beta slides along it
  5. blurry          -- pixelwise likelihood averages many sharp outputs -> blur
  6. posterior_collapse -- per-dim KL over training: healthy (active units) vs collapsed
  7. beta_sweep      -- beta vs reconstruction error and KL (twin axis)
  8. interpolation   -- latent interpolation: smooth VAE path vs broken AE path
Run from repo root:  python3 lecture19/diagrams/l19_figs.py
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Circle
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
OUT='lecture19/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── 1 — AE latent islands: encoded points cluster off-origin, prior samples fall in gaps ──
def f_latent_islands():
    rng=np.random.default_rng(3)
    centers=[(-3.0,2.6),(3.1,3.0),(-2.7,-2.9),(3.3,-2.2),(-0.2,4.4)]
    cols=[TEAL,ACC,BLUE,GREEN,RED]
    fig,ax=plt.subplots(figsize=(6.6,5.4))
    # prior N(0,I) mass — faint rings at 1 and 2 sigma
    for r,a in ((1,0.16),(2,0.09)):
        ax.add_patch(Circle((0,0),r,facecolor=MUTED,edgecolor='none',alpha=a,zorder=0))
    for (cx,cy),c in zip(centers,cols):
        pts=rng.normal((cx,cy),0.42,(70,2))
        ax.scatter(pts[:,0],pts[:,1],s=13,color=c,alpha=0.8,edgecolors='none',zorder=2)
    # prior samples: most near origin -> land in the empty middle
    zs=rng.normal(0,1,(9,2))
    ax.scatter(zs[:,0],zs[:,1],s=140,marker='X',color=INK,zorder=4,edgecolors='white',linewidths=1.3)
    ax.annotate(r'$z\sim\mathcal{N}(0,I)$ lands here',xy=(0.3,0.2),xytext=(0.1,-4.7),
                ha='center',fontsize=12.5,color=INK,weight=600,
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.8))
    ax.text(0,-5.9,'gap between islands  ->  decoder outputs nonsense',ha='center',fontsize=11.5,color=RED)
    ax.text(-3.0,2.6+1.0,'encoded data',ha='center',fontsize=11,color=MUTED)
    ax.set_xlim(-5,5); ax.set_ylim(-6.3,5.4); ax.set_aspect('equal')
    ax.set_xlabel(r'$z_1$'); ax.set_ylabel(r'$z_2$')
    ax.set_title('plain autoencoder: latent space is a set of islands',fontsize=12.5,color=INK)
    save(fig,'latent_islands')

# ── 2 — VAE 2-D latent decoded on a grid: procedural morphing "generated images" ──
def f_latent_manifold():
    fig,ax=plt.subplots(figsize=(6.4,6.0))
    g=np.linspace(-2,2,9)
    def sig(t): return 1/(1+np.exp(-t))
    for z1 in g:
        for z2 in g:
            w=0.30+0.30*sig(1.3*z1); h=0.30+0.30*sig(1.3*z2)
            ang=42*np.tanh(0.45*z1*z2)
            col=CMAP(0.5+0.5*np.tanh((z1+z2)/4.2))
            ax.add_patch(Ellipse((z1,z2),w,h,angle=ang,facecolor=col,edgecolor=INK,lw=0.7))
            ax.add_patch(Circle((z1,z2),0.05,facecolor=INK,edgecolor='none',alpha=0.55))
    ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5); ax.set_aspect('equal')
    ax.set_xlabel(r'latent $z_1$',fontsize=13); ax.set_ylabel(r'latent $z_2$',fontsize=13)
    ax.set_xticks([-2,-1,0,1,2]); ax.set_yticks([-2,-1,0,1,2])
    ax.set_title(r'decode a grid of $z$: the learned manifold varies smoothly',fontsize=12,color=INK)
    save(fig,'latent_manifold')

# ── 3 — Gaussian KL vs sigma for several mu; minimum at mu=0, sigma=1 (KL=0) ──
def f_gaussian_kl():
    s=np.linspace(0.18,3.0,500)
    fig,ax=plt.subplots(figsize=(7.0,4.2))
    for mu,c in ((0.0,TEAL),(0.5,BLUE),(1.0,ACC),(2.0,RED)):
        kl=0.5*(mu**2+s**2-np.log(s**2)-1.0)
        ax.plot(s,kl,color=c,label=fr'$\mu={mu}$')
    ax.scatter([1.0],[0.0],s=90,color=TEAL,zorder=5,edgecolors='white',linewidths=1.4)
    ax.annotate('minimum:  $\\mu=0,\\ \\sigma=1$\n$\\Rightarrow$ matches the prior  (KL $=0$)',
                xy=(1.0,0.0),xytext=(1.35,1.6),fontsize=11.5,color=INK,
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.6))
    ax.axhline(0,color=MUTED,lw=0.8,ls='--')
    ax.set_xlabel(r'$\sigma$  (posterior std)'); ax.set_ylabel(r'$D_{\mathrm{KL}}(\,q\;\Vert\;\mathcal{N}(0,1)\,)$')
    ax.set_ylim(-0.3,4.2); ax.set_xlim(0,3)
    ax.legend(frameon=False,fontsize=11,loc='upper center',ncol=4,handlelength=1.2,columnspacing=1.0)
    ax.set_title(r'$D_{\mathrm{KL}}=\frac{1}{2}(\mu^2+\sigma^2-\log\sigma^2-1)$',fontsize=12.5,pad=8)
    save(fig,'gaussian_kl')

# ── 4 — reconstruction vs KL Pareto frontier; beta slides along it ──
def f_kl_recon():
    kl=np.linspace(0.35,9,400)
    recon=1.4+9.0/kl                # convex trade-off frontier (both are losses, lower better)
    fig,ax=plt.subplots(figsize=(6.8,4.6))
    ax.plot(kl,recon,color=INK,lw=2.6,zorder=2)
    ax.fill_between(kl,recon,recon.max()+2,color=MUTED,alpha=0.06)
    pts=[(0.9,'high  $\\beta$',ACC,'low KL, blurry\nrecon'),
         (2.4,'$\\beta\\approx1$',TEAL,'balanced'),
         (7.2,'low  $\\beta$',BLUE,'sharp recon,\nunruly latent')]
    for k,lab,c,note in pts:
        r=1.4+9.0/k
        ax.scatter([k],[r],s=150,color=c,zorder=5,edgecolors='white',linewidths=1.5)
        dy = 1.7 if c!=ACC else 2.2
        ax.annotate(lab,xy=(k,r),xytext=(k,r+dy),ha='center',fontsize=12,color=c,weight=600,
                    arrowprops=dict(arrowstyle='-|>',color=c,lw=1.4))
        ax.text(k,r-1.1,note,ha='center',fontsize=9.5,color=MUTED)
    ax.set_xlabel(r'KL term  $D_{\mathrm{KL}}(q\Vert p)$   (organize latent)')
    ax.set_ylabel('reconstruction term\n(explain the data)')
    ax.set_xlim(0,9.5); ax.set_ylim(0,14)
    ax.set_title(r'the ELBO picks a balance; $\beta$ slides you along the frontier',fontsize=12,color=INK)
    save(fig,'kl_recon')

# ── 5 — blurry samples: several sharp plausible outputs, and their pixelwise average ──
def f_blurry():
    rng=np.random.default_rng(1)
    n=96; ax_=np.linspace(-1,1,n); X,Y=np.meshgrid(ax_,ax_)
    def ring(cx,cy,r0,w=0.045):
        r=np.sqrt((X-cx)**2+(Y-cy)**2); return np.exp(-((r-r0)/w)**2)
    samples=[ring(*p) for p in [(-0.18,0.10,0.55),(0.14,-0.12,0.52),(-0.02,0.20,0.60),(0.20,0.05,0.48)]]
    avg=np.mean(samples,axis=0)
    fig,axes=plt.subplots(1,5,figsize=(11.2,2.6))
    for k in range(3):
        axes[k].imshow(samples[k],cmap=CMAP,vmin=0,vmax=1); axes[k].set_title(f'sharp sample {k+1}',fontsize=11,color=INK)
    axes[3].text(0.5,0.5,'$+\\ \\cdots$\n\naverage',ha='center',va='center',fontsize=15,color=MUTED)
    axes[3].axis('off')
    axes[4].imshow(avg,cmap=CMAP,vmin=0,vmax=1)
    axes[4].set_title('pixelwise mean = blurry',fontsize=11,color=RED)
    for k in (0,1,2,4):
        axes[k].set_xticks([]); axes[k].set_yticks([])
        for s in axes[k].spines.values(): s.set_edgecolor(MUTED); s.set_linewidth(1.0)
    fig.suptitle('each sample is a plausible sharp output; a per-pixel likelihood rewards their average',
                 fontsize=12,color=INK,y=1.06)
    save(fig,'blurry')

# ── 6 — posterior collapse: per-dimension KL over training, healthy vs collapsed ──
def f_posterior_collapse():
    rng=np.random.default_rng(5)
    t=np.linspace(0,1,120); fig,axes=plt.subplots(1,2,figsize=(10.4,4.0),sharey=True)
    D=6
    # healthy: some dims stay active (KL > 0), a few decay
    ax=axes[0]
    finals=[1.9,1.4,0.9,0.5,0.06,0.05]
    for d,(fv,c) in enumerate(zip(finals,[TEAL,ACC,BLUE,GREEN,MUTED,MUTED])):
        y=fv+ (2.6-fv)*np.exp(-4.5*t)+rng.normal(0,0.015,t.size)
        ax.plot(t,y,color=c,lw=2.2,alpha=0.9)
    ax.set_title('healthy: several active latent units',fontsize=12,color=INK)
    ax.text(0.98,2.05,'active',ha='right',fontsize=10.5,color=TEAL,weight=600)
    # collapsed: every dim -> ~0  (q(z|x) -> p(z))
    ax=axes[1]
    for d in range(D):
        fv=rng.uniform(0.01,0.06)
        y=fv+(2.4-fv)*np.exp(-9*t)+rng.normal(0,0.012,t.size)
        ax.plot(t,y,color=RED,lw=2.0,alpha=0.55)
    ax.axhline(0.0,color=MUTED,lw=0.8,ls='--')
    ax.set_title(r'collapsed: all dims $\to 0$   ($q(z\mid x)\approx p(z)$)',fontsize=12,color=INK)
    ax.text(0.98,0.18,'latent unused',ha='right',fontsize=10.5,color=RED,weight=600)
    for ax in axes:
        ax.set_xlabel('training progress'); ax.set_xlim(0,1); ax.set_ylim(-0.15,2.7)
    axes[0].set_ylabel('per-dimension KL (nats)')
    save(fig,'posterior_collapse')

# ── 7 — beta sweep: reconstruction error and KL vs beta (twin axis) ──
def f_beta_sweep():
    b=np.logspace(-1,1.3,200)
    recon=1.0+2.4*np.log1p(b)          # reconstruction error grows with beta
    klc=6.5/(1+0.9*b)                  # KL (latent info retained) shrinks with beta
    fig,ax=plt.subplots(figsize=(7.0,4.3))
    ax.set_xscale('log')
    l1=ax.plot(b,recon,color=ACC,lw=2.6,label='reconstruction error')[0]
    ax.set_xlabel(r'$\beta$  (KL weight, log scale)'); ax.set_ylabel('reconstruction error',color=ACC)
    ax.tick_params(axis='y',colors=ACC)
    ax2=ax.twinx(); ax2.spines['top'].set_visible(False)
    l2=ax2.plot(b,klc,color=TEAL,lw=2.6,label='latent info (KL)')[0]
    ax2.set_ylabel('latent information used (KL)',color=TEAL); ax2.tick_params(axis='y',colors=TEAL)
    ax.axvline(1.0,color=MUTED,lw=1.0,ls='--')
    ax.text(1.05,ax.get_ylim()[1]*0.92,r'$\beta=1$'+'\n(standard VAE)',fontsize=10,color=MUTED)
    ax.text(0.11,1.2,'low $\\beta$:\nsharp, unruly',fontsize=10,color=BLUE)
    ax.text(11,5.2,'high $\\beta$:\norganized,\nblurry / collapse',fontsize=10,color=RED,ha='right')
    ax.legend(handles=[l1,l2],frameon=False,fontsize=11,loc='center left')
    ax.set_title(r'higher $\beta$: more organized latent, worse reconstruction',fontsize=12,color=INK)
    save(fig,'beta_sweep')

# ── 8 — latent interpolation: smooth VAE path vs broken AE path ──
def f_interpolation():
    fig,axes=plt.subplots(2,1,figsize=(9.6,3.6))
    ts=np.linspace(0,1,7)
    def sig(t): return 1/(1+np.exp(-t))
    # good VAE: shape morphs smoothly A -> B
    ax=axes[0]
    for i,t in enumerate(ts):
        z1=-1.8+3.6*t; z2=1.6-3.0*t
        w=0.34+0.30*sig(1.3*z1); h=0.34+0.30*sig(1.3*z2); ang=42*np.tanh(0.45*z1*z2)
        col=CMAP(0.5+0.5*np.tanh((z1+z2)/4.2))
        ax.add_patch(Ellipse((i,0),w,h,angle=ang,facecolor=col,edgecolor=INK,lw=0.9))
    ax.text(-0.9,0,'A',ha='center',va='center',fontsize=14,color=INK,weight=700)
    ax.text(6.9,0,'B',ha='center',va='center',fontsize=14,color=INK,weight=700)
    ax.set_title('VAE latent: interpolation decodes to a gradual, valid morph',fontsize=11.5,color=TEAL)
    ax.set_xlim(-1.4,7.4); ax.set_ylim(-0.7,0.7)
    # bad AE: valid, nonsense, valid ...
    ax=axes[1]; rng=np.random.default_rng(4)
    for i,t in enumerate(ts):
        if i in (0,6):
            ax.add_patch(Ellipse((i,0),0.55,0.5,facecolor=CMAP(0.2 if i==0 else 0.85),edgecolor=INK,lw=0.9))
        else:
            # scrambled speckle = nonsense reconstruction in a latent gap
            xs=rng.uniform(-0.3,0.3,26)+i; ys=rng.uniform(-0.3,0.3,26)
            ax.scatter(xs,ys,s=16,color=RED,alpha=0.6,edgecolors='none')
    ax.text(-0.9,0,'A',ha='center',va='center',fontsize=14,color=INK,weight=700)
    ax.text(6.9,0,'B',ha='center',va='center',fontsize=14,color=INK,weight=700)
    ax.set_title('plain AE: the path crosses empty latent gaps -> invalid decodings',fontsize=11.5,color=RED)
    ax.set_xlim(-1.4,7.4); ax.set_ylim(-0.7,0.7)
    for ax in axes:
        ax.set_aspect('equal'); ax.axis('off')
    save(fig,'interpolation')

for f in [f_latent_islands,f_latent_manifold,f_gaussian_kl,f_kl_recon,
          f_blurry,f_posterior_collapse,f_beta_sweep,f_interpolation]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

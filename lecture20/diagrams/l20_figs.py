"""Metropolis-style figures for Lecture 20 (Generative Adversarial Networks).
Schematic / synthetic ONLY — NO real images. Transparent bg, ink + orange/teal palette.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).

Every STRUCTURAL diagram (adversarial G/D setup, training loop, gradient path,
detach, mode-collapse illustration, conditional GAN, image-to-image, generator
geometry) is NATIVE fletcher in the deck. These plots are the quantitative /
distribution figures only:
  1. sat_vs_nonsat  — saturating log(1-D) vs non-saturating -log D generator loss + gradient magnitude
  2. optimal_d      — two 1-D densities p_data, p_G and the optimal discriminator D*(x)=pd/(pd+pg)
  3. js_value       — the value C(G)=-log4+2*JSD as p_G shifts; min -log4 at p_G=p_data
  4. decision_surf  — 2-D discriminator field D(x), real vs fake clouds, D=0.5 boundary, generator gradient
  5. toy_2d         — ring of 8 Gaussians: good coverage vs mode collapse (generated overlaid)
  6. coverage       — 1-D multimodal real + three generator behaviours (blurry-broad / sharp-partial / good)
  7. earth_mover    — Wasserstein transport: moving probability mass from p_G to p_data across a line
  8. training_curves— oscillating, non-monotone D and G losses (loss curves are not enough)
  9. latent_interp  — schematic latent interpolation z0 -> z1 morphing smoothly

Run from repo root:  python3 lecture20/diagrams/l20_figs.py
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
OUT='lecture20/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

def gauss(x,mu,sig): return np.exp(-0.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi))
def sigmoid(z): return 1/(1+np.exp(-z))

# ── 1 — saturating vs non-saturating generator loss + gradient magnitude ──
def f_sat_vs_nonsat():
    D=np.linspace(0.001,0.999,400)
    L_sat = np.log(1-D)        # original minimax generator term (minimized)
    L_non = -np.log(D)         # non-saturating generator loss
    g_sat = 1/(1-D)            # |d/dD log(1-D)|
    g_non = 1/D                # |d/dD (-log D)|
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,3.7))
    ax1.plot(D,L_sat,color=RED,label=r'saturating  $\log(1-D)$')
    ax1.plot(D,L_non,color=TEAL,label=r'non-saturating  $-\log D$')
    ax1.axvline(0.01,color=MUTED,ls='--',lw=1.3)
    ax1.text(0.03,3.6,r'$D(G(z))\approx0$'+'\n(early training)',fontsize=10,color=MUTED)
    ax1.set_xlabel(r'$D(G(z))$  (how real the detective thinks the fake is)',fontsize=11)
    ax1.set_ylabel('generator loss',fontsize=11)
    ax1.set_ylim(-0.3,5.0); ax1.legend(frameon=False,fontsize=10.5,loc='upper right')
    ax1.set_title('what the generator minimizes',fontsize=11.5)
    ax2.plot(D,g_sat,color=RED,label=r'saturating  $|\,1/(1-D)\,|$')
    ax2.plot(D,g_non,color=TEAL,label=r'non-saturating  $|\,1/D\,|$')
    ax2.axvline(0.01,color=MUTED,ls='--',lw=1.3)
    ax2.set_yscale('log'); ax2.set_ylim(0.5,2e3)
    ax2.set_xlabel(r'$D(G(z))$',fontsize=11)
    ax2.set_ylabel('gradient magnitude (log)',fontsize=11)
    ax2.legend(frameon=False,fontsize=10.5,loc='upper right')
    ax2.annotate(r'$\approx100\times$ stronger'+'\nsignal at D=0.01',xy=(0.01,100),xytext=(0.28,150),
                 fontsize=10,color=INK,arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.6))
    ax2.set_title('gradient near a confident detective',fontsize=11.5)
    fig.tight_layout()
    save(fig,'sat_vs_nonsat')
    print(f'  sat loss@0.01 = log(0.99) = {np.log(0.99):.5f}   non-sat = -log(0.01) = {-np.log(0.01):.5f}')
    print(f'  |grad| sat@0.01 = {1/(1-0.01):.4f}   non-sat = {1/0.01:.4f}   ratio = {(1/0.01)/(1/(1-0.01)):.2f}x')

# ── 2 — optimal discriminator D*(x) = p_data / (p_data + p_G) ──
def f_optimal_d():
    x=np.linspace(-6,6,600)
    pd=gauss(x,-1.4,1.0); pg=gauss(x,1.4,1.0)
    Dstar=pd/(pd+pg)
    fig,ax=plt.subplots(figsize=(7.6,3.9))
    ax.fill_between(x,pd,color=BLUE,alpha=0.28,label=r'$p_\mathrm{data}(x)$')
    ax.fill_between(x,pg,color=ACC,alpha=0.28,label=r'$p_G(x)$')
    ax.plot(x,pd,color=BLUE,lw=2.2); ax.plot(x,pg,color=ACC,lw=2.2)
    ax.set_ylabel('density',fontsize=12); ax.set_xlabel('x',fontsize=12)
    ax.set_ylim(0,0.62)
    ax2=ax.twinx()
    ax2.plot(x,Dstar,color=INK,lw=2.8,label=r'$D^*(x)=\frac{p_\mathrm{data}}{p_\mathrm{data}+p_G}$')
    ax2.axhline(0.5,color=MUTED,ls='--',lw=1.3)
    ax2.set_ylabel(r'$D^*(x)$',fontsize=12); ax2.set_ylim(0,1.02)
    ax2.spines['top'].set_visible(False)
    # crossing point where pd=pg -> D*=0.5 (x=0 by symmetry)
    ax2.plot(0,0.5,'o',color=RED,ms=9,zorder=5)
    ax2.annotate(r'$p_\mathrm{data}=p_G\Rightarrow D^*=\frac{1}{2}$',xy=(0,0.5),xytext=(1.3,0.24),
                 fontsize=10.5,color=RED,arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6))
    l1,lab1=ax.get_legend_handles_labels(); l2,lab2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,lab1+lab2,frameon=False,fontsize=10.5,loc='upper left')
    ax.set_title(r'the best possible detective, for a fixed generator',fontsize=12)
    fig.tight_layout(); save(fig,'optimal_d')
    print(f'  D*(x=0) with symmetric pd,pg = {(gauss(0,-1.4,1.0)/(gauss(0,-1.4,1.0)+gauss(0,1.4,1.0))):.4f}')

# ── 3 — value C(G) = -log4 + 2 JSD ; min -log4 at p_G = p_data ──
def f_js_value():
    x=np.linspace(-12,12,4000); dx=x[1]-x[0]
    def jsd(mg):
        pd=gauss(x,0,1.0); pg=gauss(x,mg,1.0); m=0.5*(pd+pg)
        def kl(p,q):
            mask=p>1e-12
            return np.sum(p[mask]*np.log(p[mask]/q[mask]))*dx
        return 0.5*kl(pd,m)+0.5*kl(pg,m)
    mus=np.linspace(-6,6,120)
    C=np.array([-np.log(4)+2*jsd(mg) for mg in mus])
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    ax.plot(mus,C,color=INK,lw=2.8)
    ax.axhline(-np.log(4),color=RED,ls='--',lw=1.5)
    ax.text(-5.8,-np.log(4)+0.06,r'$-\log 4\approx-1.386$',fontsize=11,color=RED)
    ax.plot(0,-np.log(4),'o',color=TEAL,ms=10,zorder=5)
    ax.annotate(r'$p_G=p_\mathrm{data}$'+'\n(global minimum)',xy=(0,-np.log(4)),xytext=(1.6,-0.55),
                fontsize=10.5,color=TEAL,arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=1.6))
    ax.set_xlabel(r'generator mean shift  $\mu_G$  (data mean at 0)',fontsize=11)
    ax.set_ylabel(r'value at optimal $D$:  $C(G)$',fontsize=11)
    ax.set_title(r'substituting $D^*$ turns the game into a divergence',fontsize=12)
    fig.tight_layout(); save(fig,'js_value')
    print(f'  -log4 = {-np.log(4):.5f}   C(G) min at mu_G=0 = {(-np.log(4)+2*jsd(0.0)):.5f}')

# ── 4 — 2-D discriminator decision surface + generator gradient on fakes ──
def f_decision_surf():
    rng=np.random.default_rng(3)
    real=rng.normal([-1.5,0.0],0.55,(120,2))
    fake=rng.normal([1.5,0.2],0.55,(120,2))
    gx,gy=np.meshgrid(np.linspace(-4,4,300),np.linspace(-3,3,220))
    # D high (real) on the left; smooth logistic boundary near x0=0
    Dfield=sigmoid(-2.1*gx)
    fig,ax=plt.subplots(figsize=(7.6,3.9))
    cf=ax.contourf(gx,gy,Dfield,levels=np.linspace(0,1,21),cmap=CMAP,alpha=0.9)
    ax.contour(gx,gy,Dfield,levels=[0.5],colors=[INK],linewidths=2.6)
    ax.scatter(real[:,0],real[:,1],s=16,color=BLUE,edgecolor='white',linewidth=0.4,label='real  (want D=1)',zorder=3)
    ax.scatter(fake[:,0],fake[:,1],s=16,color=RED,edgecolor='white',linewidth=0.4,label='fake  (D pushes to 0)',zorder=3)
    # generator gradient: fakes climb toward higher D (leftward)
    for k in range(0,120,12):
        ax.annotate('',xy=(fake[k,0]-0.9,fake[k,1]),xytext=(fake[k,0],fake[k,1]),
                    arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.7))
    ax.text(0.2,-2.75,r'$D=0.5$ boundary',fontsize=10,color=INK)
    ax.text(0.6,2.45,'fakes climb toward\nthe D≈1 (real) region',fontsize=10.5,color=INK,ha='center')
    cb=fig.colorbar(cf,ax=ax,fraction=0.045,pad=0.02); cb.set_label(r'$D(x)$',fontsize=11)
    ax.set_xlim(-4,4); ax.set_ylim(-3,3); ax.set_aspect('equal')
    ax.legend(frameon=True,facecolor='white',framealpha=0.85,fontsize=10,loc='lower left')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('discriminator field and the direction fakes move',fontsize=12)
    fig.tight_layout(); save(fig,'decision_surf')

# ── 5 — ring of 8 Gaussians: good coverage vs mode collapse ──
def f_toy_2d():
    rng=np.random.default_rng(5)
    ang=np.linspace(0,2*np.pi,8,endpoint=False)
    centers=np.stack([3*np.cos(ang),3*np.sin(ang)],1)
    real=np.concatenate([rng.normal(c,0.28,(60,2)) for c in centers])
    good=np.concatenate([rng.normal(c,0.30,(45,2)) for c in centers])
    # collapse: nearly all mass on 2 modes
    coll=np.concatenate([rng.normal(centers[1],0.30,(190,2)),rng.normal(centers[5],0.30,(170,2))])
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.6))
    for ax,gen,ttl,col in [(axes[0],good,'good coverage\ngenerated mass on every mode',GREEN),
                           (axes[1],coll,'mode collapse\ngenerated mass on a few modes',RED)]:
        ax.scatter(real[:,0],real[:,1],s=12,color=BLUE,alpha=0.35,edgecolor='none',label='real data',zorder=1)
        ax.scatter(gen[:,0],gen[:,1],s=12,color=ACC,alpha=0.65,edgecolor='none',label='generated',zorder=2)
        for c in centers: ax.add_patch(plt.Circle(c,0.55,fill=False,ec=MUTED,lw=1.0,ls='--'))
        ax.set_title(ttl,fontsize=12,color=col)
        ax.set_xlim(-4.5,4.5); ax.set_ylim(-4.5,4.5); ax.set_aspect('equal'); ax.axis('off')
        ax.legend(frameon=False,fontsize=10,loc='upper right')
    fig.tight_layout(); save(fig,'toy_2d')

# ── 6 — coverage vs realism on a 1-D multimodal target ──
def f_coverage():
    x=np.linspace(-6,6,600)
    real=(gauss(x,-3.2,0.45)+gauss(x,0,0.45)+gauss(x,3.2,0.45))/3
    blurry=gauss(x,0,2.4)                                   # broad, covers all but low fidelity
    sharp=gauss(x,0,0.45)                                    # one crisp mode only
    good=(gauss(x,-3.2,0.5)+gauss(x,0,0.5)+gauss(x,3.2,0.5))/3
    fig,axes=plt.subplots(1,3,figsize=(10.4,3.3),sharey=True)
    panels=[(blurry,'blurry broad coverage','high recall, low precision',ACC),
            (sharp,'sharp partial coverage','high precision, low recall',RED),
            (good,'good — sharp and complete','high precision and recall',GREEN)]
    for ax,(g,ttl,sub,col) in zip(axes,panels):
        ax.fill_between(x,real,color=BLUE,alpha=0.22)
        ax.plot(x,real,color=BLUE,lw=2.0,label='real')
        ax.plot(x,g,color=col,lw=2.8,label='generated')
        ax.set_title(ttl,fontsize=11.5,color=col)
        ax.text(0,-0.14,sub,fontsize=9.8,color=MUTED,ha='center',transform=ax.get_xaxis_transform())
        ax.set_yticks([]); ax.set_xticks([-3.2,0,3.2]); ax.set_xticklabels(['mode 1','mode 2','mode 3'],fontsize=9)
        ax.legend(frameon=False,fontsize=9.5,loc='upper right')
    fig.tight_layout(); save(fig,'coverage')

# ── 7 — Wasserstein earth-mover transport ──
def f_earth_mover():
    fig,ax=plt.subplots(figsize=(8.2,3.4))
    xs=np.arange(6)
    pg=np.array([0.05,0.10,0.55,0.25,0.05,0.0])
    pd=np.array([0.0,0.05,0.10,0.30,0.40,0.15])
    w=0.38
    ax.bar(xs-w/2,pg,w,color=ACC,edgecolor='white',label=r'$p_G$  (generated mass)')
    ax.bar(xs+w/2,pd,w,color=BLUE,edgecolor='white',label=r'$p_\mathrm{data}$  (real mass)')
    # transport arrows: move mass rightward from p_G toward p_data
    for (a,b,h) in [(2,4,0.56),(3,4,0.40),(1,3,0.28)]:
        ax.annotate('',xy=(b,h),xytext=(a,h),
                    arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.0,
                                    connectionstyle="arc3,rad=-0.26"))
    ax.text(2.9,0.86,'earth-mover: least total work to reshape '+r'$p_G$ into $p_\mathrm{data}$',
            fontsize=10.5,color=INK,ha='center')
    ax.set_xlabel('sample value (1-D)',fontsize=11); ax.set_ylabel('probability mass',fontsize=11)
    ax.set_ylim(0,0.96); ax.set_xticks(xs)
    ax.legend(frameon=False,fontsize=10.5,loc='upper left')
    ax.set_title('Wasserstein distance = cost of moving mass',fontsize=12)
    fig.tight_layout(); save(fig,'earth_mover')

# ── 8 — oscillating, non-monotone GAN training curves ──
def f_training_curves():
    rng=np.random.default_rng(9)
    t=np.linspace(0,1,400)
    Dl=0.62+0.16*np.sin(13*t)+0.05*np.sin(41*t)+0.04*rng.standard_normal(400)
    Gl=0.95+0.22*np.sin(13*t+1.1)+0.06*np.sin(37*t)+0.05*rng.standard_normal(400)
    fig,ax=plt.subplots(figsize=(7.8,3.6))
    ax.plot(t*100,Dl,color=ACC,lw=2.0,label='discriminator loss')
    ax.plot(t*100,Gl,color=TEAL,lw=2.0,label='generator loss')
    ax.axhline(np.log(2),color=MUTED,ls='--',lw=1.3)
    ax.text(50,np.log(2)-0.005,r'$\log 2\approx0.693$  (equilibrium hint)',fontsize=9.8,color=MUTED,
            ha='center',va='center',bbox=dict(facecolor='white',alpha=0.85,edgecolor='none',pad=1.5))
    ax.set_xlabel('training iteration (thousands)',fontsize=11)
    ax.set_ylabel('loss',fontsize=11); ax.set_ylim(0.2,1.5)
    ax.legend(frameon=False,fontsize=10.5,loc='upper right')
    ax.set_title('losses oscillate — a lower D loss is not a better generator',fontsize=11.5)
    fig.tight_layout(); save(fig,'training_curves')

# ── 9 — schematic latent interpolation z0 -> z1 ──
def f_latent_interp():
    n=7
    fig,ax=plt.subplots(figsize=(9.6,2.2))
    for k in range(n):
        a=k/(n-1)
        # schematic "image": a small colored disc whose hue morphs smoothly
        col=(1-a)*np.array([0.17,0.48,0.48])+a*np.array([0.92,0.51,0.11])  # teal -> orange
        r=0.30+0.10*np.sin(a*np.pi)                                        # shape also morphs
        ax.add_patch(plt.Circle((k*1.2,0),r,facecolor=col,edgecolor=INK,lw=1.4))
        ax.text(k*1.2,-0.62,f'{a:.2f}',ha='center',fontsize=9,color=MUTED)
    ax.annotate('',xy=(n*1.2-0.6,0),xytext=(-0.55,0),
                arrowprops=dict(arrowstyle='-|>',color=MUTED,lw=1.4,alpha=0.6))
    ax.text(0,0.62,r'$z_0$',ha='center',fontsize=12,color=TEAL,weight=600)
    ax.text((n-1)*1.2,0.62,r'$z_1$',ha='center',fontsize=12,color=ACC,weight=600)
    ax.text((n-1)*0.6,-0.95,r'interpolation weight  $\alpha$:  $z_\alpha=(1-\alpha)z_0+\alpha z_1$',
            ha='center',fontsize=10.5,color=INK)
    ax.set_xlim(-0.9,n*1.2-0.3); ax.set_ylim(-1.15,0.9); ax.set_aspect('equal'); ax.axis('off')
    save(fig,'latent_interp')

for f in [f_sat_vs_nonsat,f_optimal_d,f_js_value,f_decision_surf,f_toy_2d,
          f_coverage,f_earth_mover,f_training_curves,f_latent_interp]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

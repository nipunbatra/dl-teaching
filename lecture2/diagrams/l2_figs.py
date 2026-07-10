"""Metropolis-style plot figures for Lecture 2 (Linear Models to MLPs).
Transparent background, ink + orange/teal/green accents. Emits SVG (for any
Marp/web use) and PNG (dpi 200, what the Typst deck reads). Run from repo root:
    python3 lecture2/diagrams/l2_figs.py
Architecture / graph diagrams are drawn natively in Typst (fletcher), not here."""
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
OUT='lecture2/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)
def sig(z): return 1/(1+np.exp(-z))

# 1 — linear regression geometry: points, fitted line, residuals
def f_linreg():
    rng=np.random.default_rng(1); x=np.linspace(0.4,5.6,11); y=0.8*x+1+rng.normal(0,0.6,x.size)
    w,b=0.8,1.05
    fig,ax=plt.subplots(figsize=(4.6,3.0)); xs=np.linspace(0,6,2)
    for xi,yi in zip(x,y): ax.plot([xi,xi],[yi,w*xi+b],color=RED,lw=1.3,alpha=.8)
    ax.plot(xs,w*xs+b,color=ACC,lw=2.6,label=r'$\hat y=w^\top x+b$')
    ax.scatter(x,y,s=40,color=INK,zorder=3)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False,loc='upper left',fontsize=12); save(fig,'linreg_geometry')

# 2 — sigmoid
def f_sigmoid():
    z=np.linspace(-6,6,300); fig,ax=plt.subplots(figsize=(4.2,2.7))
    ax.plot(z,sig(z),color=TEAL,lw=2.8); ax.axhline(.5,color=MUTED,lw=.8,ls='--'); ax.axvline(0,color=MUTED,lw=.8,ls='--')
    ax.scatter([0],[.5],color=ACC,zorder=3,s=36); ax.annotate(r'$\sigma(0)=0.5$',(0,.5),(1.1,.30),color=INK,fontsize=12,
        arrowprops=dict(arrowstyle='-|>',color=MUTED))
    ax.set_xlabel('z'); ax.set_ylabel(r'$\sigma(z)$'); ax.set_yticks([0,.5,1]); save(fig,'sigmoid')

# 3 — activation zoo (curves + derivatives faint)
def f_activations():
    z=np.linspace(-4,4,300)
    acts=[('sigmoid',sig(z),sig(z)*(1-sig(z))),('tanh',np.tanh(z),1-np.tanh(z)**2),
          ('ReLU',np.maximum(0,z),(z>0).astype(float)),('GELU',z*sig(1.702*z),None)]
    fig,axes=plt.subplots(1,4,figsize=(9.6,2.4));
    for ax,(nm,a,d) in zip(axes,acts):
        ax.plot(z,a,color=INK,lw=2.4)
        if d is not None: ax.plot(z,d,color=ACC,lw=1.6,ls='--',alpha=.9)
        ax.set_title(nm,fontsize=13,pad=6); ax.axhline(0,color=MUTED,lw=.6); ax.axvline(0,color=MUTED,lw=.6)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].plot([],[],color=INK,label=r'$\phi$'); axes[0].plot([],[],color=ACC,ls='--',label=r"$\phi'$")
    axes[0].legend(frameon=False,fontsize=11,loc='upper left'); save(fig,'activations')

# 4 — MSE bowl (contour) with gradient-descent path
def f_mse_gd():
    w=np.linspace(-1,3,120); b=np.linspace(-1,3,120); W,B=np.meshgrid(w,b)
    wt,bt=1.6,1.2; Z=(W-wt)**2+0.6*(B-bt)**2+0.5*(W-wt)*(B-bt)
    fig,ax=plt.subplots(figsize=(4.3,3.1)); ax.contour(W,B,Z,levels=12,colors=[TEAL],linewidths=.8,alpha=.7)
    # gd path
    p=np.array([-0.5,2.6]); eta=0.18; pts=[p.copy()]
    for _ in range(14):
        g=np.array([2*(p[0]-wt)+0.5*(p[1]-bt),1.2*(p[1]-bt)+0.5*(p[0]-wt)]); p=p-eta*g; pts.append(p.copy())
    pts=np.array(pts); ax.plot(pts[:,0],pts[:,1],'-o',color=ACC,ms=3.5,lw=1.6)
    ax.scatter([wt],[bt],marker='*',s=180,color=RED,zorder=4)
    ax.set_xlabel('w'); ax.set_ylabel('b'); ax.set_xticks([]); ax.set_yticks([]); save(fig,'mse_gd')

# 5 — linear decision boundary (2 classes)
def f_linear_boundary():
    rng=np.random.default_rng(3)
    a=rng.normal([1.2,1.2],.5,(25,2)); c=rng.normal([3.2,3.0],.5,(25,2))
    fig,ax=plt.subplots(figsize=(4.0,3.0)); ax.scatter(*a.T,color=TEAL,s=32,label='y=0')
    ax.scatter(*c.T,color=ACC,s=32,label='y=1'); xs=np.linspace(0,4.5,2)
    ax.plot(xs,-0.9*xs+4.4,color=INK,lw=2.2); ax.text(3.0,3.7,r'$w^\top x+b=0$',color=INK,fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False,fontsize=11,loc='lower left'); save(fig,'linear_boundary')

# 6 — XOR not linearly separable
def f_xor():
    fig,ax=plt.subplots(figsize=(3.4,3.0))
    ax.scatter([0,1],[0,1],color=TEAL,s=90,zorder=3,label='class A')
    ax.scatter([0,1],[1,0],color=ACC,s=90,marker='s',zorder=3,label='class B')
    for ang in (0,): pass
    ax.plot([-.3,1.3],[1.05,-.05],color=MUTED,lw=1.4,ls='--'); ax.plot([-.3,1.3],[.4,1.6],color=MUTED,lw=1.4,ls='--')
    ax.text(-.28,1.5,'no single line\nseparates them',color=RED,fontsize=12)
    ax.set_xlim(-.4,1.5); ax.set_ylim(-.3,1.9); ax.set_xticks([0,1]); ax.set_yticks([0,1])
    # legend above the plot so it never sits on the corner markers
    ax.legend(frameon=False,fontsize=11,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2,
              handletextpad=0.3,columnspacing=1.2); save(fig,'xor')

# 7 — feature transform: input (not sep) -> hidden (sep)
def f_feature_transform():
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.0),gridspec_kw={'wspace':0.35})
    A=np.array([[0,0],[1,1]]); B=np.array([[0,1],[1,0]])
    axes[0].scatter(*A.T,color=TEAL,s=90); axes[0].scatter(*B.T,color=ACC,marker='s',s=90)
    axes[0].set_title('input space  x',fontsize=13,pad=6); axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    # hidden coords (ReLU features make it separable)
    Ah=np.array([[0.2,0.15],[1.8,0.15]]); Bh=np.array([[0.6,1.35],[1.4,1.35]])
    axes[1].scatter(*Ah.T,color=TEAL,s=90); axes[1].scatter(*Bh.T,color=ACC,marker='s',s=90)
    xs=np.linspace(-.2,2.1,2); axes[1].plot(xs,0.7+0*xs,color=INK,lw=2.2)
    axes[1].set_title(r'hidden space  h$=\phi$(W₁x+b₁)',fontsize=13,pad=6); axes[1].set_xticks([]); axes[1].set_yticks([])
    save(fig,'feature_transform')

# 8 — sum of ReLUs approximating a curve (least-squares fit so it actually tracks)
def f_relu_build():
    x=np.linspace(-3,3,400); target=np.sin(1.5*x)+0.3*x
    kinks=np.linspace(-2.7,2.7,12)
    Phi=np.column_stack([np.maximum(0,x-k) for k in kinks]+[np.ones_like(x)])  # ReLU basis + bias
    coef,*_=np.linalg.lstsq(Phi,target,rcond=None); approx=Phi@coef
    fig,ax=plt.subplots(figsize=(5.0,3.0))
    for j in range(0,12,3): ax.plot(x,coef[j]*np.maximum(0,x-kinks[j]),color=TEAL,lw=0.7,alpha=.28)
    ax.plot(x,target,color=INK,lw=2.6,label='target f(x)')
    ax.plot(x,approx,color=ACC,lw=2.4,ls='--',label='sum of ReLUs')
    ax.set_ylim(target.min()-1.0,target.max()+0.9)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False,fontsize=11,loc='upper left'); save(fig,'relu_build')

# 9 — softmax 3-class linear regions
def f_softmax_regions():
    xx,yy=np.meshgrid(np.linspace(0,4,300),np.linspace(0,4,300))
    W=np.array([[1.0,0.2],[0.3,1.0],[-0.7,-0.6]]); bs=np.array([-1.5,-1.8,3.0])
    Z=np.stack([W[k,0]*xx+W[k,1]*yy+bs[k] for k in range(3)],-1); lab=Z.argmax(-1)
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(4.0,3.0))
    ax.contourf(xx,yy,lab,levels=[-.5,.5,1.5,2.5],colors=['#2C7A7B','#EB811B','#14B03D'],alpha=.22)
    ax.contour(xx,yy,lab,levels=[.5,1.5],colors=[INK],linewidths=1.4)
    for k,(cx,cy) in enumerate([(3.2,1.0),(1.0,3.2),(1.0,1.0)]): ax.text(cx,cy,f'class {k+1}',color=INK,fontsize=11,ha='center')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title('linear (polyhedral) regions',fontsize=12,pad=6); save(fig,'softmax_regions')

for f in [f_linreg,f_sigmoid,f_activations,f_mse_gd,f_linear_boundary,f_xor,f_feature_transform,f_relu_build,f_softmax_regions]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

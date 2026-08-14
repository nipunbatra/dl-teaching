"""Metropolis-style plot figures for Lecture 2 (Linear Models to MLPs).
Transparent background, ink + orange/teal/green accents. Emits SVG and PNG;
the newer slide figures also emit vector PDF for Typst. Run from repo root:
    python3 lecture2/diagrams/l2_figs.py
or regenerate selected figures by passing their names, for example:
    python3 lecture2/diagrams/l2_figs.py relu_2d_fold relu_multiclass_map
Architecture / graph diagrams are drawn natively in Typst (fletcher), not here."""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
import os, sys

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture2/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name,pdf=False):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    if pdf:
        fig.savefig(f'{OUT}/{name}.pdf',bbox_inches='tight',transparent=True)
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

# 10 — the same two-unit XOR construction on points versus filled quadrants
def f_xor_scope_contrast():
    n=501
    axis=np.linspace(0,1,n)
    xx,yy=np.meshgrid(axis,axis)
    truth=np.logical_xor(xx>.5,yy>.5).astype(int)
    h1=np.maximum(0,xx+yy)
    h2=np.maximum(0,xx+yy-1)
    logit=2*h1-4*h2-1
    pred=(logit>=0).astype(int)
    class_colors=[TEAL,ACC]
    cmap=ListedColormap(class_colors)

    fig,axes=plt.subplots(1,2,figsize=(9.7,3.45),gridspec_kw={'wspace':.25})
    # Left: the hand-chosen network is exact on the Boolean truth table.
    axes[0].contourf(xx,yy,pred,levels=[-.5,.5,1.5],cmap=cmap,alpha=.13)
    axes[0].plot([0,.5],[.5,0],color=INK,lw=1.7)
    axes[0].plot([.5,1],[1,.5],color=INK,lw=1.7)
    pts=np.array([[0,0],[1,0],[0,1],[1,1]])
    labels=np.array([0,1,1,0])
    for k,marker in ((0,'o'),(1,'s')):
        p=pts[labels==k]
        axes[0].scatter(p[:,0],p[:,1],s=84,marker=marker,color=class_colors[k],
                        edgecolor='white',linewidth=1.3,zorder=5)
    axes[0].text(.50,.43,'network predicts class 1\ninside this diagonal band',ha='center',va='center',
                 fontsize=11.5,color=INK,weight='semibold')
    axes[0].set_title('four Boolean inputs: all correct',fontsize=13.5,weight='semibold',pad=7)

    # Right: Playground-style XOR fills all four regions. Hatching exposes
    # precisely where this same diagonal-band rule disagrees with the target.
    axes[1].contourf(xx,yy,truth,levels=[-.5,.5,1.5],cmap=cmap,alpha=.16)
    mismatch=truth!=pred
    axes[1].contourf(xx,yy,mismatch,levels=[.5,1.5],colors='none',hatches=['////'])
    axes[1].plot([0,.5],[.5,0],color=INK,lw=1.7)
    axes[1].plot([.5,1],[1,.5],color=INK,lw=1.7)
    axes[1].axvline(.5,color=MUTED,lw=.75,ls='--',alpha=.75)
    axes[1].axhline(.5,color=MUTED,lw=.75,ls='--',alpha=.75)
    for px,py,txt,c in ((.12,.12,'class 0',TEAL),(.88,.12,'class 1',ACC),
                        (.12,.88,'class 1',ACC),(.88,.88,'class 0',TEAL)):
        axes[1].text(px,py,txt,ha='center',va='center',color=c,fontsize=11.5,weight='semibold',
                     bbox=dict(boxstyle='round,pad=.18',fc='white',ec='none',alpha=.84))
    axes[1].set_title('filled XOR regions: same rule misses areas',fontsize=13.5,weight='semibold',pad=7)
    axes[1].legend(handles=[Line2D([0],[0],color=INK,lw=1.7,label='same two-unit boundary'),
                            mpl.patches.Patch(facecolor='white',edgecolor=MUTED,hatch='////',label='misclassified area')],
                   frameon=False,fontsize=11,loc='lower center',bbox_to_anchor=(.5,-.29),ncol=2,
                   handlelength=2.0,columnspacing=1.2)
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(-.04,1.04); ax.set_ylim(-.04,1.04)
        ax.set_xlabel('x1',labelpad=1)
        ax.set_ylabel('x2',rotation=0,labelpad=12)
        ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
        ax.tick_params(labelsize=11,length=3)
    save(fig,'xor_scope_contrast',pdf=True)

# 11 — a 2-D input plane and its accurate 3-D ReLU fold
def f_relu_2d_fold():
    n=121
    x=np.linspace(-1,1,n); y=np.linspace(-1,1,n)
    xx,yy=np.meshgrid(x,y)
    u=xx+.65*yy-.35
    hh=np.maximum(0,u)
    relu_cmap=LinearSegmentedColormap.from_list('relu_teal',['#F7FAFA','#CFE5E3',TEAL])

    fig=plt.figure(figsize=(10.2,3.55))
    gs=fig.add_gridspec(1,2,width_ratios=[.92,1.22],wspace=.08)
    ax=fig.add_subplot(gs[0,0])
    ax.contourf(xx,yy,hh,levels=np.linspace(0,hh.max(),12),cmap=relu_cmap)
    hinge_y=(.35-x)/.65
    valid=(hinge_y>=-1)&(hinge_y<=1)
    ax.plot(x[valid],hinge_y[valid],color=ACC,lw=2.4)
    ax.text(-.64,-.50,'inactive\n$h=0$',color=MUTED,fontsize=11.5,ha='center',va='center')
    ax.text(.60,.44,'active\nlinear ramp',color=TEAL,fontsize=11.5,weight='semibold',ha='center',va='center')
    ax.annotate(r'hinge: $w^\top x+b=0$',xy=(.10,.385),xytext=(-.82,.82),fontsize=11.5,color=ACC,
                arrowprops=dict(arrowstyle='-|>',color=ACC,lw=1.0))
    ax.set_title('input plane: activation value',fontsize=13.5,weight='semibold',pad=7)
    ax.set_xlabel('x1',labelpad=1)
    ax.set_ylabel('x2',rotation=0,labelpad=12)
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.tick_params(labelsize=11.5)
    ax.set_aspect('equal')

    ax3=fig.add_subplot(gs[0,1],projection='3d')
    norm=mpl.colors.Normalize(vmin=0,vmax=hh.max())
    face=relu_cmap(norm(hh))
    ax3.plot_surface(xx,yy,hh,facecolors=face,rstride=3,cstride=3,
                     linewidth=.28,edgecolor=MUTED,antialiased=True,shade=False,alpha=.98)
    ax3.plot(x[valid],hinge_y[valid],np.zeros(valid.sum()),color=ACC,lw=3.0,zorder=10)
    ax3.set_title('output surface: flat plane folded into a ramp',fontsize=13.5,weight='semibold',pad=3)
    ax3.set_xlabel(''); ax3.set_ylabel(''); ax3.set_zlabel('')
    ax3.set_xticks([-1,0,1]); ax3.set_yticks([-1,0,1]); ax3.set_zticks([0,.5,1])
    ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
    ax3.text2D(.81,.10,'x1',transform=ax3.transAxes,fontsize=15,weight='semibold')
    ax3.text2D(.10,.16,'x2',transform=ax3.transAxes,fontsize=15,weight='semibold')
    ax3.text2D(.08,.55,'h',transform=ax3.transAxes,fontsize=15,weight='semibold')
    ax3.view_init(elev=28,azim=-132); ax3.set_proj_type('ortho')
    ax3.set_box_aspect((1.25,1,.72))
    for pane in (ax3.xaxis.pane,ax3.yaxis.pane,ax3.zaxis.pane):
        pane.set_facecolor((1,1,1,0)); pane.set_edgecolor((1,1,1,0))
    ax3.grid(False)
    save(fig,'relu_2d_fold',pdf=True)

# 12 — a fixed ReLU classifier in input space; every boundary is genuinely
# piecewise linear and can kink only when an activation pattern changes.
def f_relu_multiclass_map():
    axis=np.linspace(-1.8,1.8,601)
    xx,yy=np.meshgrid(axis,axis)
    flat=np.stack([xx,yy],axis=-1)
    W1=np.array([[1.0,.35],[-.45,1.0],[-1.0,-.20],[.25,-1.0]])
    b1=np.array([.10,-.12,.16,-.08])
    H=np.maximum(0,np.einsum('...d,md->...m',flat,W1)+b1)
    W2=np.array([[1.05,-.28,-.18,-.42],
                 [-.35,1.00,.22,-.24],
                 [-.28,-.32,.55,1.02]])
    b2=np.array([.03,.00,.04])
    Z=np.einsum('...m,km->...k',H,W2)+b2
    lab=Z.argmax(-1)
    colors=[BLUE,ACC,TEAL]
    cmap=ListedColormap(colors)

    fig,ax=plt.subplots(figsize=(5.7,4.25))
    ax.contourf(xx,yy,lab,levels=[-.5,.5,1.5,2.5],cmap=cmap,alpha=.22)
    ax.contour(xx,yy,lab,levels=[.5,1.5],colors=[INK],linewidths=2.0)
    # Hidden-unit hinges are context, not class boundaries.
    for j,(w,b) in enumerate(zip(W1,b1)):
        if abs(w[1])>.05:
            yline=-(w[0]*axis+b)/w[1]
            valid=(yline>=axis.min())&(yline<=axis.max())
            ax.plot(axis[valid],yline[valid],color=MUTED,lw=.8,ls='--',alpha=.62)
    # Stable labels placed well inside the three computed winner regions.
    label_specs=[((1.18,.80),'class 1',BLUE),((-.62,1.13),'class 2',ACC),((-.83,-1.08),'class 3',TEAL)]
    for (px,py),txt,c in label_specs:
        ax.text(px,py,txt,ha='center',va='center',fontsize=11.5,weight='semibold',color=c,
                bbox=dict(boxstyle='round,pad=.22',fc='white',ec='none',alpha=.82))
    ax.legend(handles=[Line2D([0],[0],color=INK,lw=2,label='visible class boundary'),
                       Line2D([0],[0],color=MUTED,lw=.9,ls='--',label='hidden-unit hinge')],
              frameon=False,fontsize=11.5,loc='lower center',bbox_to_anchor=(.5,-.23),ncol=2,
              handlelength=2.4,columnspacing=1.3)
    ax.set_xlabel('x1',labelpad=1)
    ax.set_ylabel('x2',rotation=0,labelpad=12)
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.tick_params(labelsize=10.5)
    ax.set_xlim(axis.min(),axis.max()); ax.set_ylim(axis.min(),axis.max()); ax.set_aspect('equal')
    ax.set_title('input space: largest of three ReLU-based scores',fontsize=13.5,weight='semibold',pad=7)
    save(fig,'relu_multiclass_map',pdf=True)

FIGURES={
    'linreg':f_linreg,
    'sigmoid':f_sigmoid,
    'activations':f_activations,
    'mse_gd':f_mse_gd,
    'linear_boundary':f_linear_boundary,
    'xor':f_xor,
    'feature_transform':f_feature_transform,
    'relu_build':f_relu_build,
    'softmax_regions':f_softmax_regions,
    'xor_scope_contrast':f_xor_scope_contrast,
    'relu_2d_fold':f_relu_2d_fold,
    'relu_multiclass_map':f_relu_multiclass_map,
}

if __name__ == '__main__':
    names=sys.argv[1:] or list(FIGURES)
    unknown=[name for name in names if name not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s): {', '.join(unknown)}; choose from {', '.join(FIGURES)}")
    for name in names:
        FIGURES[name](); print('ok',name)
    print('done ->',OUT)

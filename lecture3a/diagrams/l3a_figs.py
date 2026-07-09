"""Metropolis-style figures for Lecture 3A (Calculus Toolkit).
Transparent bg, ink + orange/teal accents. Emits SVG + PNG (dpi 200 -> Typst).
Run from repo root:  python3 lecture3a/diagrams/l3a_figs.py"""
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
OUT='lecture3a/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)
def bare3d(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.set_pane_color((1,1,1,0)); a.line.set_color(MUTED)
    ax.grid(False)

# 1 — tangent line / local linear approximation
def f_tangent():
    x=np.linspace(-1,4,300); f=lambda t:t**2; x0=2.0; dx=1.2
    fig,ax=plt.subplots(figsize=(4.4,3.0)); ax.plot(x,f(x),color=INK,lw=2.6,label='f(x)')
    ax.plot(x,f(x0)+2*x0*(x-x0),color=ACC,lw=2.2,label='tangent')
    ax.plot([x0,x0+dx],[f(x0),f(x0+dx)],'--',color=TEAL,lw=1.8,label='secant')
    ax.scatter([x0,x0+dx],[f(x0),f(x0+dx)],color=INK,zorder=3,s=30)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('x'); ax.legend(frameon=False,fontsize=11,loc='upper left'); save(fig,'tangent')

# 2 — 3D bowl z = x^2 + y^2
def f_bowl():
    X,Y=np.meshgrid(np.linspace(-2,2,60),np.linspace(-2,2,60)); Z=X**2+Y**2
    fig=plt.figure(figsize=(3.8,3.1)); ax=fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z,cmap=CMAP,alpha=.92,linewidth=0,antialiased=True,rstride=2,cstride=2)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(r'$z=x^2+y^2$',fontsize=13); bare3d(ax); ax.view_init(28,-52); save(fig,'bowl3d')

# 3 — surface with x-slice and y-slice highlighted
def f_slices():
    X,Y=np.meshgrid(np.linspace(-2,2,60),np.linspace(-2,2,60)); Z=X**2+3*Y**2
    fig=plt.figure(figsize=(4.0,3.1)); ax=fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z,cmap=CMAP,alpha=.35,linewidth=0,rstride=2,cstride=2)
    t=np.linspace(-2,2,80)
    ax.plot(t,np.full_like(t,-1.0),t**2+3*1.0,color=TEAL,lw=3)      # y = -1 slice (vary x)
    ax.plot(np.full_like(t,1.0),t,1.0+3*t**2,color=ACC,lw=3)        # x = 1 slice (vary y)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('slices = partial derivatives',fontsize=12); bare3d(ax); ax.view_init(26,-58); save(fig,'surface_slices')

# 4 — contours (circles) with gradient field perpendicular
def f_contours_grad():
    g=np.linspace(-2.2,2.2,220); X,Y=np.meshgrid(g,g); Z=X**2+Y**2
    fig,ax=plt.subplots(figsize=(3.5,3.2)); ax.contour(X,Y,Z,levels=8,colors=[TEAL],linewidths=.9,alpha=.8)
    xs=np.linspace(-1.7,1.7,7); Xa,Ya=np.meshgrid(xs,xs)
    ax.quiver(Xa,Ya,2*Xa,2*Ya,color=ACC,scale=40,width=.006,alpha=.9)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(r'$\nabla f \perp$ contours',fontsize=12); save(fig,'contours_grad')

# 5 — gradient descent on elongated contours (zigzag) vs circular
def f_gd_zigzag():
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
    cfgs=[dict(a=1,b=1,eta=0.20,p0=(2.1,1.7),ttl='well-conditioned',ylim=2.3),
          dict(a=1,b=9,eta=0.092,p0=(2.1,0.9),ttl='ill-conditioned',ylim=1.15)]
    for ax,c in zip(axes,cfgs):
        a,b,eta=c['a'],c['b'],c['eta']
        gx=np.linspace(-2.4,2.4,240); gy=np.linspace(-c['ylim'],c['ylim'],240); X,Y=np.meshgrid(gx,gy); Z=a*X**2+b*Y**2
        ax.contour(X,Y,Z,levels=10,colors=[TEAL],linewidths=.7,alpha=.75)
        p=np.array(c['p0'],float); pts=[p.copy()]
        for _ in range(26):
            p=p-eta*np.array([2*a*p[0],2*b*p[1]]); pts.append(p.copy())
        pts=np.array(pts); ax.plot(pts[:,0],pts[:,1],'-o',color=ACC,ms=3,lw=1.5)
        ax.scatter([0],[0],marker='*',s=150,color=RED,zorder=4)
        ax.set_xlim(-2.4,2.4); ax.set_ylim(-c['ylim'],c['ylim'])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(c['ttl'],fontsize=12)
    save(fig,'gd_zigzag')

# 6 — saddle surface x^2 - y^2 + contours
def f_saddle():
    X,Y=np.meshgrid(np.linspace(-2,2,60),np.linspace(-2,2,60)); Z=X**2-Y**2
    fig=plt.figure(figsize=(4.0,3.1)); ax=fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z,cmap=CMAP,alpha=.9,linewidth=0,rstride=2,cstride=2)
    ax.scatter([0],[0],[0],color=RED,s=40); ax.set_title(r'$f=x^2-y^2$  (saddle)',fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y'); bare3d(ax); ax.view_init(30,-60); save(fig,'saddle')

# 7 — 1D curvature: convex / concave / flat
def f_curv1d():
    x=np.linspace(-2,2,200); fig,axes=plt.subplots(1,3,figsize=(7.6,2.2))
    for ax,(y,ttl,c) in zip(axes,[(x**2,"f''>0  convex",TEAL),(-x**2,"f''<0  concave",ACC),(0.4*x,"f''=0  flat",MUTED)]):
        ax.plot(x,y,color=c,lw=2.6); ax.set_title(ttl,fontsize=12); ax.axhline(0,color=MUTED,lw=.6); ax.set_xticks([]); ax.set_yticks([])
    save(fig,'curvature_1d')

# 8 — Jacobian as local warp: grid -> warped grid, circle -> ellipse
def f_jacobian_warp():
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.3))
    def F(x,y): return x+0.5*np.sin(y), y+0.5*np.sin(x)
    g=np.linspace(-2.5,2.5,11)
    axes[0].set_title('input space',fontsize=12); axes[1].set_title('output f(x,y)',fontsize=12)
    for gi in g:
        yy=np.linspace(-2.5,2.5,60)
        axes[0].plot(np.full_like(yy,gi),yy,color=TEAL,lw=.7,alpha=.6); axes[0].plot(yy,np.full_like(yy,gi),color=TEAL,lw=.7,alpha=.6)
        u1,v1=F(np.full_like(yy,gi),yy); u2,v2=F(yy,np.full_like(yy,gi))
        axes[1].plot(u1,v1,color=TEAL,lw=.7,alpha=.6); axes[1].plot(u2,v2,color=TEAL,lw=.7,alpha=.6)
    # point + circle -> ellipse (via Jacobian)
    x0,y0=0.8,0.8; th=np.linspace(0,2*np.pi,80); r=0.4
    axes[0].plot(x0+r*np.cos(th),y0+r*np.sin(th),color=ACC,lw=2.2)
    J=np.array([[1,0.5*np.cos(y0)],[0.5*np.cos(x0),1]]); u0,v0=F(x0,y0)
    e=J@np.vstack([r*np.cos(th),r*np.sin(th)]); axes[1].plot(u0+e[0],v0+e[1],color=ACC,lw=2.2)
    for ax in axes: ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    save(fig,'jacobian_warp')

for f in [f_tangent,f_bowl,f_slices,f_contours_grad,f_gd_zigzag,f_saddle,f_curv1d,f_jacobian_warp]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

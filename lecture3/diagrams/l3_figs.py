"""Publication-style figures for Lecture 3 (Backpropagation & Autodiff).

Uses sustainability-lab/latexify for clean axes, presentation sizing, and
LaTeX-friendly typography, then applies the course palette. Emits SVG + PNG.

Run from the repo root:
  uv run --with matplotlib --with numpy \
    --with git+https://github.com/sustainability-lab/latexify.git \
    lecture3/diagrams/l3_figs.py
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
import latexify
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
latexify.latexify(largeFonts=True, fig_width=5.0, fig_height=3.1)
mpl.rcParams.update({
  'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','savefig.transparent':False,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.usetex':False,
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':0.8,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture3/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    latexify.save_fig(f'{OUT}/{name}.svg',fig=fig,facecolor='white')
    latexify.save_fig(f'{OUT}/{name}.png',fig=fig,facecolor='white',dpi=220)
    plt.close(fig)

# 1 — sigmoid and its derivative, with the 1/4 cap
def f_sigmoid_deriv():
    z=np.linspace(-8,8,400)
    s=1/(1+np.exp(-z)); ds=s*(1-s)
    fig,ax=plt.subplots(figsize=(5.0,3.1))
    ax.axvspan(-8,-4,color=BLUE,alpha=.06)
    ax.axvspan(4,8,color=BLUE,alpha=.06)
    ax.plot(z,s,color=INK,lw=2.5,label=r'$\sigma(z)$')
    ax.plot(z,ds,color=ACC,lw=2.7,label=r"$\sigma'(z)$")
    ax.axhline(0.25,ls='--',color=TEAL,lw=1.4)
    ax.annotate(r"peak $=1/4$",xy=(0,.25),xytext=(1.25,.39),color=TEAL,
                arrowprops=dict(arrowstyle='->',color=TEAL,lw=1.0))
    ax.text(-6,.07,'saturated',ha='center',color=BLUE,fontsize=10)
    ax.text(6,.07,'saturated',ha='center',color=BLUE,fontsize=10)
    ax.set_xlim(-8,8); ax.set_ylim(-.02,1.03)
    ax.set_xticks([-8,-4,0,4,8]); ax.set_yticks([0,.25,.5,1.0])
    ax.set_xlabel(r'pre-activation $z$'); ax.set_ylabel('value')
    latexify.format_axes(ax)
    fig.legend(loc='lower center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False)
    fig.subplots_adjust(bottom=.22)
    save(fig,'sigmoid_deriv')

# 2 — gradient magnitude vs depth for three repeated local slopes
def f_grad_flow():
    L=10; layers=np.arange(L+1)
    def chain(scale):
        return scale**layers
    fig,ax=plt.subplots(figsize=(5.2,3.1))
    ax.semilogy(layers,chain(.5),'-o',color=BLUE,ms=3.5,lw=2.2,label=r'local slope $0.5$')
    ax.semilogy(layers,chain(1.0),'-o',color=TEAL,ms=3.5,lw=2.2,label=r'local slope $1.0$')
    ax.semilogy(layers,chain(1.5),'-o',color=ACC,ms=3.5,lw=2.2,label=r'local slope $1.5$')
    ax.axhline(1,color=MUTED,lw=.8,ls='--')
    ax.set_xlim(0,L); ax.set_xticks([0,2,4,6,8,10])
    ax.set_xlabel('local derivatives multiplied'); ax.set_ylabel('relative gradient magnitude')
    latexify.format_axes(ax)
    fig.legend(loc='lower center',bbox_to_anchor=(.5,-.02),ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.24)
    save(fig,'grad_flow')

for f in [f_sigmoid_deriv,f_grad_flow]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

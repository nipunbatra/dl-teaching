"""Metropolis-style figures for Lecture 24 (Frontier Topics & Course Synthesis).
Schematic / illustrative trend plots ONLY — NO real images, NO fitted benchmark numbers.
Every STRUCTURAL diagram (course-arc map, agent loop, search loop, residual stream,
SAE, retrieval, systems thread) is NATIVE fletcher in the deck. These trend plots only:
  1. scaling_law  — illustrative power-law: loss vs compute (log-log), irreducible floor
  2. emergent     — same skill, two metrics: discontinuous (looks emergent) vs continuous (smooth)
  3. test_time    — accuracy vs number of samples: reliable vs flawed verifier vs no search
  4. double_desc  — test error vs capacity: classic double descent past interpolation
Transparent bg, ink + orange/teal palette. Emits SVG + PNG (dpi 200 -> Typst reads PNG twin).
Run from repo root:  python3 lecture24/diagrams/l24_figs.py
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
import os

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.6,'lines.solid_capstyle':'round',
})
OUT='lecture24/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── 1 — illustrative scaling law: loss vs compute, power-law region + irreducible floor ──
def f_scaling_law():
    C=np.logspace(0,8,300)          # arbitrary compute units
    E=1.8                            # irreducible error floor
    L=E + 30.0*C**(-0.32)           # illustrative power law (NOT fitted to any model)
    fig,ax=plt.subplots(figsize=(7.4,3.9))
    ax.plot(C,L,color=ACC)
    ax.axhline(E,color=MUTED,lw=1.6,ls='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('training compute  (log scale, arbitrary units)',fontsize=12)
    ax.set_ylabel('test loss  (log scale)',fontsize=12)
    ax.text(3e6,E*1.05,'irreducible error  $E$',fontsize=12,color=MUTED)
    ax.annotate('power-law region:\napprox. straight on log-log',xy=(3e2,L[np.argmin(abs(C-3e2))]),
                xytext=(2e3,9.5),fontsize=11,color=INK,
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.6))
    ax.set_title('illustrative scaling curve  —  schematic, not fitted to any model',fontsize=11.5)
    save(fig,'scaling_law')

# ── 2 — "emergence" as partly a metric artifact: one skill, two metrics ──
def f_emergent():
    N=np.logspace(0,4,300)                       # model scale (log)
    s=1/(1+np.exp(-(np.log10(N)-2.1)*2.4))       # smooth underlying per-token skill in [0,1]
    k=25                                          # a task needs ~k tokens all correct
    exact=s**k                                   # exact-match accuracy: sharp, looks "emergent"
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.8))
    ax=axes[0]
    ax.plot(N,exact,color=RED)
    ax.set_xscale('log'); ax.set_xlabel('model scale (log)',fontsize=12)
    ax.set_ylabel('exact-match accuracy',fontsize=12); ax.set_ylim(-0.03,1.03)
    ax.set_title('discontinuous metric\n"looks like sudden emergence"',fontsize=11.5,color=RED)
    ax=axes[1]
    ax.plot(N,s,color=TEAL)
    ax.set_xscale('log'); ax.set_xlabel('model scale (log)',fontsize=12)
    ax.set_ylabel('per-token skill',fontsize=12); ax.set_ylim(-0.03,1.03)
    ax.set_title('continuous metric\n"smooth, gradual improvement"',fontsize=11.5,color=TEAL)
    fig.suptitle('the SAME underlying model — the metric shapes the story  (hedge all emergence claims)',
                 fontsize=11.5,y=1.06)
    save(fig,'emergent')

# ── 3 — test-time compute: accuracy vs number of samples N ──
def f_test_time():
    N=np.arange(1,65)
    p=0.40                                   # single-sample success prob
    reliable=1-(1-p)**N                      # best-of-N, reliable verifier -> approaches 1
    ceiling=0.72
    flawed=ceiling*(1-(1-p)**N)              # flawed verifier caps the gain
    nosearch=np.full_like(N,p,dtype=float)   # one long sample, no search
    fig,ax=plt.subplots(figsize=(7.4,3.9))
    ax.plot(N,reliable,color=GREEN,label='best-of-$N$, reliable verifier')
    ax.plot(N,flawed,color=ACC,label='best-of-$N$, flawed verifier (caps out)')
    ax.plot(N,nosearch,color=MUTED,ls='--',label='single sample, no search')
    ax.axhline(ceiling,color=ACC,lw=1.0,ls=':',alpha=0.7)
    ax.set_xscale('log',base=2)
    ax.set_xlabel('inference-time samples  $N$  (log$_2$)',fontsize=12)
    ax.set_ylabel('task success',fontsize=12); ax.set_ylim(0,1.03)
    ax.annotate('more compute\n= higher cost / latency',xy=(48,reliable[47]),xytext=(6,0.45),
                fontsize=10.5,color=INK,arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.4))
    ax.legend(frameon=False,fontsize=10.5,loc='lower right')
    ax.set_title('spending compute at inference time  (illustrative)',fontsize=11.5)
    save(fig,'test_time')

# ── 4 — double descent: test error vs capacity, peak at the interpolation threshold ──
def f_double_desc():
    x=np.linspace(0.05,3.0,400)                       # capacity = params / data
    thr=1.0
    classical=0.40+0.35*(x-0.55)**2                   # shallow classical U (bounded)
    modern=0.33+0.12/x                                # over-parameterized second descent
    spike=0.85*np.exp(-((x-thr)**2)/(2*0.05**2))      # variance blow-up at interpolation
    test=np.where(x<=thr, classical, modern)+spike
    train=np.clip(0.6*(thr-x),0,None)                 # train error -> 0 at interpolation
    fig,ax=plt.subplots(figsize=(7.4,3.9))
    ax.plot(x,test,color=ACC,label='test error')
    ax.plot(x,train,color=TEAL,ls='--',label='train error')
    ax.axvline(thr,color=MUTED,lw=1.4,ls=':')
    ax.text(thr+0.04,1.28,'interpolation\nthreshold',fontsize=10.5,color=MUTED)
    ax.set_xlabel('model capacity  (params / data)',fontsize=12)
    ax.set_ylabel('error',fontsize=12); ax.set_ylim(0,1.5); ax.set_xlim(0,3)
    ax.legend(frameon=False,fontsize=11,loc='center right')
    ax.set_title('double descent  —  classical U, then a second descent (illustrative)',fontsize=11.5)
    save(fig,'double_desc')

for f in [f_scaling_law,f_emergent,f_test_time,f_double_desc]:
    f(); print('ok',f.__name__)
print('done ->',OUT)

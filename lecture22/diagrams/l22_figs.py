"""Metropolis-style figures for Lecture 22 (Diffusion Models — Practice).
Schematic / synthetic ONLY — NO real images. Transparent bg, ink + orange/teal palette.
Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).

Every STRUCTURAL diagram (system map, latent-diffusion end-to-end, U-Net, cross-attention,
CFG flow, DDPM vs DDIM sampler, full Stable-Diffusion pipeline) is NATIVE fletcher in the deck.
These quantitative / schematic plots only:
  1. latent_compression — pixel tensor (786,432 values) vs latent tensor (16,384): ~48x fewer
  2. cfg_vectors        — classifier-free guidance as vector extrapolation in 2-D
  3. cfg_scale          — guidance scale w: prompt adherence up, diversity/naturalness down
  4. sampler_steps      — number of sampling steps vs quality: DDPM vs DDIM
  5. noise_schedule     — signal level sqrt(alpha-bar_t) vs t for linear vs cosine schedules
Run from repo root:  python3 lecture22/diagrams/l22_figs.py
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
OUT='lecture22/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── 1 — why latents: 512x512x3 pixels vs 64x64x4 latent, ~48x fewer values ──
def f_latent_compression():
    pix=512*512*3          # 786,432
    lat=64*64*4            # 16,384
    fig,ax=plt.subplots(figsize=(7.0,3.4))
    bars=ax.bar([0,1],[pix,lat],width=0.55,color=[RED,TEAL],edgecolor='white',linewidth=1.4)
    ax.set_yscale('log'); ax.set_ylim(5e3,2e6)
    ax.set_xticks([0,1]); ax.set_xticklabels(['pixel tensor\n$512\\times512\\times3$','latent tensor\n$64\\times64\\times4$'],fontsize=12)
    ax.set_ylabel('values to denoise (log)',fontsize=12)
    for b,v,c in zip(bars,[pix,lat],[RED,TEAL]):
        ax.text(b.get_x()+b.get_width()/2,v*1.3,f'{v:,}',ha='center',fontsize=12,color=c,weight=700)
    ax.annotate('',xy=(1,lat*1.9),xytext=(0,pix*0.9),
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.2,connectionstyle='arc3,rad=-0.25'))
    ax.text(0.5,3.0e5,r'$\approx 48\times$ fewer',ha='center',fontsize=14,color=INK,weight=700)
    ax.set_title('diffuse in a compact latent, not in raw pixels',fontsize=12.5)
    save(fig,'latent_compression')

# ── 2 — classifier-free guidance is vector extrapolation (landscape) ──
def f_cfg_vectors():
    e0=np.array([2.0,0.5])          # unconditional prediction  eps_uncond
    ec=np.array([3.0,1.3])          # conditional prediction     eps_cond
    w=2.2
    ecfg=e0+w*(ec-e0)               # guided prediction
    fig,ax=plt.subplots(figsize=(7.4,3.5))
    def vec(a,b,c,lw=2.8):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=c,lw=lw))
    vec((0,0),e0,MUTED)
    vec((0,0),ec,BLUE)
    vec((0,0),ecfg,ACC,lw=3.4)
    # the extrapolation leg: from eps_cond onward, dashed
    ax.annotate('',xy=ecfg,xytext=ec,arrowprops=dict(arrowstyle='-|>',color=ACC,lw=2.0,ls='--'))
    ax.plot(*e0,'o',color=MUTED,ms=6); ax.plot(*ec,'o',color=BLUE,ms=6); ax.plot(*ecfg,'o',color=ACC,ms=7)
    ax.text(e0[0]+0.05,e0[1]-0.36,r'$\epsilon_\varnothing$ (uncond.)',color=MUTED,fontsize=12,weight=600)
    ax.text(ec[0]-0.02,ec[1]-0.40,r'$\epsilon_c$ (cond.)',color=BLUE,fontsize=12,weight=600,ha='center')
    ax.text(ecfg[0]-0.05,ecfg[1]+0.16,r'$\hat\epsilon=\epsilon_\varnothing+w(\epsilon_c-\epsilon_\varnothing)$',
            color=ACC,fontsize=12,weight=700,ha='center')
    ax.text((ec[0]+ecfg[0])/2+0.30,(ec[1]+ecfg[1])/2-0.02,'extrapolate\nbeyond '+r'$\epsilon_c$',color=ACC,fontsize=10.5)
    ax.set_xlim(-0.3,5.5); ax.set_ylim(-0.5,2.95); ax.set_aspect('equal')
    ax.axhline(0,color=MUTED,lw=0.8); ax.axvline(0,color=MUTED,lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r'$w=2.2$:  push past the conditional direction',fontsize=12,pad=6)
    save(fig,'cfg_vectors')

# ── 3 — guidance scale trade-off: adherence up, diversity/naturalness down ──
def f_cfg_scale():
    w=np.linspace(1,16,200)
    adher=1-np.exp(-(w-1)/3.0)                 # prompt adherence, saturating
    divers=np.exp(-(w-1)/6.0)                   # diversity / naturalness, decaying
    fig,ax=plt.subplots(figsize=(7.0,3.6))
    ax.plot(w,adher,color=ACC,label='prompt adherence')
    ax.plot(w,divers,color=TEAL,label='diversity / naturalness')
    ax.axvspan(6,10,color=GREEN,alpha=0.10)
    ax.text(8,0.5,'typical\nsweet spot',ha='center',color=GREEN,fontsize=11,weight=600)
    ax.set_xlabel('guidance scale  $w$',fontsize=12); ax.set_ylabel('relative (schematic)',fontsize=12)
    ax.set_ylim(0,1.05); ax.set_xlim(1,16)
    ax.legend(frameon=False,fontsize=11,loc='center right')
    ax.set_title('higher $w$ trades diversity for adherence',fontsize=12.5)
    save(fig,'cfg_scale')

# ── 4 — sampling steps vs quality: DDPM needs many, DDIM good with few ──
def f_sampler_steps():
    steps=np.array([5,10,20,50,100,250,1000])
    q_ddim=np.array([0.55,0.74,0.86,0.94,0.96,0.97,0.975])
    q_ddpm=np.array([0.12,0.22,0.40,0.68,0.83,0.93,0.975])
    fig,ax=plt.subplots(figsize=(7.0,3.6))
    ax.plot(steps,q_ddim,'-o',color=ACC,ms=6,label='DDIM (deterministic, few steps)')
    ax.plot(steps,q_ddpm,'-o',color=BLUE,ms=6,label='DDPM (stochastic, many steps)')
    ax.set_xscale('log')
    ax.set_xlabel('number of sampling steps (denoiser calls)',fontsize=12)
    ax.set_ylabel('sample quality (schematic)',fontsize=12)
    ax.set_ylim(0,1.02)
    ax.axvline(30,color=MUTED,lw=1.0,ls='--')
    ax.text(27,0.30,'few steps:\nDDIM already good',color=MUTED,fontsize=10,ha='right')
    ax.legend(frameon=False,fontsize=10.5,loc='lower right')
    ax.set_title('same denoiser, fewer scheduled steps = lower latency',fontsize=12)
    save(fig,'sampler_steps')

# ── 5 — noise schedules: signal level sqrt(alpha-bar_t) for linear vs cosine ──
def f_noise_schedule():
    T=1000; t=np.arange(T+1)
    beta=np.linspace(1e-4,0.02,T)
    abar_lin=np.concatenate([[1.0],np.cumprod(1-beta)])
    s=0.008
    f=np.cos(((t/T)+s)/(1+s)*np.pi/2)**2
    abar_cos=f/f[0]
    fig,ax=plt.subplots(figsize=(7.0,3.6))
    ax.plot(t,np.sqrt(abar_lin),color=BLUE,label='linear schedule')
    ax.plot(t,np.sqrt(abar_cos),color=ACC,label='cosine schedule')
    ax.set_xlabel('timestep  $t$   (0 = clean, T = pure noise)',fontsize=12)
    ax.set_ylabel(r'signal retained  $\sqrt{\bar\alpha_t}$',fontsize=12)
    ax.set_ylim(0,1.02); ax.set_xlim(0,T)
    ax.fill_between(t,0,np.sqrt(abar_cos),color=ACC,alpha=0.06)
    ax.legend(frameon=False,fontsize=11,loc='upper right')
    ax.set_title('cosine keeps signal longer at high $t$',fontsize=12.5)
    save(fig,'noise_schedule')

for fn in [f_latent_compression,f_cfg_vectors,f_cfg_scale,f_sampler_steps,f_noise_schedule]:
    fn(); print('ok',fn.__name__)
print('done ->',OUT)

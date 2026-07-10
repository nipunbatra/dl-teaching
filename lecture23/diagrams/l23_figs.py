r"""Metropolis-style figures for Lecture 23 (Efficient Inference).
Schematic / quantitative plots ONLY — NO real images. All STRUCTURAL diagrams
(KV-cache prefill/decode, quantization flow, MoE routing, speculative decoding,
memory-vs-compute) are NATIVE fletcher in the deck. These matplotlib figures only:
  1. roofline      — arithmetic-intensity roofline: decode is memory-bound, prefill compute-bound
  2. latency_batch — per-token latency (flat, then rising) + throughput (rising, saturating) vs batch
  3. quant_bits    — bits vs memory (bars) and quality retention (line): fp32/fp16/int8/int4
  4. kv_growth     — KV-cache memory vs context length T, for batch B = 1, 8, 32 (log-log)
Transparent bg, ink + orange/teal palette. Emits SVG + PNG (dpi 200 -> Typst reads the PNG twin).
Run from repo root:  python3 lecture23/diagrams/l23_figs.py
matplotlib note: NO LaTeX \le — use the unicode char if needed.
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
OUT='lecture23/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.svg',bbox_inches='tight',transparent=True)
    fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight',transparent=True,dpi=200)
    plt.close(fig)

# ── 1 — roofline: attainable throughput vs arithmetic intensity ──────────────
# memory-bound diagonal (y = BW * AI) meets the compute ceiling (y = PEAK) at the ridge.
# decode (AI ~ 1 FLOP/byte) sits deep in the memory-bound region; prefill (AI ~ 500) is compute-bound.
def f_roofline():
    PEAK = 312.0            # TFLOP/s  (fp16 tensor, A100-class)
    BW   = 2.0              # TB/s     (~2000 GB/s HBM)  -> FLOP/byte units line up in TFLOP/s
    ridge = PEAK / BW       # = 156 FLOP/byte
    ai = np.logspace(-0.3, 3.4, 400)          # 0.5 .. ~2500 FLOP/byte
    attain = np.minimum(PEAK, BW*ai)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    # shade the memory-bound region (left of ridge)
    ax.axvspan(ai.min(), ridge, color=RED, alpha=0.06)
    ax.axvspan(ridge, ai.max(), color=GREEN, alpha=0.06)
    # roofline
    ax.plot(ai, attain, color=INK, lw=3.0, zorder=5)
    ax.plot(ai[ai<=ridge], (BW*ai)[ai<=ridge], color=RED, lw=3.0, zorder=6)
    ax.axhline(PEAK, color=MUTED, lw=1.2, ls=':')
    ax.text(ai.max()*0.82, PEAK*1.12, f'compute ceiling  {PEAK:.0f} TFLOP/s',
            ha='right', fontsize=10.5, color=MUTED)
    # ridge marker
    ax.axvline(ridge, color=MUTED, lw=1.1, ls='--')
    ax.text(ridge*1.12, 3.0, f'ridge  {ridge:.0f}', fontsize=10.5, color=MUTED, rotation=0)
    # operating points
    ax.scatter([1.0], [BW*1.0], s=150, color=RED, zorder=8, edgecolor='white', lw=1.5)
    ax.annotate('decode\n(matrix–vector, AI ≈ 1)', xy=(1.0, BW*1.0), xytext=(1.5, 22),
                fontsize=11, color=RED, weight=600, ha='left',
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.8))
    ax.scatter([520.0], [PEAK], s=150, color=GREEN, zorder=8, edgecolor='white', lw=1.5)
    ax.annotate('prefill\n(matrix–matrix, AI ≈ T)', xy=(520.0, PEAK), xytext=(60, 60),
                fontsize=11, color=GREEN, weight=600, ha='left',
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.8))
    ax.text(2.2, 130, 'memory-bound', fontsize=12, color=RED, weight=700, rotation=34, alpha=0.85)
    ax.text(320, 150, 'compute-bound', fontsize=12, color=GREEN, weight=700, alpha=0.85)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(ai.min(), ai.max()); ax.set_ylim(0.7, 700)
    ax.set_xlabel('arithmetic intensity  (FLOP / byte)', fontsize=12)
    ax.set_ylabel('attainable throughput  (TFLOP/s)', fontsize=12)
    ax.set_title('one decode step does few FLOPs per weight byte fetched', fontsize=12, color=INK)
    save(fig, 'roofline')

# ── 2 — per-token latency + throughput vs batch size ─────────────────────────
# decode fetches the weights once per step regardless of batch (until compute-bound),
# so per-token latency stays ~flat for small batch while throughput scales ~linearly.
def f_latency_batch():
    B = np.array([1,2,4,8,16,32,64,128])
    t_weight = 20.0          # ms to stream weights once (fixed cost per step)
    t_compute = 0.28         # ms of compute added per sequence in the batch
    step_ms = t_weight + t_compute*B      # per-step latency: flat, then compute grows
    lat_per_tok = step_ms / 1.0           # one token per sequence per step -> per-token = step/1 ...
    # per-token latency the USER sees ~ step latency (each seq still emits 1 tok/step)
    thr = B / (step_ms/1000.0)            # tokens per second across the batch
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(B, step_ms, '-o', color=RED, lw=2.8, ms=7, label='per-step latency (ms)')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('batch size  (concurrent sequences)', fontsize=12)
    ax.set_ylabel('per-step latency  (ms)', color=RED, fontsize=12)
    ax.tick_params(axis='y', labelcolor=RED)
    ax.set_ylim(0, 70)
    ax.axvspan(1, 8, color=RED, alpha=0.06)
    ax.text(2.6, 62, 'weights dominate\n(memory-bound)', fontsize=10.5, color=RED, ha='center')
    ax.text(48, 62, 'compute\nstarts to bite', fontsize=10.5, color=GREEN, ha='center')
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    ax2.plot(B, thr, '-s', color=TEAL, lw=2.8, ms=7, label='throughput (tok/s)')
    ax2.set_ylabel('throughput  (tokens / s)', color=TEAL, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=TEAL)
    ax2.set_ylim(0, 3500)
    ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
    ax.set_title('batching: ~free tokens until the step turns compute-bound', fontsize=12, color=INK)
    save(fig, 'latency_batch')

# ── 3 — quantization: bits vs memory (bars) and quality retention (line) ──────
def f_quant_bits():
    fmts = ['fp32','fp16','int8','int4']
    bytes_pp = np.array([4,2,1,0.5])                 # bytes per parameter
    mem7b = bytes_pp * 7                              # GB for a 7B model
    quality = np.array([100.0, 100.0, 99.1, 96.5])   # illustrative % of fp16 quality retained
    x = np.arange(len(fmts))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(x, mem7b, width=0.56, color=[MUTED,BLUE,TEAL,ACC], edgecolor='white', lw=1.4, zorder=3)
    for b,v in zip(bars, mem7b):
        ax.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.0f} GB', ha='center', fontsize=11, color=INK, weight=600)
    ax.set_ylabel('weight memory for a 7B model  (GB)', fontsize=12)
    ax.set_ylim(0, 32)
    ax.set_xticks(x); ax.set_xticklabels(fmts, fontsize=13)
    ax.set_xlabel('storage format', fontsize=12)
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    ax2.plot(x, quality, '-o', color=RED, lw=2.8, ms=9, zorder=5)
    for xi,q in zip(x,quality):
        ax2.text(xi, q-1.4, f'{q:.1f}%', ha='center', fontsize=10.5, color=RED, weight=600)
    ax2.set_ylabel('quality retained vs fp16  (%)', color=RED, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax2.set_ylim(90, 101.5)
    ax.set_title('fewer bits: linear memory savings, gently falling quality', fontsize=12, color=INK)
    save(fig, 'quant_bits')

# ── 4 — KV-cache memory vs context length T, for B = 1, 8, 32 ────────────────
def f_kv_growth():
    L,n_kv,d_h,b = 32,8,128,2
    T = np.array([512,1024,2048,4096,8192,16384,32768])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for B,c,mk in [(1,TEAL,'o'),(8,BLUE,'s'),(32,ACC,'^')]:
        M = 2*L*B*T*n_kv*d_h*b / 1024**3           # GiB
        ax.plot(T, M, '-'+mk, color=c, lw=2.8, ms=7, label=f'batch B = {B}')
    # mark the board-calculation point: B=1, T=4096 -> 512 MiB = 0.5 GiB
    ax.scatter([4096],[0.5], s=170, facecolor='none', edgecolor=INK, lw=2.4, zorder=8)
    ax.annotate('board calc:\nB=1, T=4096 → 512 MiB', xy=(4096,0.5), xytext=(600,4),
                fontsize=10.5, color=INK, weight=600,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6))
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks(T); ax.set_xticklabels([f'{t//1024}k' if t>=1024 else str(t) for t in T], fontsize=10.5)
    ax.set_xlabel('context length  T  (tokens)', fontsize=12)
    ax.set_ylabel('KV-cache memory  (GiB)', fontsize=12)
    ax.set_ylim(0.02, 200)
    ax.axhspan(80, 200, color=RED, alpha=0.07)
    ax.text(700, 110, '80 GB device budget', fontsize=10.5, color=RED)
    ax.axhline(80, color=RED, lw=1.1, ls='--')
    ax.legend(frameon=False, fontsize=11, loc='lower right')
    ax.set_title(r'KV cache grows linearly in T and in batch — quadratic pressure on serving',
                 fontsize=11.5, color=INK)
    save(fig, 'kv_growth')

for f in [f_roofline, f_latency_batch, f_quant_bits, f_kv_growth]:
    f(); print('ok', f.__name__)
print('done ->', OUT)

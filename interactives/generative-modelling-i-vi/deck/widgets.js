// Small live widgets. Each has a meaningful static state for the PDF.
(() => {
  const log = Math.log;
  const f3 = v => (Math.abs(v) < 5e-4 ? '0.000' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(3));

  // KL between P = (0.8, 0.2) and Q = (q, 1 - q), both directions
  function kl(root) {
    const input = root.querySelector('input');
    const out = root.querySelector('.readout');
    const pq = root.querySelector('.kl-pq');
    const qp = root.querySelector('.kl-qp');
    const P = [0.8, 0.2];
    const update = () => {
      const q = Number(input.value), Q = [q, 1 - q];
      const a = P[0] * log(P[0] / Q[0]) + P[1] * log(P[1] / Q[1]);
      const b = Q[0] * log(Q[0] / P[0]) + Q[1] * log(Q[1] / P[1]);
      out.textContent = q.toFixed(2);
      pq.textContent = a.toFixed(3);
      qp.textContent = b.toFixed(3);
    };
    input.addEventListener('input', update);
    update();
  }

  // Neat/messy writer: ELBO + KL(q || posterior) = log p(x) for every q
  function elbo(root) {
    const slide = root.closest('section');
    const input = root.querySelector('input');
    const out = root.querySelector('.q0');
    const set = (cls, v) => { const el = slide.querySelector(cls); if (el) el.textContent = f3(v); };
    const update = () => {
      const q0 = Number(input.value), q = [q0, 1 - q0];
      const lik = [0.8, 0.2], prior = [0.5, 0.5], post = [0.8, 0.2];
      const ell = q[0] * log(lik[0]) + q[1] * log(lik[1]);
      const klp = q[0] * log(q[0] / prior[0]) + q[1] * log(q[1] / prior[1]);
      const gap = q[0] * log(q[0] / post[0]) + q[1] * log(q[1] / post[1]);
      out.textContent = q0.toFixed(2);
      set('.v-ell', ell); set('.v-klp', -klp); set('.v-elbo', ell - klp); set('.v-gap', gap); set('.v-lpx', ell - klp + gap);
    };
    input.addEventListener('input', update);
    update();
  }

  // Latent map magnifier: hover a cell of the decoded grid to enlarge it
  function explorer(root) {
    const img = root.querySelector('img');
    const lens = document.createElement('div');
    lens.className = 'lens';
    root.appendChild(lens);
    const n = 18;
    root.addEventListener('mousemove', e => {
      const r = img.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
      if (fx < 0 || fy < 0 || fx > 1 || fy > 1) { lens.style.display = 'none'; return; }
      const cx = Math.min(n - 1, Math.floor(fx * n)), cy = Math.min(n - 1, Math.floor(fy * n));
      lens.style.display = 'block';
      lens.style.backgroundImage = `url(${img.getAttribute('src')})`;
      lens.style.backgroundSize = `${n * 100}% ${n * 100}%`;
      lens.style.backgroundPosition = `${cx / (n - 1) * 100}% ${cy / (n - 1) * 100}%`;
    });
    root.addEventListener('mouseleave', () => { lens.style.display = 'none'; });
  }


  // Latent-space app: decode any 2-D code in the browser with the trained decoder's own weights.
  const DIGIT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948', '#7c5a3a', '#5f6b7a'];
  async function latent(root) {
    const name = root.dataset.model;
    const base = 'figures/data/explorer-' + name;
    let meta, W, sprite;
    try {
      [meta, W] = await Promise.all([
        fetch(base + '.json').then(r => r.json()),
        fetch(base + '.bin').then(r => r.arrayBuffer()).then(b => new Float32Array(b)),
      ]);
      sprite = new Image();
      sprite.src = 'figures/data/explorer-sprite.png';
      await sprite.decode();
    } catch { root.classList.add('lx-offline'); return; }
    const [H1, H2, OUT] = meta.sizes;
    const o1 = 0, ob1 = H1 * 2, o2 = ob1 + H1, ob2 = o2 + H2 * H1, o3 = ob2 + H2, ob3 = o3 + OUT * H2;
    const h1 = new Float32Array(H1), h2 = new Float32Array(H2);
    function decode(z1, z2) {
      for (let i = 0; i < H1; i++) h1[i] = Math.max(0, W[o1 + 2 * i] * z1 + W[o1 + 2 * i + 1] * z2 + W[ob1 + i]);
      for (let i = 0; i < H2; i++) { let a = W[ob2 + i]; const r = o2 + i * H1; for (let j = 0; j < H1; j++) a += W[r + j] * h1[j]; h2[i] = Math.max(0, a); }
      const px = new Float32Array(OUT);
      for (let i = 0; i < OUT; i++) { let a = W[ob3 + i]; const r = o3 + i * H2; for (let j = 0; j < H2; j++) a += W[r + j] * h2[j]; px[i] = 1 / (1 + Math.exp(-a)); }
      return px;
    }
    const plot = root.querySelector('.lx-plot'), g = plot.getContext('2d');
    const out = root.querySelector('.lx-out').getContext('2d');
    const orig = root.querySelector('.lx-orig').getContext('2d');
    const origFig = root.querySelector('.lx-origfig');
    const tip = root.querySelector('.lx-tip'), tipCtx = tip.querySelector('canvas').getContext('2d'), tipLab = tip.querySelector('span');
    const s1 = root.querySelector('.lx-s1'), s2 = root.querySelector('.lx-s2');
    const r1 = root.querySelector('.lx-z1'), r2 = root.querySelector('.lx-z2');
    const [[x0, x1], [y0, y1]] = meta.range;
    const S = plot.width, M = 90;
    const X = v => M + (v - x0) / (x1 - x0) * (S - 2 * M), Y = v => S - M - (v - y0) / (y1 - y0) * (S - 2 * M);
    const iX = px => x0 + (px - M) / (S - 2 * M) * (x1 - x0), iY = py => y0 + (S - M - py) / (S - 2 * M) * (y1 - y0);
    for (const [sl, lo, hi] of [[s1, x0, x1], [s2, y0, y1]]) { sl.min = lo; sl.max = hi; sl.step = (hi - lo) / 500; }
    let z = [0, 0], picked = -1;

    function spriteTo(ctx, k) {
      ctx.clearRect(0, 0, 28, 28);
      ctx.drawImage(sprite, (k % meta.cols) * 28, Math.floor(k / meta.cols) * 28, 28, 28, 0, 0, 28, 28);
    }
    function draw() {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      g.strokeStyle = '#D9DDE5'; g.lineWidth = 2;
      g.beginPath(); g.moveTo(M, Y(0)); g.lineTo(S - M, Y(0)); g.moveTo(X(0), M); g.lineTo(X(0), S - M); g.stroke();
      g.fillStyle = '#6B7280'; g.font = '40px Avenir Next, sans-serif'; g.textAlign = 'center';
      for (const t of [x0, (x0 + x1) / 2, x1]) g.fillText(t.toFixed(0), X(t), S - M + 50);
      g.textAlign = 'right';
      for (const t of [y0, (y0 + y1) / 2, y1]) g.fillText(t.toFixed(0), M - 10, Y(t) + 8);
      meta.points.forEach(([a, b, c]) => { g.fillStyle = DIGIT[c] + 'AA'; g.beginPath(); g.arc(X(a), Y(b), 4.5, 0, 7); g.fill(); });
      g.setLineDash([10, 8]); g.strokeStyle = '#AA4E08'; g.lineWidth = 2.5;
      for (const r of [1, 2]) { g.beginPath(); g.ellipse(X(0), Y(0), Math.abs(X(r) - X(0)), Math.abs(Y(r) - Y(0)), 0, 0, 7); g.stroke(); }
      g.setLineDash([]);
      if (picked >= 0) { const [a, b] = meta.points[picked]; g.strokeStyle = '#14171F'; g.lineWidth = 3; g.beginPath(); g.arc(X(a), Y(b), 13, 0, 7); g.stroke(); }
      g.fillStyle = '#AA4E08'; g.strokeStyle = '#fff'; g.lineWidth = 4;
      g.beginPath(); g.arc(X(z[0]), Y(z[1]), 13, 0, 7); g.fill(); g.stroke();
    }
    function render() {
      const px = decode(z[0], z[1]);
      const im = out.createImageData(28, 28);
      px.forEach((v, i) => { const c = Math.round(255 * (1 - v)); im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = c; im.data[4 * i + 3] = 255; });
      out.putImageData(im, 0, 0);
      s1.value = z[0]; s2.value = z[1];
      r1.textContent = z[0].toFixed(2); r2.textContent = z[1].toFixed(2);
      origFig.style.visibility = picked >= 0 ? 'visible' : 'hidden';
      if (picked >= 0) spriteTo(orig, picked);
      draw();
    }
    const setZ = (a, b, k = -1) => { z = [a, b]; picked = k; render(); };
    const toPlot = e => { const r = plot.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * S, (e.clientY - r.top) / r.height * S, r]; };
    plot.addEventListener('click', e => { const [px, py] = toPlot(e); setZ(iX(px), iY(py)); });
    plot.addEventListener('mousemove', e => {
      const [px, py, r] = toPlot(e);
      let best = -1, bd = 14 * 14;
      meta.points.forEach(([a, b], k) => { const d = (X(a) - px) ** 2 + (Y(b) - py) ** 2; if (d < bd) { bd = d; best = k; } });
      if (best < 0) { tip.style.display = 'none'; return; }
      spriteTo(tipCtx, best);
      tipLab.textContent = 'a ' + meta.points[best][2];
      tip.style.display = 'flex';
      tip.style.left = (px / S * 100) + '%';
      tip.style.top = (py / S * 100) + '%';
    });
    plot.addEventListener('mouseleave', () => (tip.style.display = 'none'));
    s1.addEventListener('input', () => setZ(+s1.value, z[1]));
    s2.addEventListener('input', () => setZ(z[0], +s2.value));
    const randn = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    root.querySelector('[data-a="prior"]').addEventListener('click', () => setZ(randn(), randn()));
    root.querySelector('[data-a="real"]').addEventListener('click', () => {
      const k = Math.floor(Math.random() * meta.points.length);
      setZ(meta.points[k][0], meta.points[k][1], k);
    });
    setZ(0, 0);
  }

  // MNIST as a Gaussian mixture: pick the PCA size d and the number of bumps K, see real samples from that fit
  async function gmmx(root) {
    let info;
    try { info = await (await fetch('figures/data/gmm-explore/index.json')).json(); } catch { return; }
    const [dIn, kIn] = [root.querySelector('.gx-d'), root.querySelector('.gx-k')];
    const $ = c => root.querySelector(c);
    const num = v => Number(v).toLocaleString('en-US').replace('-', '−');
    const update = () => {
      const d = info.d_grid[+dIn.value], k = info.k_grid[+kIn.value];
      $('.gx-dv').textContent = d;
      $('.gx-kv').textContent = k;
      $('.gx-recon').src = `figures/data/gmm-explore/recon_d${d}.png`;
      $('.gx-samples').src = `figures/data/gmm-explore/d${d}_k${k}.png`;
      $('.gx-var').textContent = `${info.var_kept[d]}%`;
      $('.gx-params').textContent = num(info.params[d][k]);
      $('.gx-test').textContent = Number(info.test[d][k]).toFixed(1).replace('-', '−');
      $('.gx-best').textContent = info.best_k[d];
    };
    dIn.addEventListener('input', update);
    kIn.addEventListener('input', update);
    update();
  }


  // ======================================================================== the VI lecture (vi.html)
  // Every widget below draws a meaningful state on load, so ?print shows something useful.
  // The quantities match scripts/make_figures_vi.py: same grids, same formulas.
  const PAL = { x: '#245EDB', xt: '#E4ECFF', enc: '#8B2CDE', enct: '#F1E5FC', dec: '#0F766E', dect: '#D9F2EF',
    z: '#AA4E08', zt: '#FCEFD9', loss: '#BE123C', losst: '#FDE2E7', ink: '#14171F', ink2: '#4A5160', ink3: '#6B7280',
    line: '#D9DDE5', grid: '#ECEEF2', grey: '#C7CCD4' };
  const PRINT = new URLSearchParams(location.search).has('print');
  const FONT = '13px "Avenir Next", Avenir, "Segoe UI", sans-serif';
  const fx = (v, k = 3) => (Math.abs(v) < 0.5 * 10 ** -k ? 0 : v).toFixed(k).replace('-', '−');
  const SQ2PI = Math.sqrt(2 * Math.PI);
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQ2PI);
  const sig = a => 1 / (1 + Math.exp(-a));
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

  function rng(seed) {                      // a seeded uniform (mulberry32) and standard normal (Box-Muller)
    let a = seed >>> 0, spare = null;
    const u = () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const n = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x = 0; while (x === 0) x = u();
      const y = u(), rr = Math.sqrt(-2 * Math.log(x));
      spare = rr * Math.sin(2 * Math.PI * y);
      return rr * Math.cos(2 * Math.PI * y);
    };
    return { u, n };
  }

  function crisp(cv) {                      // draw in CSS pixels on a 2x backing store
    if (!cv.dataset.w) {
      cv.dataset.w = cv.getAttribute('width'); cv.dataset.h = cv.getAttribute('height');
      cv.style.width = cv.dataset.w + 'px'; cv.style.height = cv.dataset.h + 'px';
      cv.width = 2 * cv.dataset.w; cv.height = 2 * cv.dataset.h;
    }
    const g = cv.getContext('2d');
    g.setTransform(2, 0, 0, 2, 0, 0);
    g.clearRect(0, 0, +cv.dataset.w, +cv.dataset.h);
    return [g, +cv.dataset.w, +cv.dataset.h];
  }

  function axes(box, xr, yr) {
    const [L, T, W, Hh] = box;
    return { L, T, W, H: Hh, xr, yr,
      X: v => L + (v - xr[0]) / (xr[1] - xr[0]) * W,
      Y: v => T + Hh - (v - yr[0]) / (yr[1] - yr[0]) * Hh };
  }
  function path(g, A, xs, ys) {
    g.beginPath();
    xs.forEach((x, i) => (i ? g.lineTo(A.X(x), A.Y(ys[i])) : g.moveTo(A.X(x), A.Y(ys[i]))));
  }
  function line(g, A, xs, ys, col, lw = 2.4, dash = null, alpha = 1) {
    g.save(); g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = lw; g.lineJoin = 'round';
    if (dash) g.setLineDash(dash);
    path(g, A, xs, ys); g.stroke(); g.restore();
  }
  function area(g, A, xs, ys, col, alpha = 1) {
    g.save(); g.globalAlpha = alpha; g.fillStyle = col;
    path(g, A, xs, ys); g.lineTo(A.X(xs[xs.length - 1]), A.Y(0)); g.lineTo(A.X(xs[0]), A.Y(0)); g.closePath(); g.fill(); g.restore();
  }
  function xaxis(g, A, ticks, lab) {
    g.save(); g.strokeStyle = '#9AA1AD'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(A.L, A.Y(A.yr[0]) + 0.5); g.lineTo(A.L + A.W, A.Y(A.yr[0]) + 0.5); g.stroke();
    g.fillStyle = PAL.ink3; g.font = FONT; g.textAlign = 'center';
    for (const t of ticks) {
      g.beginPath(); g.moveTo(A.X(t), A.Y(A.yr[0])); g.lineTo(A.X(t), A.Y(A.yr[0]) + 5); g.stroke();
      g.fillText(fx(t, Number.isInteger(t) ? 0 : 1), A.X(t), A.Y(A.yr[0]) + 19);
    }
    if (lab) { g.font = 'italic 15px "Times New Roman", serif'; g.fillText(lab, A.L + A.W / 2, A.Y(A.yr[0]) + 38); }
    g.restore();
  }
  const set = (root, cls, v) => root.querySelectorAll(cls).forEach(el => { el.textContent = v; });
  function adam(lr) {                      // Adam on a parameter vector, for gradient ASCENT
    const st = { m: null, v: null, t: 0 };
    return (p, grad) => {
      if (!st.m) { st.m = grad.map(() => 0); st.v = grad.map(() => 0); }
      st.t += 1;
      for (let i = 0; i < p.length; i++) {
        st.m[i] = 0.9 * st.m[i] + 0.1 * grad[i];
        st.v[i] = 0.999 * st.v[i] + 0.001 * grad[i] * grad[i];
        p[i] += lr * (st.m[i] / (1 - 0.9 ** st.t)) / (Math.sqrt(st.v[i] / (1 - 0.999 ** st.t)) + 1e-8);
      }
    };
  }
  function onSlide(root) { const s = root.closest('section'); return !s || s.classList.contains('current'); }

  // The PML course's examples (vi-lab/vi_lecture.ipynb): the Beta(5, 3) coin, p = N(1, 4), two mixtures of Gaussians.
  const softplus = v => (v > 20 ? v : Math.log1p(Math.exp(v)));
  const invSoftplus = v => (v > 20 ? v : Math.log(Math.expm1(v)));
  const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  function lgamma(z) {
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1;
    let x = LANCZOS[0];
    for (let i = 1; i < 9; i++) x += LANCZOS[i] / (z + i);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  const lbeta = (a, b) => lgamma(a) + lgamma(b) - lgamma(a + b);
  const betaLogPdf = (t, a, b) => (a - 1) * Math.log(t) + (b - 1) * Math.log1p(-t) - lbeta(a, b);
  const nums = s => s.split(',').map(Number);
  function mixture(spec) {                     // "w1,w2|m1,m2|s1,s2"
    const [w, m, s] = spec.split('|').map(nums);
    const comp = x => w.map((wk, k) => Math.log(wk) - 0.5 * ((x - m[k]) / s[k]) ** 2 - Math.log(s[k] * SQ2PI));
    const logp = x => { const c = comp(x), mx = Math.max(...c); return mx + Math.log(c.reduce((a, v) => a + Math.exp(v - mx), 0)); };
    const dlogp = x => {                       // d log p / dx = sum_k r_k * (-(x - m_k) / s_k^2)
      const c = comp(x), mx = Math.max(...c), e = c.map(v => Math.exp(v - mx)), tot = e.reduce((a, v) => a + v, 0);
      return e.reduce((a, ek, k) => a + ek / tot * (-(x - m[k]) / (s[k] * s[k])), 0);
    };
    return { logp, dlogp, pdf: x => Math.exp(logp(x)) };
  }

  // ---------------------------------------------------------------- W1: prior x likelihood -> posterior, the Beta coin
  function betacoin(root) {
    const cv = root.querySelector('canvas');
    const [ia, ib, ih, it] = root.querySelectorAll('input[type=range]');
    const xs = linspace(0.001, 0.999, 400);
    function draw() {
      const a = +ia.value, b = +ib.value, h = +ih.value, t = +it.value;
      const prior = xs.map(x => Math.exp(betaLogPdf(x, a, b)));
      const post = xs.map(x => Math.exp(betaLogPdf(x, a + h, b + t)));
      const pmax = Math.max(...post, ...prior.map(v => Math.min(v, 12)));
      const llk = xs.map(x => h * Math.log(x) + t * Math.log1p(-x)), lmax = Math.max(...llk);
      const lik = llk.map(v => Math.exp(v - lmax) * Math.max(...post));
      const [g, W, Hh] = crisp(cv);
      const A = axes([16, 10, W - 32, Hh - 52], [0, 1], [0, Math.min(pmax, 14) * 1.1]);
      area(g, A, xs, post, PAL.xt);
      line(g, A, xs, post, PAL.x, 3);
      if (h + t > 0) line(g, A, xs, lik, PAL.dec, 2.2, [6, 4]);
      line(g, A, xs, prior.map(v => Math.min(v, 14)), PAL.z, 2.4);
      xaxis(g, A, [0, 0.25, 0.5, 0.75, 1], '');
      set(root, '.bc-a', fx(a, 1)); set(root, '.bc-b', fx(b, 1)); set(root, '.bc-h', h); set(root, '.bc-t', t);
      set(root, '.bc-pa', fx(a + h, 1)); set(root, '.bc-pb', fx(b + t, 1));
      set(root, '.bc-mean', fx((a + h) / (a + b + h + t), 3)); set(root, '.bc-mle', h + t ? fx(h / (h + t), 3) : '—');
    }
    [ia, ib, ih, it].forEach(el => el.addEventListener('input', draw));
    draw();
  }

  // ---------------------------------------------------------------- W2: KL between two Gaussians, both directions, as areas
  function klgauss(root) {
    const cv = root.querySelector('canvas');
    const [im, is] = root.querySelectorAll('input[type=range]');
    const pm = +(root.dataset.pmu || 0), ps = +(root.dataset.psd || 1);
    const lo = +(root.dataset.xlo || -4.5), hi = +(root.dataset.xhi || 4.5);
    let dir = 'qp';
    const xs = linspace(lo, hi, 801), dx = xs[1] - xs[0];
    function draw() {
      const mu = +im.value, sd = +is.value;
      const p = xs.map(x => npdf(x, pm, ps)), q = xs.map(x => npdf(x, mu, sd));
      const f = dir === 'qp' ? q.map((v, i) => v * Math.log(v / p[i])) : p.map((v, i) => v * Math.log(v / q[i]));
      const areaNum = f.reduce((a, b) => (Number.isFinite(b) ? a + b : a), 0) * dx;
      const [g, W, Hh] = crisp(cv);
      const A = axes([16, 8, W - 32, 150], [lo, hi], [0, Math.max(...p, ...q) * 1.08]);
      area(g, A, xs, p, PAL.xt); line(g, A, xs, p, PAL.x, 2.4);
      line(g, A, xs, q, PAL.enc, 2.6);
      const top = Math.max(0.02, ...f.filter(Number.isFinite)), bot = Math.min(-0.005, ...f.filter(Number.isFinite));
      const B = axes([16, 178, W - 32, Hh - 220], [lo, hi], [bot * 1.15, top * 1.15]);
      g.save();
      for (const [sign, col] of [[1, '#F7B8C6'], [-1, '#E5E7EB']]) {
        g.fillStyle = col; g.beginPath(); g.moveTo(B.X(xs[0]), B.Y(0));
        xs.forEach((x, i) => g.lineTo(B.X(x), B.Y(sign * f[i] > 0 ? f[i] : 0)));
        g.lineTo(B.X(xs[xs.length - 1]), B.Y(0)); g.closePath(); g.fill();
      }
      g.restore();
      line(g, B, xs, f.map(v => (Number.isFinite(v) ? v : 0)), PAL.loss, 2);
      g.save(); g.strokeStyle = PAL.ink3; g.lineWidth = 0.8; g.beginPath(); g.moveTo(B.L, B.Y(0)); g.lineTo(B.L + B.W, B.Y(0)); g.stroke(); g.restore();
      const step = (hi - lo) > 12 ? 5 : 2, ticks = [];
      for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
      xaxis(g, B, ticks, '');
      const qp = Math.log(ps / sd) + (sd * sd + (mu - pm) ** 2) / (2 * ps * ps) - 0.5;
      const pq = Math.log(sd / ps) + (ps * ps + (pm - mu) ** 2) / (2 * sd * sd) - 0.5;
      set(root, '.kg-mu', fx(mu, 2)); set(root, '.kg-sd', fx(sd, 2));
      set(root, '.kg-qp', fx(qp, 3)); set(root, '.kg-pq', fx(pq, 3)); set(root, '.kg-area', fx(areaNum, 3));
      root.querySelectorAll('[data-dir]').forEach(b => b.classList.toggle('on', b.dataset.dir === dir));
    }
    [im, is].forEach(el => el.addEventListener('input', draw));
    root.querySelectorAll('[data-dir]').forEach(b => b.addEventListener('click', () => { dir = b.dataset.dir; draw(); }));
    draw();
  }

  // ---------------------------------------------------------------- W3: forward vs reverse KL on a mixture, by gradient descent
  function fwdrev(root) {
    const cvs = [...root.querySelectorAll('canvas')];
    const xs = linspace(-8, 8, 801), dx = xs[1] - xs[0], xd = linspace(-6, 6, 481);   // the notebook's grid; draw the middle
    let width = +(root.dataset.width || 0.5), start = -0.5, mix, p, pd;
    const kl = (dir, m, rs) => {
      const s = softplus(rs); let a = 0;
      for (let i = 0; i < xs.length; i++) {
        const q = npdf(xs[i], m, s) + 1e-300, pp = p[i] + 1e-300;
        a += dir === 'fwd' ? pp * Math.log(pp / q) : q * Math.log(q / pp);
      }
      return a * dx;
    };
    let st, steps, timer = null;
    function reset() {
      if (timer) cancelAnimationFrame(timer);
      timer = null;
      mix = mixture(`0.7,0.3|-2,2|${width},${width}`);
      p = xs.map(mix.pdf); pd = xd.map(mix.pdf);
      st = [{ d: 'fwd', p: [0.5, invSoftplus(0.6)], opt: adam(0.05) }, { d: 'rev', p: [start, invSoftplus(0.6)], opt: adam(0.05) }];
      steps = 0; draw();
    }
    const step = () => {
      for (const s of st) {
        const h = 1e-4, [m, rs] = s.p;
        s.opt(s.p, [-(kl(s.d, m + h, rs) - kl(s.d, m - h, rs)) / (2 * h), -(kl(s.d, m, rs + h) - kl(s.d, m, rs - h)) / (2 * h)]);
      }
      steps += 1;
    };
    function draw() {
      st.forEach((s, k) => {
        const [g, W, Hh] = crisp(cvs[k]);
        const A = axes([12, 8, W - 24, Hh - 46], [-6, 6], [0, Math.max(...pd) * 1.5]);
        area(g, A, xd, pd, PAL.xt); line(g, A, xd, pd, PAL.x, 1.6);
        const [m, rs] = s.p;
        line(g, A, xd, xd.map(x => npdf(x, m, softplus(rs))), PAL.enc, 2.8);
        xaxis(g, A, [-4, -2, 0, 2, 4], '');
        set(root, `.fr-${s.d}-m`, fx(m, 2)); set(root, `.fr-${s.d}-s`, fx(softplus(rs), 2)); set(root, `.fr-${s.d}-kl`, fx(kl(s.d, m, rs), 3));
      });
      set(root, '.fr-step', steps);
      root.querySelectorAll('[data-width]').forEach(b => b.classList.toggle('on', +b.dataset.width === width));
      root.querySelectorAll('[data-start]').forEach(b => b.classList.toggle('on', +b.dataset.start === start));
      const pb = root.querySelector('[data-a="play"]'); if (pb) pb.textContent = timer ? 'pause' : 'play';
    }
    const tick = () => {
      if (!onSlide(root) || steps >= 800) { timer = null; draw(); return; }
      for (let i = 0; i < 4; i++) step();
      draw(); timer = requestAnimationFrame(tick);
    };
    root.querySelector('[data-a="play"]').addEventListener('click', () => {
      if (timer) { cancelAnimationFrame(timer); timer = null; draw(); return; }
      if (steps >= 800) reset();
      timer = requestAnimationFrame(tick); draw();
    });
    root.querySelector('[data-a="reset"]').addEventListener('click', reset);
    root.querySelectorAll('[data-width]').forEach(b => b.addEventListener('click', () => { width = +b.dataset.width; reset(); }));
    root.querySelectorAll('[data-start]').forEach(b => b.addEventListener('click', () => { start = +b.dataset.start; reset(); }));
    reset();
    if (PRINT) { while (steps < 800) step(); draw(); }
  }

  // ---------------------------------------------------------------- W4 / W5: q chases a target by gradient descent on KL(q || p)
  // data-mode="exact": p = N(pmu, psd), the closed-form KL and its gradient (notebook N3, G2)
  // data-mode="mc":    p = a mixture, Monte Carlo KL with reparameterised draws (notebook N5, G4)
  function chase(root) {
    const mode = root.dataset.mode || 'exact';
    const [cq, cl] = root.querySelectorAll('canvas');
    const lo = +root.dataset.xlo, hi = +root.dataset.xhi, STEPS = +(root.dataset.steps || 400), S = +(root.dataset.draws || 200);
    const init = nums(root.dataset.init || '4,0.5'), lr = +(root.dataset.lr || 0.05);
    const pm = +(root.dataset.pmu || 1), ps = +(root.dataset.psd || 4);
    const mix = mode === 'mc' ? mixture(root.dataset.mix) : null;
    const xs = linspace(lo, hi, 400);
    const p = xs.map(x => (mix ? mix.pdf(x) : npdf(x, pm, ps)));
    let P, opt, step, trace, R, timer = null;
    function lossGrad() {
      const s = softplus(P[1]), ds = sig(P[1]);
      if (!mix) {                                 // KL(N(m, s) || N(pm, ps)) and its gradient
        const kl = Math.log(ps / s) + (s * s + (P[0] - pm) ** 2) / (2 * ps * ps) - 0.5;
        return [kl, [(P[0] - pm) / (ps * ps), (s / (ps * ps) - 1 / s) * ds]];
      }
      let kl = 0, gm = 0, gs = 0;                 // z = m + s * eps; loss = mean(log q(z) - log p(z))
      for (let k = 0; k < S; k++) {
        const e = R.n(), z = P[0] + s * e, g = mix.dlogp(z);
        kl += -Math.log(s) - 0.5 * e * e - 0.5 * Math.log(2 * Math.PI) - mix.logp(z);
        gm += -g; gs += -g * e;
      }
      return [kl / S, [gm / S, (gs / S - 1 / s) * ds]];
    }
    function reset() {
      if (timer) cancelAnimationFrame(timer);
      timer = null; P = [init[0], invSoftplus(init[1])]; opt = adam(lr); step = 0; R = rng(0); trace = [lossGrad()[0]]; draw();
    }
    function once() {
      const [, grad] = lossGrad();
      opt(P, grad.map(v => -v));                  // adam() ascends, so pass minus the gradient
      step += 1;
      trace.push(lossGrad()[0]);
    }
    function draw() {
      const s = softplus(P[1]);
      let [g, W, Hh] = crisp(cq);
      const q = xs.map(x => npdf(x, P[0], s));
      const A = axes([12, 8, W - 24, Hh - 46], [lo, hi], [0, Math.max(...p) * 1.9]);
      area(g, A, xs, p, PAL.xt); line(g, A, xs, p, PAL.x, 2.2);
      const top = A.yr[1] * 0.98, qmax = Math.max(...q);
      line(g, A, xs, q.map(v => Math.min(v, top)), PAL.enc, 2.8);
      if (qmax > top) {                           // say so when q's peak is cut off
        const right = P[0] > (lo + hi) / 2, off = A.X(P[0] + 2.2 * s) - A.X(P[0]) + 6;
        g.save(); g.fillStyle = PAL.enc; g.font = FONT; g.textAlign = right ? 'right' : 'left';
        g.fillText(`peak ${fx(qmax, 2)} ↑`, A.X(P[0]) + (right ? -off : off), A.T + 14); g.restore();
      }
      const span = hi - lo, stp = span > 10 ? 5 : 2, ticks = [];
      for (let t = Math.ceil(lo / stp) * stp; t <= hi; t += stp) ticks.push(t);
      xaxis(g, A, ticks, '');
      [g, W, Hh] = crisp(cl);
      const pos = trace.map(v => Math.max(v, 0.01));
      const logScale = mode === 'mc';
      const yv = logScale ? pos.map(Math.log10) : trace;
      const ymax = logScale ? Math.log10(Math.max(...pos.slice(0, 1).concat([10]))) : Math.max(trace[0], 0.5);
      const B = axes([48, 10, W - 80, Hh - 52], [0, STEPS], [logScale ? -2 : 0, ymax * 1.05]);
      g.save(); g.fillStyle = PAL.ink3; g.font = FONT; g.textAlign = 'right';
      const yt = logScale ? [-2, -1, 0, 1] : [0, ymax / 2, ymax];
      for (const v of yt) { g.fillText(logScale ? (10 ** v).toString() : fx(v, 1), B.L - 6, B.Y(v) + 4); }
      g.restore();
      line(g, B, yv.map((_, i) => i), yv, PAL.loss, 2);
      xaxis(g, B, [0, STEPS / 2, STEPS], 'step');
      set(root, '.ch-step', step); set(root, '.ch-m', fx(P[0], 2)); set(root, '.ch-s', fx(s, 2)); set(root, '.ch-kl', fx(trace[trace.length - 1], 3));
      const pb = root.querySelector('[data-a="play"]'); if (pb) pb.textContent = timer ? 'pause' : (step >= STEPS ? 'play again' : 'play');
    }
    function tick() {
      if (!onSlide(root) || step >= STEPS) { timer = null; draw(); return; }
      for (let i = 0; i < Math.max(1, Math.round(STEPS / 240)); i++) if (step < STEPS) once();
      draw(); timer = requestAnimationFrame(tick);
    }
    root.querySelector('[data-a="play"]').addEventListener('click', () => {
      if (timer) { cancelAnimationFrame(timer); timer = null; draw(); return; }
      if (step >= STEPS) reset();
      timer = requestAnimationFrame(tick); draw();
    });
    root.querySelector('[data-a="reset"]').addEventListener('click', reset);
    reset();
    if (PRINT) { while (step < STEPS) once(); draw(); }
  }

  // ---------------------------------------------------------------- W6: the coin's ELBO and KL for any Beta(alpha, beta), and VI by play
  function coinelbo(root) {
    const cv = root.querySelector('canvas');
    const [ia, ib] = root.querySelectorAll('input[type=range]');
    const h = +(root.dataset.heads || 17), t = +(root.dataset.tails || 13), a0 = +(root.dataset.pa || 5), b0 = +(root.dataset.pb || 3);
    const N = 2000, th = Array.from({ length: N }, (_, i) => (i + 0.5) / N);
    const l0 = th.map(Math.log), l1 = th.map(x => Math.log1p(-x));
    const logJoint = th.map((x, i) => (a0 - 1 + h) * l0[i] + (b0 - 1 + t) * l1[i] - lbeta(a0, b0));
    const logEv = lbeta(a0 + h, b0 + t) - lbeta(a0, b0);
    const elbo = (a, b) => {                   // E_q[log p(D, theta) - log q(theta)], by the midpoint rule
      const lb = lbeta(a, b);
      let e = 0;
      for (let i = 0; i < N; i++) { const lq = (a - 1) * l0[i] + (b - 1) * l1[i] - lb; e += Math.exp(lq) * (logJoint[i] - lq); }
      return e / N;
    };
    const lo = logEv - 2.5, hi = logEv + 0.6, pct = v => `${(Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * 100}%`;
    const xs = linspace(0.001, 0.999, 300), post = xs.map(x => Math.exp(betaLogPdf(x, a0 + h, b0 + t)));
    let P = null, opt, step = 0, timer = null;
    function draw() {
      const a = +ia.value, b = +ib.value, e = elbo(a, b);
      const [g, W, Hh] = crisp(cv);
      const q = xs.map(x => Math.exp(betaLogPdf(x, a, b)));
      const A = axes([12, 8, W - 24, Hh - 40], [0, 1], [0, Math.max(...post, ...q.map(v => Math.min(v, 12))) * 1.08]);
      area(g, A, xs, post, PAL.xt); line(g, A, xs, post, PAL.x, 2.4);
      line(g, A, xs, q.map(v => Math.min(v, A.yr[1])), PAL.enc, 2.6);
      xaxis(g, A, [0, 0.5, 1], '');
      set(root, '.ce-a', fx(a, 1)); set(root, '.ce-b', fx(b, 1));
      set(root, '.ce-elbo', fx(e, 3)); set(root, '.ce-kl', fx(logEv - e, 3)); set(root, '.ce-lpx', fx(logEv, 3)); set(root, '.ce-step', step);
      set(root, '.ce-lo', fx(lo, 1)); set(root, '.ce-hi', fx(hi, 1));
      const nl = root.querySelector('.vf-line');
      nl.querySelector('.ce-mark-elbo').style.left = pct(e);
      nl.querySelector('.ce-mark-lpx').style.left = pct(logEv);
      const gp = nl.querySelector('.nl-gap'); gp.style.left = pct(e); gp.style.width = `calc(${pct(logEv)} - ${pct(e)})`;
      const pb = root.querySelector('[data-a="play"]'); if (pb) pb.textContent = timer ? 'pause' : 'play: fit from Beta(45, 45)';
    }
    function once() {                          // gradient ascent on the exact ELBO, in softplus coordinates
      const f = (u, v) => elbo(softplus(u), softplus(v)), hh = 1e-3;
      const g = [(f(P[0] + hh, P[1]) - f(P[0] - hh, P[1])) / (2 * hh), (f(P[0], P[1] + hh) - f(P[0], P[1] - hh)) / (2 * hh)];
      opt(P, g); step += 1;
      ia.value = softplus(P[0]); ib.value = softplus(P[1]);
    }
    function tick() {
      if (!onSlide(root) || step >= 3000) { timer = null; draw(); return; }
      for (let i = 0; i < 15; i++) once();
      draw(); timer = requestAnimationFrame(tick);
    }
    [ia, ib].forEach(el => el.addEventListener('input', () => { if (timer) { cancelAnimationFrame(timer); timer = null; } P = null; step = 0; draw(); }));
    const pb = root.querySelector('[data-a="play"]');
    if (pb) pb.addEventListener('click', () => {
      if (timer) { cancelAnimationFrame(timer); timer = null; draw(); return; }
      if (!P || step >= 3000) { P = [invSoftplus(45), invSoftplus(45)]; opt = adam(0.05); step = 0; ia.value = 45; ib.value = 45; }
      timer = requestAnimationFrame(tick); draw();
    });
    if (PRINT && pb) {                         // print the end of the fit
      P = [invSoftplus(45), invSoftplus(45)]; opt = adam(0.05);
      while (step < 3000) once();
    }
    draw();
  }

  const widgets = { kl, elbo, explorer, latent, gmmx, betacoin, klgauss, fwdrev, chase, coinelbo };
  document.querySelectorAll('[data-widget]').forEach(el => {
    const w = widgets[el.dataset.widget];
    if (w) w(el);
  });
})();

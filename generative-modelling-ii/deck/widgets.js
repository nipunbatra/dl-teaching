// Live widgets for Generative Modelling II. Each one is also driven by the slide's clicks,
// so every click (and every PDF page) shows a meaningful, fixed state.
(() => {
  const f3 = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3);
  const data = id => JSON.parse(document.getElementById(id).textContent);

  // Noise-level explorer: x_t = sqrt(abar) x0 + sqrt(1 - abar) eps for our 7, one fixed eps image.
  function explorerT(root) {
    const D = data('explorer-data');
    const cv = root.querySelector('canvas');
    const ctx = cv.getContext('2d');
    const input = root.querySelector('input');
    const slide = root.closest('section');
    const set = (cls, v) => { const el = root.querySelector(cls); if (el) el.textContent = v; };
    const p = D.pix[0] * 28 + D.pix[1];
    const draw = t => {
      const ab = D.ab[t], sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);
      for (let i = 0; i < 784; i++) {
        const x = Math.max(-1, Math.min(1, sa * D.x0[i] + sn * D.eps[i]));
        const a = (x + 1) / 2;   // ink amount
        const r = Math.round(255 * (1 - a) + 0x14 * a), g = Math.round(255 * (1 - a) + 0x17 * a), b = Math.round(255 * (1 - a) + 0x1F * a);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect((i % 28) * 10, Math.floor(i / 28) * 10, 10, 10);
      }
      ctx.strokeStyle = '#AA4E08'; ctx.lineWidth = 3;
      ctx.strokeRect(D.pix[1] * 10 + 1.5, D.pix[0] * 10 + 1.5, 7, 7);
      set('.v-t', t); set('.v-sab', sa.toFixed(3)); set('.v-snb', sn.toFixed(3));
      set('.v-pix', f3(sa * 1 + sn * D.eps[p]));
      set('.v-snr', ab / (1 - ab) >= 0.01 ? (ab / (1 - ab)).toFixed(3) : (ab / (1 - ab)).toExponential(1));
    };
    input.addEventListener('input', () => draw(Number(input.value)));
    slide.addEventListener('build', e => { const t = [100, 350, 920][e.detail - 1] ?? 920; input.value = t; draw(t); });
    draw(Number(input.value));
  }

  // One reverse step on the toy pixel: x2 = 0.5, beta = 0.2, abar2 = 0.64, abar1 = 0.8.
  function reverseStep(root) {
    const [ein, zin] = root.querySelectorAll('input');
    const set = (cls, v) => root.querySelectorAll(cls).forEach(el => { el.textContent = v; });
    const x = 0.5, beta = 0.2, a = 0.8, ab = 0.64, abp = 0.8;
    const c = beta / Math.sqrt(1 - ab), bt = beta * (1 - abp) / (1 - ab), sig = Math.sqrt(bt);
    const line = root.querySelector('.numline-live');
    const X = v => 40 + (v - 0.2) / (1.3 - 0.2) * 860;
    const upd = () => {
      const e = Number(ein.value), z = Number(zin.value);
      const mu = (x - c * e) / Math.sqrt(a), x1 = mu + sig * z, x0h = (x - Math.sqrt(1 - ab) * e) / Math.sqrt(ab);
      set('.v-e', f3(e)); set('.v-z', f3(z)); set('.v-mu', f3(mu)); set('.v-x1', f3(x1)); set('.v-x0h', f3(x0h));
      set('.v-sz', f3(sig * z));
      if (line) {
        line.querySelector('.m-mu').setAttribute('transform', `translate(${X(mu)},0)`);
        line.querySelector('.m-x1').setAttribute('transform', `translate(${X(x1)},0)`);
        line.querySelector('.m-x0h').setAttribute('transform', `translate(${X(x0h)},0)`);
      }
    };
    ein.addEventListener('input', upd); zin.addEventListener('input', upd);
    root.closest('section').addEventListener('build', ev => {
      const states = [[-0.4, 0], [-0.4, 0.3], [-0.4, 0.3]];
      const s = states[ev.detail - 1] || states[states.length - 1];
      ein.value = s[0]; zin.value = s[1]; upd();
    });
    upd();
  }

  // Scrub through recorded sampler states (our model): row = seed, column = recorded step.
  function scrub(root) {
    const D = data('scrub-data');
    const [cx, ce] = root.querySelectorAll('canvas');
    const input = root.querySelector('input');
    const imgs = ['figures/data/scrub-x.png', 'figures/data/scrub-e.png'].map(src => { const im = new Image(); im.src = src; return im; });
    let seed = 0;
    const draw = () => {
      const k = Number(input.value), t = D.ts[k];
      [[cx, imgs[0]], [ce, imgs[1]]].forEach(([c, im]) => {
        const g = c.getContext('2d'); g.imageSmoothingEnabled = false;
        if (im.complete) g.drawImage(im, k * 28, seed * 28, 28, 28, 0, 0, c.width, c.height);
      });
      root.querySelector('.v-t').textContent = t;
      root.querySelectorAll('[data-seed]').forEach(b => b.classList.toggle('on', Number(b.dataset.seed) === seed));
    };
    imgs.forEach(im => { im.onload = draw; });
    input.max = D.ts.length - 1;
    input.addEventListener('input', draw);
    root.querySelectorAll('[data-seed]').forEach(b => b.addEventListener('click', () => { seed = Number(b.dataset.seed); draw(); }));
    root.closest('section').addEventListener('build', e => {
      const at = { 1: 0, 2: D.ts.indexOf(400), 3: D.ts.length - 1 }[e.detail] ?? D.ts.length - 1;
      input.value = at; draw();
    });
    draw();
  }

  const widgets = { 'explorer-t': explorerT, 'reverse-step': reverseStep, scrub };
  document.querySelectorAll('[data-widget]').forEach(el => {
    const w = widgets[el.dataset.widget];
    if (w) w(el);
  });
})();

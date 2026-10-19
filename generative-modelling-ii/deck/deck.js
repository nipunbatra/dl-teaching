// Generative Modelling II — presentation runtime (the same runtime as Generative Modelling I).
// Keys: →/Space/PgDn next click · ← previous · Shift+→/← slide · Home/End
//       O overview · S speaker notes · F fullscreen · R reading view
// URL:  ?print (one page per slide, final click) · ?print=builds (every click) · ?reading
// Speaker notes: <p data-say="n"> is the cue for click n; the S panel highlights it.
(() => {
  const params = new URLSearchParams(location.search);
  const deck = document.getElementById('deck');
  const slides = [...deck.querySelectorAll('section.slide')];
  const ATTRS = ['b', 'hl', 'dim', 'gone'];

  // ---------------------------------------------------------------- builds
  // "3" = from click 3 on · "2-4" = clicks 2..4 · "=2" = only click 2 · combine with commas
  function ranges(spec) {
    return spec.split(',').map(p => {
      p = p.trim();
      if (p.startsWith('=')) { const v = Number(p.slice(1)); return [v, v]; }
      if (p.includes('-')) { const [lo, hi] = p.split('-'); return [Number(lo), hi === '' ? Infinity : Number(hi)]; }
      return [Number(p), Infinity];
    });
  }
  const inRange = (spec, b) => ranges(spec).some(([lo, hi]) => b >= lo && b <= hi);
  // a cue "2" belongs to click 2 only; "2-4" to clicks 2..4
  const sayAt = (spec, b) => spec.split(',').some(p => {
    p = p.trim().replace(/^=/, '');
    if (p.includes('-')) { const [lo, hi] = p.split('-').map(Number); return b >= lo && b <= hi; }
    return Number(p) === b;
  });

  function buildCount(s) {
    let m = Number(s.dataset.builds || 1);
    s.querySelectorAll(ATTRS.map(a => `[data-${a}]`).join(',')).forEach(el => {
      for (const a of ATTRS) {
        const v = el.dataset[a];
        if (v) (v.match(/\d+/g) || []).forEach(n => { m = Math.max(m, Number(n)); });
      }
    });
    return m;
  }

  function apply(s, b) {
    s.dataset.build = b;
    s.querySelectorAll('[data-b]').forEach(el => el.classList.toggle('hidden', !inRange(el.dataset.b, b)));
    s.querySelectorAll('[data-gone]').forEach(el => el.classList.toggle('gone', inRange(el.dataset.gone, b)));
    s.querySelectorAll('[data-hl]').forEach(el => el.classList.toggle('hl', inRange(el.dataset.hl, b)));
    s.querySelectorAll('[data-dim]').forEach(el => el.classList.toggle('dim', inRange(el.dataset.dim, b)));
    s.dispatchEvent(new CustomEvent('build', { detail: b }));
  }

  slides.forEach((s, i) => {
    s.dataset.n = buildCount(s);
    const no = document.createElement('div');
    no.className = 'pageno';
    no.textContent = i + 1;
    s.appendChild(no);
  });

  // ---------------------------------------------------------------- maths
  function renderMath(root) {
    if (!window.renderMathInElement) return;
    renderMathInElement(root, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
      macros: window.MATH_MACROS || {},
    });
  }

  // ---------------------------------------------------------------- print / reading
  const printMode = params.get('print');
  if (printMode !== null) {
    document.body.classList.add('print');
    if (printMode === 'builds') {
      for (const s of slides) {
        const n = Number(s.dataset.n);
        for (let b = 1; b < n; b++) {
          const c = s.cloneNode(true);
          c.removeAttribute('id');
          c.querySelectorAll('[id]').forEach(e => e.removeAttribute('id'));
          s.before(c);
          apply(c, b);
          c.querySelector('.pageno').textContent = `${slides.indexOf(s) + 1}.${b}`;
        }
        apply(s, n);
        if (n > 1) s.querySelector('.pageno').textContent = `${slides.indexOf(s) + 1}.${n}`;
      }
    } else {
      slides.forEach(s => apply(s, Number(s.dataset.n)));
    }
    renderMath(document.body);
    window.deckReady = true;
    return;
  }
  if (params.has('reading')) {
    document.body.classList.add('reading');
    slides.forEach(s => apply(s, Number(s.dataset.n)));
    document.querySelectorAll('aside.notes [data-say]').forEach(p => {
      const tag = document.createElement('b');
      tag.className = 'say-tag';
      tag.textContent = `click ${p.dataset.say.replace(/^=/, '')}`;
      p.prepend(tag);
    });
    renderMath(document.body);
    window.deckReady = true;
    return;
  }

  // ---------------------------------------------------------------- present
  let cur = 0, build = 1;
  const hud = document.createElement('div');
  hud.id = 'hud';
  document.body.appendChild(hud);
  const notesPanel = document.createElement('div');
  notesPanel.id = 'notes-panel';
  document.body.appendChild(notesPanel);
  const overview = document.createElement('div');
  overview.id = 'overview';
  document.body.appendChild(overview);

  function fit() {
    const sc = Math.min(innerWidth / 1280, innerHeight / 720);
    document.documentElement.style.setProperty('--scale', sc);
  }
  addEventListener('resize', fit);
  fit();

  function show(i, b, push = true) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    const s = slides[i];
    b = Math.max(1, Math.min(Number(s.dataset.n), b));
    if (i !== cur) slides[cur].classList.remove('current');
    cur = i; build = b;
    s.classList.add('current');
    apply(s, b);
    hud.textContent = `${i + 1} / ${slides.length}` + (Number(s.dataset.n) > 1 ? ` · ${b}/${s.dataset.n}` : '');
    const notes = s.querySelector('aside.notes');
    notesPanel.innerHTML = `<h4>Speaker notes · slide ${i + 1} · click ${b} of ${s.dataset.n}</h4>` +
      (notes ? notes.innerHTML : '<p>—</p>');
    notesPanel.querySelectorAll('[data-say]').forEach(p => p.classList.toggle('now', sayAt(p.dataset.say, b)));
    if (push) history.replaceState(null, '', `#${s.id || i + 1}/${b}`);
  }

  function next() {
    const n = Number(slides[cur].dataset.n);
    if (build < n) show(cur, build + 1); else if (cur < slides.length - 1) show(cur + 1, 1);
  }
  function prev() {
    if (build > 1) show(cur, build - 1);
    else if (cur > 0) show(cur - 1, Number(slides[cur - 1].dataset.n));
  }

  function buildOverview() {
    let html = '<ol>';
    let lastSec = '';
    slides.forEach((s, i) => {
      const sec = s.dataset.section || '';
      if (sec && sec !== lastSec) { html += `<li class="sec">${sec}</li>`; lastSec = sec; }
      const t = (s.querySelector('h2')?.childNodes[0]?.textContent || s.id || '').trim();
      html += `<li><a href="#" data-i="${i}">${i + 1}. ${t}</a></li>`;
    });
    overview.innerHTML = html + '</ol>';
    overview.querySelectorAll('a').forEach(a => a.onclick = e => {
      e.preventDefault(); overview.classList.remove('open'); show(Number(a.dataset.i), 1);
    });
  }

  addEventListener('keydown', e => {
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) {
      if (e.key === 'Escape') t.blur();
      return;
    }
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ') { e.preventDefault(); e.shiftKey ? show(cur + 1, 1) : next(); }
    else if (k === 'ArrowLeft' || k === 'PageUp') { e.preventDefault(); e.shiftKey ? show(cur - 1, 1) : prev(); }
    else if (k === 'Home') show(0, 1);
    else if (k === 'End') show(slides.length - 1, 1);
    else if (k === 'o' || k === 'O') { buildOverview(); overview.classList.toggle('open'); }
    else if (k === 's' || k === 'S') notesPanel.classList.toggle('open');
    else if (k === 'f' || k === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    else if (k === 'r' || k === 'R') location.search = '?reading';
    else if (k === 'Escape') { overview.classList.remove('open'); notesPanel.classList.remove('open'); }
  });
  deck.addEventListener('click', e => {
    if (e.target.closest('input, button, a, label, select, .interactive')) return;
    if (e.clientX > innerWidth * 0.25) next(); else prev();
  });

  renderMath(document.body);
  const [id, b] = decodeURIComponent(location.hash.slice(1)).split('/');
  let start = slides.findIndex(s => s.id === id);
  if (start < 0) start = Number(id) - 1 || 0;
  slides[0].classList.add('current');
  show(start, Number(b) || 1, false);
  window.deck = { show, slides, get current() { return cur; }, get build() { return build; } };
  window.deckReady = true;
})();

// Right-click review notes.
//   Right-click anything on a slide → type what should change → Save (or ⌘/Ctrl+Enter).
//   Red pin = open note. Green pin = Claude made the change (hover or click to read the reply).
//   Press A to list every note (Copy all, Clear all).  Shift+right-click = normal browser menu.
// Served by scripts/review_server.py, notes go straight to review/notes.json and reach Claude
// as soon as they are saved; the page reloads itself (same slide) whenever the deck is rebuilt.
// Opened any other way (file://, GitHub Pages) notes stay in this browser's localStorage.
(() => {
  const body = document.body;
  if (body.classList.contains('print') || body.classList.contains('reading')) return;
  const deckName = location.pathname.split('/').pop() || 'index.html';
  const KEY = 'deck-notes:' + deckName;
  const API = `/api/notes?deck=${encodeURIComponent(deckName)}`;
  let server = false;
  let notes = [];

  const local = {
    load() { try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch { return []; } },
    save(n) { try { localStorage.setItem(KEY, JSON.stringify(n)); } catch { /* private mode */ } },
  };
  async function pull() {
    if (!server) return local.load();
    const r = await fetch(API, { cache: 'no-store' });
    return r.ok ? r.json() : notes;
  }
  async function push() {
    local.save(notes);
    if (!server) return;
    try {
      const r = await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(notes) });
      if (r.ok) { notes = await r.json(); toast('Sent to Claude ✓'); }
    } catch { toast('Saved in this browser only (server not reachable)'); }
  }

  // text without KaTeX's duplicate MathML copy: maths becomes its LaTeX source
  function plainText(el) {
    if (!el) return '';
    const c = el.cloneNode(true);
    c.querySelectorAll('.katex').forEach(k => {
      const tex = k.querySelector('annotation[encoding="application/x-tex"]')?.textContent || '';
      k.replaceWith(document.createTextNode('$' + tex + '$'));
    });
    return c.textContent.trim().replace(/\s+/g, ' ');
  }

  // ------------------------------------------------------------ what was clicked?
  function describe(el) {
    const tex = el.closest('.katex')?.querySelector('annotation[encoding="application/x-tex"]');
    if (tex) return { kind: 'maths', text: tex.textContent.trim() };
    const svg = el.closest('svg.diagram');
    if (svg) {
      const t = el.closest('text, foreignObject')?.textContent.trim();
      return { kind: 'diagram ' + (svg.getAttribute('aria-label') || ''), text: t || '' };
    }
    if (el.tagName === 'IMG') return { kind: 'image', text: el.getAttribute('src') };
    const block = el.closest('h2, p, li, td, th, pre, .node, .callout, .stat, .cap, .q, .kicker') || el;
    return { kind: block.tagName.toLowerCase() + (block.className ? '.' + String(block.className).split(' ')[0] : ''),
             text: plainText(block).slice(0, 200) };
  }

  // ------------------------------------------------------------ pins
  function renderPins() {
    document.querySelectorAll('.anno-pin').forEach(p => p.remove());
    const all = [...document.querySelectorAll('section.slide')];
    notes.forEach(n => {
      const slide = n.slide ? document.getElementById(n.slide) : all[n.slideNo - 1];
      if (!slide) return;
      const pin = document.createElement('button');
      const done = n.status === 'done';
      pin.className = 'anno-pin interactive' + (done ? ' done' : '');
      pin.textContent = done ? '✓' : n.n;
      pin.title = n.note + (n.reply ? '\n\nClaude: ' + n.reply : '');
      pin.style.left = (n.x ?? 40) + 'px';
      pin.style.top = (n.y ?? 40) + 'px';
      pin.onclick = e => { e.stopPropagation(); openBox(e.clientX, e.clientY, n); };
      slide.appendChild(pin);
    });
  }

  // ------------------------------------------------------------ the note box
  const box = document.createElement('div');
  box.className = 'anno-box interactive';
  box.innerHTML = '<div class="anno-target"></div><div class="anno-reply"></div>' +
    '<textarea rows="4" placeholder="What should change here? (e.g. make this equation bigger, split this slide, use the 3 instead)"></textarea>' +
    '<div class="anno-actions"><button data-a="del">Delete</button><span></span>' +
    '<button data-a="cancel">Cancel</button><button data-a="save" class="primary">Send</button></div>';
  body.appendChild(box);
  const ta = box.querySelector('textarea');
  let pending = null;

  function openBox(cx, cy, existing, draft) {
    pending = existing || draft;
    box.querySelector('.anno-target').textContent =
      `slide ${pending.slideNo}, click ${pending.build} · ${pending.target.kind}: ${pending.target.text}`.slice(0, 220);
    const reply = box.querySelector('.anno-reply');
    reply.textContent = existing?.reply ? 'Claude: ' + existing.reply : '';
    reply.style.display = existing?.reply ? 'block' : 'none';
    ta.value = existing ? existing.note : '';
    box.querySelector('[data-a="del"]').style.visibility = existing ? 'visible' : 'hidden';
    box.style.display = 'block';
    box.style.left = Math.max(8, Math.min(cx, innerWidth - 400)) + 'px';
    box.style.top = Math.max(8, Math.min(cy, innerHeight - 240)) + 'px';
    ta.focus();
  }
  function closeBox() { box.style.display = 'none'; pending = null; }
  async function save() {
    const text = ta.value.trim();
    if (!pending || !text) return closeBox();
    const existing = notes.find(n => n.id === pending.id);
    if (existing) existing.note = text;
    else notes.push({ ...pending, note: text, status: 'open',
                      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
                      n: Math.max(0, ...notes.map(n => n.n || 0)) + 1, at: new Date().toISOString() });
    closeBox(); renderPins(); await push(); renderPins();
  }
  box.addEventListener('click', async e => {
    e.stopPropagation();
    const a = e.target.dataset.a;
    if (a === 'save') save();
    if (a === 'cancel') closeBox();
    if (a === 'del' && pending) { notes = notes.filter(n => n.id !== pending.id); closeBox(); renderPins(); await push(); }
  });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); save(); }
    if (e.key === 'Escape') { e.preventDefault(); closeBox(); }
  });

  document.getElementById('deck').addEventListener('contextmenu', e => {
    if (e.shiftKey) return;
    const slide = e.target.closest('section.slide');
    if (!slide) return;
    e.preventDefault();
    const r = slide.getBoundingClientRect();
    const scale = r.width / 1280 || 1;
    const all = [...document.querySelectorAll('section.slide')];
    openBox(e.clientX, e.clientY, null, {
      deck: deckName, slide: slide.id, slideNo: all.indexOf(slide) + 1, build: Number(slide.dataset.build || 1),
      title: plainText(slide.querySelector('h2')).slice(0, 120),
      x: Math.round((e.clientX - r.left) / scale), y: Math.round((e.clientY - r.top) / scale),
      target: describe(e.target),
    });
  });

  // ------------------------------------------------------------ list (key A)
  const panel = document.createElement('div');
  panel.className = 'anno-panel interactive';
  body.appendChild(panel);
  const asText = () => notes.map(n =>
    `${n.n}. [${n.status === 'done' ? 'done' : 'open'}] ${deckName} slide ${n.slideNo} "${n.title}" (click ${n.build}) · ${n.target.kind}: ${n.target.text}\n   → ${n.note}` +
    (n.reply ? `\n   Claude: ${n.reply}` : '')).join('\n');
  function showPanel() {
    const where = server ? 'Notes go straight to Claude.' : 'Offline: copy these and paste them into the chat.';
    panel.innerHTML = `<h4>Review notes (${notes.length}) · ${where}</h4><pre>${asText().replace(/</g, '&lt;') || 'No notes yet. Right-click anything on a slide.'}</pre>` +
      '<div class="anno-actions"><button data-a="copy" class="primary">Copy all</button><button data-a="clear">Clear all</button><button data-a="close">Close</button></div>';
    panel.style.display = 'block';
  }
  panel.addEventListener('click', async e => {
    e.stopPropagation();
    const a = e.target.dataset.a;
    if (a === 'copy') { try { await navigator.clipboard.writeText(asText()); e.target.textContent = 'Copied'; } catch { /* no clipboard */ } }
    if (a === 'clear' && confirm('Delete all review notes for this deck?')) { notes = []; renderPins(); await push(); showPanel(); }
    if (a === 'close') panel.style.display = 'none';
  });
  addEventListener('keydown', e => {
    if (e.target.closest?.('textarea, input')) return;
    if (e.key === 'a' || e.key === 'A') panel.style.display === 'block' ? (panel.style.display = 'none') : showPanel();
  });

  // ------------------------------------------------------------ toast
  const tst = document.createElement('div');
  tst.className = 'anno-toast';
  body.appendChild(tst);
  let tt;
  function toast(msg) { tst.textContent = msg; tst.style.opacity = 1; clearTimeout(tt); tt = setTimeout(() => (tst.style.opacity = 0), 1800); }

  // ------------------------------------------------------------ start, then poll for replies and rebuilds
  let version = null;
  async function poll() {
    if (!server || box.style.display === 'block') return;
    try {
      const v = (await (await fetch('/api/version', { cache: 'no-store' })).json()).version;
      if (version && v !== version) { location.reload(); return; }   // deck rebuilt: reload onto the same slide
      version = v;
      const fresh = await pull();
      if (JSON.stringify(fresh) !== JSON.stringify(notes)) { notes = fresh; renderPins(); }
    } catch { /* server stopped */ }
  }
  (async () => {
    try { server = (await fetch(API, { cache: 'no-store' })).ok; } catch { server = false; }
    notes = await pull();
    renderPins();
    if (server) { await poll(); setInterval(poll, 2000); }
  })();

  window.deckNotes = () => notes;
})();

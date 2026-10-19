// Capture every scene/build of GM-II from the live preview, plus visible text per state.
const puppeteer = require('/Users/nipun/git/dl-teaching/node_modules/puppeteer');
const fs = require('fs'), path = require('path');
const OUT = path.join(__dirname, 'caps');
(async () => {
  const b = await puppeteer.launch({ headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  const p = await b.newPage(); const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  p.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await p.setViewport({ width: 1600, height: 954, deviceScaleFactor: 1 });
  await p.goto('http://127.0.0.1:8767/generative-modelling-ii/index.html?present#s01/1', { waitUntil: 'networkidle0' });
  await p.evaluate(() => lecture.ready && lecture.ready());
  await new Promise(r => setTimeout(r, 1500));
  const scenes = await p.evaluate(() => lecture.data.scenes.map(s => ({ id: s.id, builds: s.builds, title: s.title, section: s.section, talk: s.talk_through, take: s.takeaway, steps: s.steps, narration: s.narration, question: s.question, answer: s.answer, source: s.source, explanation_steps: s.explanation_steps })));
  const text = [];
  for (const s of scenes) {
    for (let k = 1; k <= s.builds; k++) {
      await p.evaluate((id, k) => lecture.goTo(id, k), s.id, k);
      await new Promise(r => setTimeout(r, 450));
      const t = await p.evaluate(() => {
        const a = document.querySelector('.slide.active');
        const vis = el => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none' && +cs.opacity > 0.05; };
        const steps = [...a.querySelectorAll('.step')].filter(vis).map(e => e.innerText.trim());
        const cue = a.querySelector('.teaching-line')?.innerText.trim();
        const foot = a.querySelector('footer')?.innerText.trim();
        const eyebrow = a.querySelector('.eyebrow')?.innerText.trim();
        const h2 = a.querySelector('h2')?.innerText.trim();
        const progress = document.querySelector('#progress')?.textContent;
        return { eyebrow, h2, steps, cue, foot, progress };
      });
      text.push({ id: s.id, build: k, ...t });
      await p.screenshot({ path: path.join(OUT, `${s.id}-${k}.png`) });
    }
  }
  fs.writeFileSync(path.join(__dirname, 'scenes.json'), JSON.stringify(scenes, null, 1));
  fs.writeFileSync(path.join(__dirname, 'state-text.json'), JSON.stringify(text, null, 1));
  console.log('states', text.length, 'errors', errors);
  await b.close();
})();

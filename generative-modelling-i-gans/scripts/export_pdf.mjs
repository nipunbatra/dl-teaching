// Export the deck to PDF with headless Chrome.
//   node scripts/export_pdf.mjs            -> exports/gans.pdf (final build of every slide)
//                                             exports/gans-builds.pdf (every click)
import puppeteer from 'puppeteer';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const exe = process.env.PUPPETEER_EXECUTABLE_PATH ||
  ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome', '/usr/bin/chromium']
    .find(p => fs.existsSync(p));
const browser = await puppeteer.launch({ headless: true, executablePath: exe, args: ['--allow-file-access-from-files'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('requestfailed', r => errors.push('failed: ' + r.url()));

fs.mkdirSync(path.join(root, 'exports'), { recursive: true });

// Any maths character outside KaTeX (in slide text, not code or notes) is a typesetting bug.
const mathCheck = () => {
  const bad = /[μσεϵφθψπΣ‖∂∼²√∑∫]|N\(0|(^|[\s(])-\d/;
  const out = [];
  document.querySelectorAll('section.slide').forEach((s, i) => {
    const walker = document.createTreeWalker(s, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const el = n.parentElement;
      if (el.closest('.katex, pre, code, script, aside.notes, .pageno')) continue;
      if (bad.test(n.textContent)) out.push(`${s.querySelector('.pageno')?.textContent || i + 1}: ${n.textContent.trim().slice(0, 80)}`);
    }
  });
  return out;
};

const jobs = [['gans.html', '', 'gans.pdf'], ['gans.html', 'builds', 'gans-builds.pdf']];
for (const [file, mode, out] of jobs) {
  const url = pathToFileURL(path.join(root, file)).href + '?print' + (mode ? '=' + mode : '');
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.deckReady === true);
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all([...document.images].map(i => i.complete ? null : new Promise(r => { i.onload = i.onerror = r; })));
  });
  const n = await page.evaluate(() => document.querySelectorAll('section.slide').length);
  if (!mode) {
    const bad = await page.evaluate(mathCheck);
    if (bad.length) errors.push(`maths outside KaTeX in ${file}:\n  ` + bad.join('\n  '));
  }
  await page.pdf({ path: path.join(root, 'exports', out), width: '1280px', height: '720px', printBackground: true, preferCSSPageSize: true });
  console.log(out, n, 'pages');
}
await browser.close();
if (errors.length) { console.error([...new Set(errors)].join('\n')); process.exitCode = 1; }

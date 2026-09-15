// CHEATSHEET_PLAYWRIGHT_PATH may point to an existing Playwright installation.
// Preview _site first: python3 -m http.server 8837 --directory _site
const { chromium } = require(process.env.CHEATSHEET_PLAYWRIGHT_PATH || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const base = process.env.CHEATSHEET_PREVIEW_URL || 'http://127.0.0.1:8837';
const out = path.resolve(__dirname, '../work/cheatsheets-a4-review');

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    await fs.mkdir(out, { recursive: true });
    const report = [];
    for (const width of [1280, 390]) {
      const context = await browser.newContext({ viewport: { width, height: 900 } });
      try {
        const page = await context.newPage();
        for (const file of ['index.html', 'slides.html']) {
          await page.goto(`${base}/${file}`, { waitUntil: 'networkidle' });
          const section = page.locator('#attention-and-language');
          await section.waitFor();
          assert.equal(await section.locator('a[href*="?present"]').count(), 2);
          assert.equal(await section.locator('a[href$="cheatsheet.pdf"]').count(), 2);
          const links = await page.locator('a[href$="cheatsheet.pdf"]').evaluateAll(
            nodes => [...new Set(nodes.map(n => n.href))]);
          if (file === 'slides.html') assert.equal(links.length, 12);
          for (const url of links) {
            const response = await page.request.get(url);
            assert.equal(response.status(), 200, url);
            assert.equal((await response.body()).subarray(0, 5).toString(), '%PDF-', url);
          }
          const dimensions = await page.evaluate(() => ({
            viewport: window.innerWidth,
            content: document.documentElement.scrollWidth,
          }));
          assert.ok(dimensions.content <= dimensions.viewport + 1, JSON.stringify(dimensions));
          await section.scrollIntoViewIfNeeded();
          await page.screenshot({ path: path.join(out, `website-${file}-${width}.png`) });
          report.push({ file, width, pdfLinks: links.length, ...dimensions });
        }
      } finally {
        await context.close();
      }
    }
    console.log(JSON.stringify(report, null, 2));
    await fs.writeFile(path.join(out, 'website-report.json'), JSON.stringify(report, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

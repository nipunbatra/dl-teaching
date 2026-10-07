const assert = require("node:assert/strict"),
  fs = require("fs"),
  puppeteer = require(
    process.env.PUPPETEER_PATH || "../../../node_modules/puppeteer",
  );
(async () => {
  const b = await puppeteer.launch({
    executablePath:
      process.env.CHROME_PATH ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
    userDataDir: process.cwd() + "/work/browser",
    args: ["--use-angle=metal"],
    protocolTimeout: 600000,
  });
  try {
    const p = await b.newPage(),
      errors = [];
    p.on("pageerror", (e) => {
      console.log("PAGEERROR", e.message);
      errors.push(e.message);
    });
    await p.setViewport({ width: 1440, height: 1050 });
    await p.goto(process.env.LAB_URL || "http://127.0.0.1:5189");
    await p.waitForFunction(() => window.embeddingLab);
    await p.screenshot({
      path: "output/verification/initial.png",
      fullPage: true,
    });
    async function run(id) {
      await p.evaluate((id) => embeddingLab.activate(id), id);
      await p.click("#run");
      await p.waitForFunction(() => !embeddingLab.state.busy, {
        timeout: 600000,
      });
      const err = await p.$eval("#error", (n) =>
        n.hidden ? "" : n.textContent,
      );
      if (err) throw Error(id + ": " + err);
      return p.evaluate(() =>
        embeddingLab.state.results.map((r) => ({
          id: r.item.id,
          score: r.score,
        })),
      );
    }
    const report = {};
    for (const id of [
      "photos",
      "sounds",
      "listen",
      "captions",
      "neighbors",
      "languages",
      "mixed",
      "moments",
      "classify",
      "documents",
      "code",
      "delta",
      "clusters",
    ]) {
      report[id] = await run(id);
      console.log(id, JSON.stringify(report[id].slice(0, 3)));
      if (id === "sounds" || id === "documents" || id === "clusters")
        await p.screenshot({
          path: "output/verification/" + id + ".png",
          fullPage: true,
        });
    }
    await run("photos");
    await p.select("#dimension", "128");
    assert(
      await p.evaluate(() =>
        embeddingLab.state.results.every((r) => Number.isFinite(r.score)),
      ),
    );
    await p.$eval("#coordinates", (n) => {
      n.value = 120;
      n.dispatchEvent(new Event("input"));
    });
    assert(
      (await p.$eval("#coordinate-range", (n) => n.textContent)).includes(
        "121",
      ),
    );
    await p.select("#dimension", "768");
    await p.evaluate(() => embeddingLab.activate("listen"));
    await p.click("#recompute");
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy, {
      timeout: 300000,
    });
    assert(
      (await p.evaluate(() => embeddingLab.state.query.origin)).includes(
        "this browser",
      ),
    );
    await p.evaluate(() => embeddingLab.activate("neighbors"));
    await p.click("#recompute");
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy, {
      timeout: 300000,
    });
    assert(
      (await p.evaluate(() => embeddingLab.state.query.origin)).includes(
        "this browser",
      ),
    );
    const file = await p.$("#upload");
    await file.uploadFile(process.cwd() + "/public/media/coffee.jpg");
    await p.waitForFunction(() => embeddingLab.state.upload);
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy, {
      timeout: 300000,
    });
    assert.equal(
      await p.evaluate(() => embeddingLab.state.query.item.type),
      "image",
    );
    await p.setViewport({ width: 390, height: 844 });
    for (const id of ["photos", "listen", "delta", "clusters"]) {
      await run(id);
      assert(
        await p.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        "mobile overflow " + id,
      );
    }
    await p.screenshot({
      path: "output/verification/mobile.png",
      fullPage: true,
    });
    await p.click("#open-library");
    assert(
      await p.evaluate(
        () =>
          document.querySelector("#library").scrollWidth <=
          document.querySelector("#library").clientWidth,
      ),
    );
    await p.screenshot({ path: "output/verification/library-mobile.png" });
    await p.click('[data-close="library"]');
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      "output/verification/browser.json",
      JSON.stringify(
        { passed: true, queries: report, mobile: 390, errors },
        null,
        2,
      ),
    );
    console.log("ALL PASSED");
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

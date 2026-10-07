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
    p.on("pageerror", (e) => errors.push(e.message));
    await p.goto(process.env.LAB_URL || "http://127.0.0.1:5191");
    await p.waitForFunction(() => window.embeddingLab);
    await p.$eval("#query", (n) => (n.value = ""));
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy);
    assert(
      (await p.$eval("#error", (n) => n.textContent)).includes("Enter a query"),
    );
    await p.evaluate(() => embeddingLab.activate("photos"));
    await p.click("#run");
    await p.click("#stop");
    await p.waitForFunction(() => !embeddingLab.state.busy);
    assert.equal(await p.evaluate(() => embeddingLab.state.ready), false);
    // A subsequent run must recreate the worker after cancellation.
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy, {
      timeout: 300000,
    });
    assert(await p.evaluate(() => embeddingLab.state.query));
    await p.select("#query-type", "video");
    await p.click(".input-options summary");
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
    assert.equal(
      await p.evaluate(() => embeddingLab.state.query.info.frames),
      12,
    );
    const upload = await p.$("#upload");
    await upload.uploadFile(process.cwd() + "/public/media/three-scenes.mp4");
    await p.waitForFunction(() => embeddingLab.state.upload?.type === "video");
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy, {
      timeout: 300000,
    });
    assert.equal(
      await p.evaluate(() => embeddingLab.state.query.info.frames),
      12,
    );
    await p.select("#query-type", "audio");
    const audio = await p.$("#upload");
    const sounds = JSON.parse(fs.readFileSync("public/gallery.json")).filter(
      (x) => x.type === "audio",
    );
    await audio.uploadFile(process.cwd() + "/public/" + sounds[0].src);
    await p.waitForFunction(() => embeddingLab.state.upload?.type === "audio");
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy, {
      timeout: 300000,
    });
    assert.equal(
      await p.evaluate(() => embeddingLab.state.query.info.sampleRate),
      16000,
    );
    await p.evaluate(() => embeddingLab.activate("delta"));
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy);
    const scores = await p.evaluate(() =>
      embeddingLab.state.results.map((r) => [r.item.id, r.score]),
    );
    await p.click("#swap");
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy);
    const swapped = await p.evaluate(() =>
      Object.fromEntries(
        embeddingLab.state.results.map((r) => [r.item.id, r.score]),
      ),
    );
    scores.forEach(([id, s]) => assert(Math.abs(s + swapped[id]) < 1e-10));
    await p.evaluate(() => embeddingLab.activate("clusters"));
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy);
    assert.equal(await p.$$eval(".map-legend span", (n) => n.length), 4);
    await p.setViewport({ width: 390, height: 844 });
    await p.click("#open-library");
    await p.select("#library-filter", "audio");
    await p.waitForFunction(() =>
      Array.from(document.querySelectorAll("#library audio")).every(
        (n) => n.duration === 5,
      ),
    );
    assert(
      await p.evaluate(
        () =>
          document.querySelector("#library").scrollWidth <=
          document.querySelector("#library").clientWidth,
      ),
    );
    await p.click('[data-close="library"]');
    // Stored vector comparisons remain available without WebGPU.
    await p.evaluate(() => {
      embeddingLab.engine.stop();
      embeddingLab.state.ready = false;
      Object.defineProperty(navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      embeddingLab.activate("captions");
    });
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy);
    assert(await p.evaluate(() => embeddingLab.state.query));
    await p.evaluate(() => embeddingLab.activate("photos"));
    await p.click("#run");
    await p.waitForFunction(() => !embeddingLab.state.busy);
    assert(
      (await p.$eval("#error", (n) => n.textContent)).includes(
        "WebGPU is unavailable",
      ),
    );
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      "output/verification/edges.json",
      JSON.stringify(
        {
          passed: true,
          cancelAndRetry: true,
          audioUpload: true,
          videoUpload: true,
          liveVideoFrames: 12,
          deltaReversal: true,
          noWebGPUFallback: true,
          audioPlaybackMetadata: true,
          errors,
        },
        null,
        2,
      ),
    );
    console.log("EDGE CHECKS PASSED");
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

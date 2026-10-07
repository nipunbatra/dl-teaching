const puppeteer = require(
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
  });
  try {
    const p = await b.newPage();
    p.on("pageerror", (e) => console.log("PAGEERROR", e.message));
    p.on("console", async (m) => {
      const t = m.text();
      if (t.includes("rror")) console.log(t);
    });
    await p.goto("http://127.0.0.1:5189");
    await p.waitForFunction(() => window.lab);
    console.log(
      "GPU",
      await p.evaluate(async () => ({
        gpu: !!navigator.gpu,
        adapter: !!(await navigator.gpu?.requestAdapter()),
      })),
    );
    const result = await p.evaluate(() =>
      lab.embed({ text: "task: search result | query: a dog barking" }),
    );
    console.log(
      "RESULT",
      JSON.stringify({
        dims: result.vector.length,
        norm: Math.hypot(...result.vector),
        shapes: result.shapes,
        elapsed: result.elapsed,
      }),
    );
    require("fs").writeFileSync(
      "output/verification/proof.json",
      JSON.stringify(result),
    );
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

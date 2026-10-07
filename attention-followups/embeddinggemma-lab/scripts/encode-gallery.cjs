const fs = require("fs"),
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
    protocolTimeout: 1200000,
  });
  try {
    const p = await b.newPage();
    p.on("pageerror", (e) => console.log("ERROR", e.message));
    await p.goto("http://127.0.0.1:5189/proof.html");
    await p.evaluate(async () => {
      const { Engine, prepare } = await import("/src/runtime.js");
      window.engine = new Engine();
      window.prepare = prepare;
      await engine.load();
    });
    console.log("MODEL READY");
    const all = JSON.parse(fs.readFileSync("public/gallery.json")),
      only = process.argv.slice(2);
    const items = only.length ? all.filter((x) => only.includes(x.id)) : all;
    const rows = only.length
      ? JSON.parse(fs.readFileSync("public/embeddings.json")).items
      : {};
    for (const item of items) {
      const data = await p.evaluate(async (item) => {
        const { input, info } = await prepare(
          item,
          "search result",
          "document",
        );
        return { ...(await engine.embed(input)), info };
      }, item);
      rows[item.id] = data;
      console.log(item.id, data.elapsed.toFixed(0), data.vector.length);
      fs.writeFileSync(
        "output/verification/gallery-progress.json",
        JSON.stringify(rows),
      );
    }
    fs.writeFileSync(
      "public/embeddings.json",
      JSON.stringify({
        model: "onnx-community/embeddinggemma-2-ONNX",
        revision: "daa72c51243991dfcaf9f9137d2c573d8f7790c0",
        dtype: "q4",
        runtime: "Transformers.js 4.3.1",
        device: "webgpu",
        date: new Date().toISOString(),
        dimensions: 768,
        items: rows,
      }),
    );
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

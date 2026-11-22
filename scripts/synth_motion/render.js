// Render synthetic motion-graphics jobs to PNG frames with headless Chromium (Playwright).
//   node scripts/synth_motion/render.js jobs.json out_dir
// jobs.json: [{id, scene, width, height, frames, params:{...}}]; writes out_dir/<id>/f0000.png ...
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

(async () => {
  const jobs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const out = process.argv[3];
  const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.goto("file://" + path.resolve(__dirname, "scenes.html"));
  await page.addScriptTag({ url: "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" });
  for (const job of jobs) {
    const dir = path.join(out, job.id);
    if (fs.existsSync(path.join(dir, "done"))) continue;
    fs.mkdirSync(dir, { recursive: true });
    await page.setViewportSize({ width: job.width, height: job.height });
    await page.evaluate((j) => window.setup(j), job);
    const canvas = await page.$("#c2d");
    for (let i = 0; i < job.frames; i++) {
      await page.evaluate(([i, n]) => window.drawFrame(i, n), [i, job.frames]);
      await canvas.screenshot({ path: path.join(dir, `f${String(i).padStart(4, "0")}.png`) });
    }
    fs.writeFileSync(path.join(dir, "done"), "1");
    console.log("rendered", job.id);
  }
  await browser.close();
})();

// Acceptance check for M6: clicking a stone opens the correct record.
// Picks at the screen position of a known point of each of N stones and compares IDs,
// then clicks one stone through the real UI and checks the card. Uses an installed
// Chrome (no browser download). Usage: node tests/pick_check.mjs [url] [chromePath]
import puppeteer from "puppeteer-core";

const url = process.argv[2] ?? "http://localhost:4173/";
const chrome = process.argv[3] ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=d3d11"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 900 });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(url, { waitUntil: "networkidle0", timeout: 120000 });
await page.waitForFunction(() => /points/.test(document.getElementById("status").textContent),
  { timeout: 120000 });
await new Promise((r) => setTimeout(r, 1500));
const res = await page.evaluate(() => window.__stonetwin.sampleCheck(400));
console.log("programmatic picks:", JSON.stringify(res));

// Real clicks through the UI on interior points of a few stones; check the card title.
await page.screenshot({ path: "tests/viewer_rgb.png" });
const targets = await page.evaluate(() => window.__stonetwin.stoneTargets(400));
const chosen = targets.filter((_, i) => i % 40 === 0);
let uiOk = 0;
for (const t of chosen) {
  // Close any open card first: it is an overlay and would swallow the click.
  await page.evaluate(() => document.getElementById("close")?.click());
  await page.mouse.click(t.x, t.y);
  await new Promise((r) => setTimeout(r, 150));
  const title = await page.evaluate(() => {
    const c = document.getElementById("card");
    return c.hidden ? null : c.querySelector("h2").textContent;
  });
  if (title === `Stone ${t.id}`) uiOk++;
  else console.log("UI mismatch:", t.id, title);
}
console.log(`UI clicks: ${uiOk}/${chosen.length} opened the correct card`);
await page.select("#mode", "2");
await new Promise((r) => setTimeout(r, 500));
await page.screenshot({ path: "tests/viewer_recession.png" });
console.log("page errors:", errors.length ? errors : "none");
const pass = res.tested > 0 && res.ok / res.tested >= 0.95 && errors.length === 0 &&
  uiOk === chosen.length;
console.log(pass ? "PASS" : "FAIL");
await browser.close();
process.exit(pass ? 0 : 1);

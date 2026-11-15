// Rasterise SVG files to PNG at 2x with the installed Chrome (no browser download).
// Usage: node tests/svg_to_png.mjs <file.svg>... [--chrome <path>]
import { readFileSync } from "node:fs";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const ci = args.indexOf("--chrome");
const chrome = ci >= 0 ? args.splice(ci, 2)[1] : "C:/Program Files/Google/Chrome/Application/chrome.exe";
const browser = await puppeteer.launch({ executablePath: chrome, headless: true });
const page = await browser.newPage();
for (const file of args) {
  const svg = readFileSync(file, "utf-8");
  const w = Number(/width="(\d+)"/.exec(svg)[1]);
  const h = Number(/height="(\d+)"/.exec(svg)[1]);
  await page.setViewport({ width: w, height: h, deviceScaleFactor: 2 });
  await page.setContent(`<html><body style="margin:0">${svg}</body></html>`);
  const out = file.replace(/\.svg$/, ".png");
  await page.screenshot({ path: out, clip: { x: 0, y: 0, width: w, height: h } });
  console.log(out);
}
await browser.close();

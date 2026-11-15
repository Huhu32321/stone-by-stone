// Stone by Stone viewer: point cloud with per-stone colouring, ID-buffer picking and a
// stone card. Data layout is documented in src/stonetwin/stages/export_viewer.py.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const ATTRS = [
  ["face_area", "Face area", "area"],
  ["width", "Width", "len"],
  ["height", "Height", "len"],
  ["aspect_ratio", "Aspect ratio", ""],
  ["roughness_rms", "Roughness RMS (proxy)", "len"],
  ["recession", "Recession (proxy)", "len"],
  ["colour_delta_e", "Colour ΔE00 (proxy)", ""],
  ["colour_std", "L* std", ""],
];

const VERT = /* glsl */ `
  attribute vec3 color;
  attribute float stoneId;
  uniform float uSize;
  uniform float uScale;
  varying vec3 vColor;
  varying float vId;
  void main() {
    vColor = color;
    vId = stoneId;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = max(1.0, uSize * uScale / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const FRAG = /* glsl */ `
  uniform int uMode;          // 0 rgb, 1 stone id, 2 attribute, 3 picking
  uniform sampler2D uValues;  // per-stone value in [0,1], negative = missing
  uniform float uNumIds;
  uniform float uSelected;
  varying vec3 vColor;
  varying float vId;
  vec3 viridis(float t) {
    const vec3 c0 = vec3(0.2777, 0.0054, 0.3341), c1 = vec3(0.1051, 1.4046, 1.3846);
    const vec3 c2 = vec3(-0.3309, 0.2148, 0.0951), c3 = vec3(-4.6342, -5.7991, -19.3324);
    const vec3 c4 = vec3(6.2283, 14.1799, 56.6906), c5 = vec3(4.7764, -13.7451, -65.3530);
    const vec3 c6 = vec3(-5.4355, 4.6459, 26.3124);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
  }
  vec3 hashColor(float id) {
    vec3 h = fract(sin(vec3(id * 12.9898, id * 78.233, id * 37.719)) * 43758.5453);
    return 0.25 + 0.75 * h;
  }
  void main() {
    float id = floor(vId + 0.5);
    if (uMode == 3) {
      if (id < 0.5) discard;
      gl_FragColor = vec4(mod(id, 256.0) / 255.0, floor(id / 256.0) / 255.0, 1.0, 1.0);
      return;
    }
    vec3 c = vColor;
    if (uMode == 1) {
      c = id < 0.5 ? vColor * 0.35 : hashColor(id);
    } else if (uMode == 2) {
      float v = id < 0.5 ? -1.0 : texture2D(uValues, vec2((id + 0.5) / uNumIds, 0.5)).r;
      c = v < 0.0 ? vColor * 0.3 : viridis(clamp(v, 0.0, 1.0));
    }
    if (uSelected > 0.5 && abs(id - uSelected) < 0.5) c = mix(c, vec3(1.0, 0.85, 0.1), 0.6);
    gl_FragColor = vec4(c, 1.0);
  }`;

const RELATIVE = " Lengths are relative (scene units): the capture has no scale reference.";
const el = (id) => document.getElementById(id);
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
el("app").appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const pickTarget = new THREE.WebGLRenderTarget(1, 1);
const pickBuf = new Uint8Array(4);

const uniforms = {
  uSize: { value: 1.5 },
  uScale: { value: 1 },
  uMode: { value: 0 },
  uValues: { value: null },
  uNumIds: { value: 1 },
  uSelected: { value: 0 },
};
const material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms });
let points = null;
let state = null; // { meta, stones, byId }

function themeBackground() {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  scene.background = new THREE.Color(dark ? 0x161615 : 0xf4f3ef);
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  uniforms.uScale.value = h * renderer.getPixelRatio() * 0.02;
}

function unitLabel(kind) {
  const u = "scene units";
  return kind === "area" ? `${u}²` : kind === "len" ? u : "";
}

function fmt(v, kind) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  const s = a !== 0 && (a < 0.01 || a >= 1e4) ? v.toExponential(2) : v.toFixed(3);
  return kind ? `${s} ${unitLabel(kind)}` : s;
}

function setAttribute() {
  const key = el("attr").value;
  const kind = ATTRS.find((a) => a[0] === key)[2];
  const vals = state.stones.map((s) => s[key]).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const q = (p) => vals[Math.min(vals.length - 1, Math.floor(p * (vals.length - 1)))];
  const lo = q(0.02), hi = q(0.98);
  const n = state.maxId + 1;
  const data = new Float32Array(n).fill(-1);
  for (const s of state.stones) {
    const v = s[key];
    if (Number.isFinite(v)) data[s.id] = hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0.5;
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RedFormat, THREE.FloatType);
  tex.needsUpdate = true;
  uniforms.uValues.value?.dispose();
  uniforms.uValues.value = tex;
  uniforms.uNumIds.value = n;
  el("legend-min").textContent = fmt(lo, kind);
  el("legend-max").textContent = fmt(hi, kind);
  el("legend-bar").style.background =
    "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)";
}

function updateMode() {
  const mode = Number(el("mode").value);
  uniforms.uMode.value = mode;
  el("attr-row").hidden = mode !== 2;
  el("legend").hidden = mode !== 2;
  if (mode === 2) setAttribute();
}

function showCard(id) {
  const s = state?.byId.get(id);
  uniforms.uSelected.value = s ? id : 0;
  const card = el("card");
  if (!s) { card.hidden = true; return; }
  const c = s.centroid_wall;
  const row = (k, v, cls = "") => `<tr class="${cls}"><td>${k}</td><td>${v}</td></tr>`;
  card.innerHTML = `
    <button id="close" aria-label="Close">×</button>
    <h2>Stone ${s.id}</h2>
    <table>
      ${row("Width × height", `${fmt(s.width, "len")} × ${fmt(s.height, "len")}`)}
      ${row("Face area", fmt(s.face_area, "area"))}
      ${row("Aspect ratio", fmt(s.aspect_ratio, ""))}
      ${row("Supporting points", s.n_points.toLocaleString())}
      ${row("Centroid (wall x, y, z)", c.map((v) => v.toFixed(3)).join(", "))}
      ${row("Roughness RMS", fmt(s.roughness_rms, "len"), "proxy")}
      ${row("Recession", fmt(s.recession, "len"), "proxy")}
      ${row("Colour ΔE00 to wall median", fmt(s.colour_delta_e, ""), "proxy")}
      ${row("Mean L*, a*, b*", s.colour_lab.map((v) => v.toFixed(1)).join(", "))}
    </table>
    <p class="muted small">† Condition proxies: simple geometric and colour measurements,
    not validated deterioration classifications. Recession is negative when the stone
    is set back from its neighbours.${RELATIVE}</p>`;
  card.hidden = false;
  el("close").onclick = () => showCard(0);
}

function pick(clientX, clientY) {
  if (!points) return 0;
  const dpr = renderer.getPixelRatio();
  const w = renderer.domElement.width, h = renderer.domElement.height;
  camera.setViewOffset(w, h, Math.round(clientX * dpr), Math.round(clientY * dpr), 1, 1);
  const prevMode = uniforms.uMode.value, prevBg = scene.background;
  uniforms.uMode.value = 3;
  scene.background = new THREE.Color(0, 0, 0);
  renderer.setRenderTarget(pickTarget);
  renderer.render(scene, camera);
  renderer.readRenderTargetPixels(pickTarget, 0, 0, 1, 1, pickBuf);
  renderer.setRenderTarget(null);
  camera.clearViewOffset();
  uniforms.uMode.value = prevMode;
  scene.background = prevBg;
  return pickBuf[2] > 0 ? pickBuf[0] + 256 * pickBuf[1] : 0;
}

async function loadScene(name) {
  el("status").textContent = `Loading ${name}…`;
  showCard(0);
  const base = `./data/${name}/`;
  const [meta, sdoc, buf] = await Promise.all([
    fetch(base + "meta.json").then((r) => r.json()),
    fetch(base + "stones.json").then((r) => r.json()),
    fetch(base + "points.bin").then((r) => r.arrayBuffer()),
  ]);
  const n = meta.n_points;
  const q = new Uint16Array(buf, 0, n * 3);
  const rgb = new Uint8Array(buf, n * 6, n * 3);
  const ids = new Uint16Array(buf.slice(n * 9, n * 11));
  const lo = meta.bbox_min, hi = meta.bbox_max;
  const centre = lo.map((v, i) => (v + hi[i]) / 2);
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) {
    const a = i % 3;
    pos[i] = lo[a] + (q[i] / 65535) * (hi[a] - lo[a]) - centre[a];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(rgb, 3, true));
  g.setAttribute("stoneId", new THREE.BufferAttribute(new Float32Array(ids), 1));
  if (points) { scene.remove(points); points.geometry.dispose(); }
  points = new THREE.Points(g, material);
  scene.add(points);

  const byId = new Map(sdoc.stones.map((s) => [s.id, s]));
  state = { meta, stones: sdoc.stones, byId, maxId: Math.max(0, ...byId.keys()) };
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  camera.near = span / 1000;
  camera.far = span * 20;
  camera.position.set(0, 0, span * 1.2);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
  updateMode();
  el("status").textContent =
    `${n.toLocaleString()} points, ${sdoc.stones.length} stones (${meta.method})` +
    " · relative units";
}

async function init() {
  themeBackground();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeBackground);
  resize();
  window.addEventListener("resize", resize);
  for (const [k, label] of ATTRS) el("attr").add(new Option(label, k));
  el("attr").value = "recession";
  el("mode").onchange = updateMode;
  el("attr").onchange = setAttribute;
  el("size").oninput = (e) => (uniforms.uSize.value = Number(e.target.value));
  let down = null;
  renderer.domElement.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    showCard(pick(e.clientX, e.clientY));
  });
  const scenes = await fetch("./data/scenes.json").then((r) => r.json());
  for (const s of scenes) el("scene").add(new Option(s, s));
  el("scene").onchange = (e) => loadScene(e.target.value);
  await loadScene(scenes[0]);
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

init().catch((err) => {
  el("status").textContent = `Failed to load: ${err.message}`;
  console.error(err);
});

// Test hook: lets an automated check pick at the screen position of known stone points.
window.__stonetwin = {
  pick,
  // For up to nStones stones, the stone point nearest the stone's mean position (an
  // interior point), as screen coordinates. Returns [{id, x, y}].
  stoneTargets(nStones = 60) {
    const pos = points.geometry.getAttribute("position");
    const sid = points.geometry.getAttribute("stoneId");
    const acc = new Map();
    for (let i = 0; i < sid.count; i++) {
      const id = sid.getX(i);
      if (id === 0) continue;
      const a = acc.get(id) ?? { n: 0, x: 0, y: 0, z: 0, idx: [] };
      a.n++; a.x += pos.getX(i); a.y += pos.getY(i); a.z += pos.getZ(i); a.idx.push(i);
      acc.set(id, a);
    }
    const rect = renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector3();
    const out = [];
    for (const [id, a] of [...acc].slice(0, nStones)) {
      const m = [a.x / a.n, a.y / a.n, a.z / a.n];
      let best = a.idx[0], bd = Infinity;
      for (const i of a.idx) {
        const d = (pos.getX(i) - m[0]) ** 2 + (pos.getY(i) - m[1]) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      v.fromBufferAttribute(pos, best).project(camera);
      out.push({ id, x: (v.x * 0.5 + 0.5) * rect.width, y: (-v.y * 0.5 + 0.5) * rect.height });
    }
    return out;
  },
  sampleCheck(nStones = 60) {
    const rect = renderer.domElement.getBoundingClientRect();
    let ok = 0, offscreen = 0;
    const wrong = [];
    for (const t of this.stoneTargets(nStones)) {
      if (t.x < 0 || t.y < 0 || t.x >= rect.width || t.y >= rect.height) { offscreen++; continue; }
      const got = pick(t.x, t.y);
      if (got === t.id) ok++; else wrong.push([t.id, got]);
    }
    return { tested: ok + wrong.length, ok, offscreen, wrong: wrong.slice(0, 10) };
  },
};

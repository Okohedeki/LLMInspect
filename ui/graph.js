// Weight graph: the model as a zoomable node-link diagram on an infinite canvas.
//
// Layers run left to right along the residual stream (the horizontal bus). Each layer has a pod of
// attention heads above the bus and a pod of MLP neuron clusters below it. Nodes glow with activity
// at the current token; thin edges carry each active unit's write into the bus; curved edges join
// units whose weights compose (a later unit's inputs read an earlier unit's outputs). Zooming in
// reveals labels, then each node turns into a card with a thumbnail of its output weights.

const COL = 96;           // world units between layer columns
const GAP = 20;           // node grid spacing inside a pod
const POD_COLS = 3;
const POD_MARGIN = 30;    // bus to the first row of a pod
const R_HEAD = 6.5;
const MIN_S = 0.05, MAX_S = 18;
export const LOD = { nodes: 0.5, labels: 2.4, cards: 4.6, cardsHi: 9 };

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function bez(x0, y0, x3, y3, t) {
  const dx = (x3 - x0) * 0.5, x1 = x0 + dx, x2 = x3 - dx, u = 1 - t;
  return [u * u * u * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * x3,
    u * u * u * y0 + 3 * u * u * t * y0 + 3 * u * t * t * y3 + t * t * t * y3];
}

export class WeightGraph {
  constructor(canvas, minimap, cb) {
    this.cv = canvas; this.mm = minimap; this.cb = cb;
    this.cam = { x: 0, y: 0, s: 0.5 }; this.target = { ...this.cam };
    this.atlas = null; this.U = 0;
    this.heat = null; this.trail = null; this.trailOn = true; this.mode = "rel";
    this.selected = null; this.hover = null; this.edited = new Set();
    this.edges = null; this.edgeMode = "circuit"; this.drawnEdges = [];
    this.thumbs = new Map(); this.C = null; this.live = false;
    this.raf = 0; this.last = performance.now(); this.W = 1; this.H = 1; this.dpr = 1;
    this.selection = new Set();   // multi-selected units (shift-click, box select)
    this.paths = [];              // [{units: [g...], score, on, focus}] highlighted by the path explorer
    this.signed = null;           // Float32Array in [-1, 1]: attribution mode colors nodes on a diverging ramp
    this.expanded = new Map();    // cluster g -> Float32Array of its neurons' activity in [-1, 1]
    this.box = null;              // live box-select rectangle in screen space
    this.trace = null;            // {start, end}: the input and output token of a traced word, labelled at the ends
    this.insets = { l: 0, r: 0 }; // screen pixels covered by side panels, kept clear when framing
    this._input();
    new ResizeObserver(() => { const fitAgain = this.fitted; this._size(); if (fitAgain) this.fit(true); this.invalidate(); }).observe(canvas.parentElement);
  }

  // ---------------------------------------------------------------- data
  setColors(C) { this.C = C; this._sprite = null; this.invalidate(); }

  setAtlas(atlas) {
    this.atlas = atlas; this.U = atlas.n_units;
    const U = this.U;
    this.x = new Float32Array(U); this.y = new Float32Array(U); this.r = new Float32Array(U);
    this.layerOf = new Uint16Array(U); this.kind = new Uint8Array(U);
    const maxSize = Math.max(1, ...atlas.units.filter((u) => u.kind === "cluster").map((u) => u.size));
    let headRows = 1, clusterRows = 1;
    this.pods = [];
    for (const l of atlas.layers) {
      const cx = l.layer * COL, hr = Math.ceil(l.heads / POD_COLS), cr = Math.ceil(l.clusters / POD_COLS);
      headRows = Math.max(headRows, hr); clusterRows = Math.max(clusterRows, cr);
      for (let i = 0; i < l.heads + l.clusters; i++) {
        const g = l.offset + i, head = i < l.heads, j = head ? i : i - l.heads;
        this.x[g] = cx + ((j % POD_COLS) - (POD_COLS - 1) / 2) * GAP;
        this.y[g] = head ? -POD_MARGIN - Math.floor(j / POD_COLS) * GAP : POD_MARGIN + Math.floor(j / POD_COLS) * GAP;
        this.r[g] = head ? R_HEAD : 3.4 + 5.2 * Math.sqrt(atlas.units[g].size / maxSize);
        this.layerOf[g] = l.layer; this.kind[g] = head ? 0 : 1;
      }
      const half = ((POD_COLS - 1) / 2) * GAP + 11;
      this.pods.push({
        layer: l.layer, x: cx, type: l.type,
        heads: l.heads ? { x0: cx - half, x1: cx + half, y0: -POD_MARGIN - (hr - 1) * GAP - 11, y1: -POD_MARGIN + 11, n: l.heads } : null,
        mlp: l.clusters ? { x0: cx - half, x1: cx + half, y0: POD_MARGIN - 11, y1: POD_MARGIN + (cr - 1) * GAP + 11, n: l.clusters } : null,
      });
    }
    const n = atlas.layers.length;
    this.ends = [{ id: "embed", x: -COL, label: "Embedding", sub: atlas.embed?.shape }, { id: "unembed", x: n * COL, label: "Unembedding", sub: atlas.unembed?.shape }];
    this.top = -POD_MARGIN - (headRows - 1) * GAP - 46;
    this.bounds = { x0: -COL - 48, x1: n * COL + 48, y0: this.top - 8, y1: POD_MARGIN + (clusterRows - 1) * GAP + 30 };
    this._size();
    this.fit(true);
  }

  setEdges(e) { this.edges = e && e.available ? e : null; this._edgesDirty = true; this.invalidate(); }

  // heat: Float32Array of 0..1 per unit (NaN = no data); trail likewise
  setState(s) { Object.assign(this, s); this._edgesDirty = true; this.invalidate(); }
  setThumb(key, bmp) { const old = this.thumbs.get(key); if (old && old !== bmp) old.close?.(); this.thumbs.set(key, bmp); this.invalidate(); }
  clearThumbs() { for (const b of this.thumbs.values()) b.close?.(); this.thumbs.clear(); this.invalidate(); }

  // ---------------------------------------------------------------- camera
  _size() {
    const dpr = window.devicePixelRatio || 1, W = this.cv.clientWidth, H = this.cv.clientHeight;
    this.dpr = dpr;
    if (!W || !H) return;
    this.W = W; this.H = H;
    if (this.cv.width !== Math.round(W * dpr) || this.cv.height !== Math.round(H * dpr)) { this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr); }
    this.dpr = dpr;
  }

  fit(now = false) {
    if (!this.bounds) return;
    const b = this.bounds, s = clamp(Math.min(this.W / (b.x1 - b.x0), (this.H - 40) / (b.y1 - b.y0)) * 0.94, MIN_S, MAX_S);
    this.target = { x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2, s };
    if (now) this.cam = { ...this.target };
    this.fitted = true;
    this.invalidate();
  }

  zoomAt(sx, sy, factor) {
    const t = this.target, s2 = clamp(t.s * factor, MIN_S, MAX_S);
    const wx = (sx - this.W / 2) / t.s + t.x, wy = (sy - this.H / 2) / t.s + t.y;
    this.target = { x: wx - (sx - this.W / 2) / s2, y: wy - (sy - this.H / 2) / s2, s: s2 };
    this.fitted = false;
    this.invalidate();
  }

  flyTo(g, scale = 6) {
    if (g == null || !this.x) return;
    this.target = { x: this.x[g], y: this.y[g], s: Math.max(this.target.s, scale) };
    this.fitted = false;
    this.invalidate();
  }

  toWorld(sx, sy) { return [(sx - this.W / 2) / this.cam.s + this.cam.x, (sy - this.H / 2) / this.cam.s + this.cam.y]; }
  sx(wx) { return (wx - this.cam.x) * this.cam.s + this.W / 2; }
  sy(wy) { return (wy - this.cam.y) * this.cam.s + this.H / 2; }
  zoomPercent() { return Math.round(this.target.s * 100); }

  // ---------------------------------------------------------------- frame loop
  invalidate() { if (!this.raf) this.raf = requestAnimationFrame((t) => this._frame(t)); }

  _frame(now) {
    this.raf = 0;
    if (this.visible === false) return;  // hidden view: stop the self-sustaining loop; setState/activate re-kick it
    const dt = Math.min(64, now - this.last); this.last = now;
    const c = this.cam, t = this.target, k = reduceMotion() ? 1 : 1 - Math.exp(-dt / 70);
    c.x = lerp(c.x, t.x, k); c.y = lerp(c.y, t.y, k); c.s = Math.exp(lerp(Math.log(c.s), Math.log(t.s), k));
    const moving = Math.abs(c.s - t.s) / t.s > 1e-3 || Math.abs(c.x - t.x) * c.s > 0.3 || Math.abs(c.y - t.y) * c.s > 0.3;
    if (!moving) Object.assign(c, t);
    this._draw(now);
    this._drawMinimap();
    this.cb.onFrame?.();
    const animateEdges = (this.drawnEdges.length || this.paths.some((p) => p.on)) && !reduceMotion() && this.visible !== false;
    if (moving || animateEdges || (this.live && !reduceMotion())) this.invalidate();
  }

  // ---------------------------------------------------------------- edges to show
  _pickEdges() {
    this._edgesDirty = false;
    const out = [], e = this.edges;
    // Highlighted paths replace the generic edges unless a single unit is open.
    const pathsOn = this.paths.some((p) => p.on);
    if (!e || this.edgeMode === "off" || !this.x || (pathsOn && this.selected == null)) { this.drawnEdges = out; return; }
    const k = e.k, norm = (x) => clamp(x / (e.p99 || 1), 0, 1.4);
    const sel = this.selected;
    if (sel != null) {
      for (let j = 0; j < Math.min(k, 10); j++) {
        const a = e.up[sel * k + j], x = e.up_x[sel * k + j];
        if (x > 0) out.push({ a, b: sel, x, w: norm(x), dir: "in" });
        const b = e.down[sel * k + j], y = e.down_x[sel * k + j];
        if (y > 0) out.push({ a: sel, b, x: y, w: norm(y), dir: "out" });
      }
    } else if (this.edgeMode === "circuit" && this.heat && this.mode !== "norm") {
      const h = this.heat;
      for (let b = 0; b < this.U; b++) {
        const hb = h[b]; if (!(hb >= 0.5)) continue;
        for (let j = 0; j < k; j++) {
          const a = e.up[b * k + j], x = e.up_x[b * k + j];
          if (!(x > 0)) break;
          const ha = h[a]; if (!(ha >= 0.5)) continue;
          out.push({ a, b, x, w: norm(x) * ha * hb, dir: "circuit" });
        }
      }
      out.sort((p, q) => q.w - p.w);
      out.length = Math.min(out.length, 140);
    } else if (this.edgeMode === "selected" && this.hover != null) {
      const g = this.hover;
      for (let j = 0; j < Math.min(k, 6); j++) {
        const a = e.up[g * k + j], x = e.up_x[g * k + j]; if (x > 0) out.push({ a, b: g, x, w: norm(x) * 0.7, dir: "in" });
        const b = e.down[g * k + j], y = e.down_x[g * k + j]; if (y > 0) out.push({ a: g, b, x: y, w: norm(y) * 0.7, dir: "out" });
      }
    }
    this.drawnEdges = out;
  }

  // ---------------------------------------------------------------- drawing
  _heatColor(v) {
    const ramp = this.C.graphRamp;
    if (!(v >= 0)) return this.C.nodeNoData;
    return ramp[Math.round(clamp(v, 0, 1) * 255)];
  }

  _signedColor(v) {
    if (!(v >= -1)) return this.C.nodeNoData;
    return this.C.attrRamp[Math.round((clamp(v, -1, 1) + 1) * 127.5)];
  }

  _nodeColor(g, v) {
    return this.signed ? this._signedColor(this.signed[g]) : this._heatColor(v);
  }

  /** Camera to frame a set of units (path explorer "show"). */
  fitUnits(units) {
    if (!units?.length || !this.x) return;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const g of units) { x0 = Math.min(x0, this.x[g]); x1 = Math.max(x1, this.x[g]); y0 = Math.min(y0, this.y[g]); y1 = Math.max(y1, this.y[g]); }
    const pad = 70, s = clamp(Math.min(this.W / (x1 - x0 + 2 * pad), (this.H - 60) / (y1 - y0 + 2 * pad)), MIN_S, 3.2);
    this.target = { x: (x0 + x1) / 2, y: (y0 + y1) / 2, s };
    this.fitted = false;
    this.invalidate();
  }

  /** Camera to the embedding (0) or the unembedding (1) node. */
  flyToEnd(i) {
    const end = this.ends?.[i]; if (!end) return;
    this.target = { x: end.x, y: 0, s: Math.max(this.target.s, 1.4) };
    this.fitted = false;
    this.invalidate();
  }

  /** Frame a trace from end to end: both end nodes and the units on its routes, in the part of the
   * canvas no side panel covers. */
  fitTrace(units) {
    if (!this.x) return;
    let y0 = -50, y1 = 50;  // the residual stream and the end nodes
    for (const g of units || []) { y0 = Math.min(y0, this.y[g]); y1 = Math.max(y1, this.y[g]); }
    const x0 = this.ends[0].x, x1 = this.ends[1].x, { l = 0, r = 0 } = this.insets || {};
    // screen margins wide enough for the START and END labels above the end nodes
    const w = Math.max(this.W - l - r, 200), margin = 72;
    const s = clamp(Math.min((w - 2 * margin) / (x1 - x0), (this.H - 150) / (y1 - y0 + 140)), MIN_S, 3.2);
    // the content's centre lands in the middle of the uncovered part, a little below the toolbar
    this.target = { x: (x0 + x1) / 2 - (l - r) / 2 / s, y: (y0 + y1) / 2 - 24 / s, s };
    this.fitted = false;
    this.invalidate();
  }

  flyToLayer(layer) {
    const p = this.pods?.[layer]; if (!p) return;
    this.target = { x: p.x, y: (this.bounds.y0 + this.bounds.y1) / 2, s: Math.max(this.target.s, 1.4) };
    this.fitted = false;
    this.invalidate();
  }

  _glowSprite() {
    if (this._sprite) return this._sprite;
    const s = 64, cv = new OffscreenCanvas(s, s), g = cv.getContext("2d");
    const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grad.addColorStop(0, this.C.glow + "cc"); grad.addColorStop(0.35, this.C.glow + "55"); grad.addColorStop(1, this.C.glow + "00");
    g.fillStyle = grad; g.fillRect(0, 0, s, s);
    this._sprite = cv;
    return cv;
  }

  _draw(now) {
    if (!this.C || !this.x) return;
    this._size();
    if (this._edgesDirty) this._pickEdges();
    const ctx = this.cv.getContext("2d"), C = this.C, s = this.cam.s, W = this.W, H = this.H;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const [wx0, wy0] = this.toWorld(0, 0), [wx1, wy1] = this.toWorld(W, H);
    const inView = (x, y, m) => x > wx0 - m && x < wx1 + m && y > wy0 - m && y < wy1 + m;
    const X = (x) => this.sx(x), Y = (y) => this.sy(y);

    // dot grid: an infinite-canvas cue that also shows the zoom
    const step = 40 * Math.pow(2, Math.max(0, Math.ceil(Math.log2(14 / (40 * s)))));
    if (step * s >= 14) {
      ctx.fillStyle = C.gridDot;
      for (let gx = Math.floor(wx0 / step) * step; gx < wx1; gx += step)
        for (let gy = Math.floor(wy0 / step) * step; gy < wy1; gy += step) ctx.fillRect(X(gx) - 0.75, Y(gy) - 0.75, 1.5, 1.5);
    }

    // pods
    const podAlpha = clamp((s - 0.25) / 0.4, 0, 1);
    ctx.lineWidth = 1;
    for (const p of this.pods) {
      if (p.x < wx0 - COL || p.x > wx1 + COL) continue;
      for (const pod of [p.heads, p.mlp]) {
        if (!pod) continue;
        const x = X(pod.x0), y = Y(pod.y0), w = (pod.x1 - pod.x0) * s, h = (pod.y1 - pod.y0) * s;
        ctx.globalAlpha = 0.55 + 0.45 * podAlpha;
        ctx.fillStyle = C.podFill; roundRect(ctx, x, y, w, h, Math.min(12, 8 * s)); ctx.fill();
        ctx.strokeStyle = C.podStroke; ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // the residual stream
    const bx0 = X(this.ends[0].x), bx1 = X(this.ends[1].x), by = Y(0), bw = clamp(3.2 * s, 1.5, 7);
    const grad = ctx.createLinearGradient(bx0, 0, bx1, 0);
    grad.addColorStop(0, C.busA); grad.addColorStop(1, C.busB);
    ctx.strokeStyle = grad; ctx.lineWidth = bw; ctx.lineCap = "round";
    ctx.globalAlpha = 0.9; ctx.beginPath(); ctx.moveTo(bx0, by); ctx.lineTo(bx1, by); ctx.stroke();
    if (this.live && !reduceMotion()) {
      ctx.setLineDash([bw * 1.5, bw * 5]); ctx.lineDashOffset = -(now / 18) % (bw * 6.5);
      ctx.strokeStyle = C.busFlow; ctx.lineWidth = bw * 0.55; ctx.beginPath(); ctx.moveTo(bx0, by); ctx.lineTo(bx1, by); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;

    const h = this.heat, U = this.U, showNodes = s >= LOD.nodes * 0.6;
    // write edges: active units feed the bus at their layer
    if (h && this.mode !== "norm" && s >= LOD.nodes) {
      ctx.lineWidth = clamp(0.9 * s, 0.6, 2);
      for (let g = 0; g < U; g++) {
        const v = h[g]; if (!(v >= 0.35)) continue;
        const x = this.x[g], y = this.y[g]; if (!inView(x, y, 60)) continue;
        const bxw = this.pods[this.layerOf[g]].x;
        ctx.strokeStyle = C.writeEdge; ctx.globalAlpha = clamp((v - 0.3) * 0.9, 0.08, 0.55);
        ctx.beginPath(); ctx.moveTo(X(x), Y(y)); ctx.quadraticCurveTo(X(x), Y(y * 0.35), X(bxw + (x - bxw) * 0.2), by); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // composition edges
    const edges = this.drawnEdges;
    if (edges.length) {
      const dim = this.selected != null ? 1 : 0.9;
      for (const e of edges) {
        const x0 = X(this.x[e.a]), y0 = Y(this.y[e.a]), x3 = X(this.x[e.b]), y3 = Y(this.y[e.b]);
        if (Math.max(x0, x3) < -50 || Math.min(x0, x3) > W + 50) continue;
        const col = e.dir === "in" ? C.edgeIn : e.dir === "out" ? C.edgeOut : C.edgeCircuit;
        const dx = (x3 - x0) * 0.5;
        ctx.strokeStyle = col; ctx.globalAlpha = clamp(0.18 + 0.6 * e.w, 0.15, 0.85) * dim;
        ctx.lineWidth = clamp((0.6 + 2.6 * e.w) * Math.sqrt(s), 0.6, 6);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.bezierCurveTo(x0 + dx, y0, x3 - dx, y3, x3, y3); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      // particles flow from source to target
      if (!reduceMotion() && edges.length <= 220) {
        const speed = 0.00032;
        for (let i = 0; i < edges.length; i++) {
          const e = edges[i], x0 = X(this.x[e.a]), y0 = Y(this.y[e.a]), x3 = X(this.x[e.b]), y3 = Y(this.y[e.b]);
          if (Math.max(x0, x3) < 0 || Math.min(x0, x3) > W) continue;
          ctx.fillStyle = e.dir === "in" ? C.edgeIn : e.dir === "out" ? C.edgeOut : C.edgeCircuitHot;
          for (let p = 0; p < 2; p++) {
            const t = ((now * speed * (0.7 + e.w)) + p / 2 + i * 0.137) % 1, [px, py] = bez(x0, y0, x3, y3, t);
            ctx.globalAlpha = 0.9 * Math.sin(Math.PI * t);
            ctx.beginPath(); ctx.arc(px, py, clamp(1.1 + 1.6 * e.w, 1, 3.2), 0, Math.PI * 2); ctx.fill();
          }
        }
        ctx.globalAlpha = 1;
      }
      if (s >= 1.4 && this.selected != null) {  // strengths on the selected unit's edges
        ctx.font = "500 10px Inter, system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        for (const e of edges) {
          const [mx, my] = bez(X(this.x[e.a]), Y(this.y[e.a]), X(this.x[e.b]), Y(this.y[e.b]), 0.5);
          if (mx < 0 || mx > W) continue;
          const label = `${e.x.toFixed(1)}×`, tw = ctx.measureText(label).width + 8;
          ctx.fillStyle = C.chipBg; roundRect(ctx, mx - tw / 2, my - 8, tw, 16, 8); ctx.fill();
          ctx.fillStyle = e.dir === "in" ? C.edgeIn : C.edgeOut; ctx.fillText(label, mx, my + 0.5);
        }
      }
    }

    // highlighted paths (path explorer, or a traced word): amber curves unit to unit, the focused one
    // brightest. A trace's routes also join the ends: dashed from the embedding (the input token's
    // stream, which every unit reads) into the first unit, solid from the last unit into the
    // unembedding (its direct write to the output token).
    const shownPaths = this.paths.filter((p) => p.on);
    const endW = clamp(70 * s, 26, 180);
    const startPt = [X(this.ends[0].x) + endW / 2, Y(0)], endPt = [X(this.ends[1].x) - endW / 2, Y(0)];
    const segments = (p) => {
      const out = [];
      if (p.ends) out.push([startPt[0], startPt[1], X(this.x[p.units[0]]), Y(this.y[p.units[0]]), "start"]);
      for (let i = 0; i + 1 < p.units.length; i++) {
        const a = p.units[i], b = p.units[i + 1];
        out.push([X(this.x[a]), Y(this.y[a]), X(this.x[b]), Y(this.y[b]), "hop"]);
      }
      if (p.ends) { const z = p.units[p.units.length - 1]; out.push([X(this.x[z]), Y(this.y[z]), endPt[0], endPt[1], "end"]); }
      return out;
    };
    if (shownPaths.length) {
      const maxScore = Math.max(...shownPaths.map((p) => p.score), 1e-9);
      const anyFocus = shownPaths.some((p) => p.focus);
      for (const p of [...shownPaths].sort((a, b) => a.focus - b.focus)) {  // the focused route on top
        const w = clamp(p.score / maxScore, 0.15, 1), focus = p.focus;
        ctx.strokeStyle = focus ? C.pathFocus : C.path;
        ctx.globalAlpha = focus ? 0.95 : (anyFocus ? 0.15 : 0.25 + 0.55 * w);
        ctx.lineWidth = clamp((focus ? 2.6 : 1 + 2 * w) * Math.sqrt(s), 0.8, focus ? 7 : 5);
        for (const dashed of [false, true]) {
          ctx.setLineDash(dashed ? [5, 5] : []);
          ctx.beginPath();
          for (const [x0, y0, x3, y3, kind] of segments(p)) {
            if ((kind === "start") !== dashed) continue;
            const dx = (x3 - x0) * 0.5;
            ctx.moveTo(x0, y0); ctx.bezierCurveTo(x0 + dx, y0, x3 - dx, y3, x3, y3);
          }
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      if (!reduceMotion()) {  // particles flow downstream along the strongest paths
        ctx.fillStyle = C.pathFocus;
        shownPaths.slice(0, 40).forEach((p, pi) => {
          if (anyFocus && !p.focus) return;
          segments(p).forEach(([x0, y0, x3, y3], i) => {
            const t = (now * 0.00035 + i * 0.29 + pi * 0.17) % 1, [px, py] = bez(x0, y0, x3, y3, t);
            ctx.globalAlpha = 0.85 * Math.sin(Math.PI * t);
            ctx.beginPath(); ctx.arc(px, py, p.focus ? 2.8 : 1.8, 0, Math.PI * 2); ctx.fill();
          });
        });
        ctx.globalAlpha = 1;
      }
    }

    // glow under hot nodes
    if (h && this.mode !== "norm") {
      const sprite = this._glowSprite();
      ctx.globalCompositeOperation = C.dark ? "lighter" : "source-over";
      for (let g = 0; g < U; g++) {
        const v = h[g]; if (!(v >= 0.45)) continue;
        const x = this.x[g], y = this.y[g]; if (!inView(x, y, 30)) continue;
        const rr = Math.max(6, this.r[g] * s * (2.4 + 2 * v));
        ctx.globalAlpha = clamp((v - 0.35) * 1.3, 0, 1) * (C.dark ? 0.9 : 0.45);
        ctx.drawImage(sprite, X(x) - rr, Y(y) - rr, rr * 2, rr * 2);
      }
      ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
    }

    // nodes (or cards when zoomed far in)
    const cards = s >= LOD.cards;
    if (showNodes || !cards) {
      for (let g = 0; g < U; g++) {
        const x = this.x[g], y = this.y[g]; if (!inView(x, y, 20)) continue;
        const v = this.mode === "norm" ? this.normT?.[g] : h ? h[g] : NaN;
        const rs = Math.max(this.r[g] * s, s < LOD.nodes ? 1.1 : 1.6);
        if (cards) { this._card(ctx, g, X(x), Y(y), v); continue; }
        ctx.fillStyle = this._nodeColor(g, v);
        if (rs < 2.2) ctx.fillRect(X(x) - rs, Y(y) - rs, rs * 2, rs * 2);
        else { ctx.beginPath(); ctx.arc(X(x), Y(y), rs, 0, Math.PI * 2); ctx.fill(); }
        if (this.kind[g] === 0 && rs >= 4) { ctx.strokeStyle = C.headRing; ctx.lineWidth = 1; ctx.stroke(); }
      }
    }

    // trail rings, edits, hover and selection
    if (this.trailOn && this.trail && h && this.mode !== "norm" && !cards) {
      ctx.strokeStyle = C.trail; ctx.lineWidth = clamp(1.2 * s, 0.8, 2);
      for (let g = 0; g < U; g++) {
        const fade = this.trail[g] - (h[g] >= 0 ? h[g] : 0); if (fade < 0.2) continue;
        const x = this.x[g], y = this.y[g]; if (!inView(x, y, 20)) continue;
        ctx.globalAlpha = clamp(fade, 0.2, 1);
        ctx.beginPath(); ctx.arc(X(x), Y(y), Math.max(this.r[g] * s, 1.5) + clamp(2.5 * s, 1.5, 5), 0, Math.PI * 2); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    if (this.edited.size && !cards) {
      ctx.fillStyle = C.edited;
      for (const g of this.edited) {
        const x = this.x[g], y = this.y[g]; if (!inView(x, y, 20)) continue;
        const rs = Math.max(this.r[g] * s, 1.5);
        ctx.beginPath(); ctx.arc(X(x) + rs * 0.75, Y(y) - rs * 0.75, clamp(rs * 0.38, 1.4, 4), 0, Math.PI * 2); ctx.fill();
      }
    }
    const ring = (g, color, width, pad) => {
      if (g == null) return;
      const rs = cards ? this._cardSize() / 2 : Math.max(this.r[g] * s, 2);
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath();
      if (cards) roundRect(ctx, X(this.x[g]) - rs - pad, Y(this.y[g]) - rs * 0.8 - pad, rs * 2 + pad * 2, rs * 1.6 + pad * 2, 12 + pad);
      else ctx.arc(X(this.x[g]), Y(this.y[g]), rs + pad, 0, Math.PI * 2);
      ctx.stroke();
    };
    ring(this.hover, C.hoverRing, 1.5, 3);
    if (shownPaths.length) {  // units on highlighted paths
      const onPath = new Set(shownPaths.flatMap((p) => p.units));
      for (const g of onPath) if (inView(this.x[g], this.y[g], 20)) ring(g, C.path, 1.5, 2);
    }
    for (const g of this.selection) if (inView(this.x[g], this.y[g], 20)) ring(g, C.multiSelect, 2, 4);
    if (this.selected != null) { ring(this.selected, C.selectHalo, 6, 5); ring(this.selected, C.selectRing, 2, 3); }

    // labels
    this._labels(ctx, s, wx0, wx1);

    if (this.box) {  // box-select rectangle
      const b = this.box, x = Math.min(b.x0, b.x1), y = Math.min(b.y0, b.y1), w = Math.abs(b.x1 - b.x0), hh = Math.abs(b.y1 - b.y0);
      ctx.fillStyle = C.multiSelectFill; ctx.fillRect(x, y, w, hh);
      ctx.strokeStyle = C.multiSelect; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.strokeRect(x, y, w, hh); ctx.setLineDash([]);
    }
  }

  /** Units whose centers fall inside a screen-space rectangle. */
  unitsIn(b) {
    const [ax, ay] = this.toWorld(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1));
    const [bx, by] = this.toWorld(Math.max(b.x0, b.x1), Math.max(b.y0, b.y1));
    const out = [];
    for (let g = 0; g < this.U; g++) if (this.x[g] >= ax && this.x[g] <= bx && this.y[g] >= ay && this.y[g] <= by) out.push(g);
    return out;
  }

  _cardSize() { return clamp(GAP * this.cam.s * 0.86, 40, 300); }

  _card(ctx, g, cx, cy, v) {
    const C = this.C, u = this.atlas.units[g], w = this._cardSize(), hh = w * 0.8, x = cx - w / 2, y = cy - hh / 2;
    if (x > this.W || y > this.H || x + w < 0 || y + hh < 0) return;
    const hot = v >= 0 ? v : 0;
    ctx.fillStyle = C.cardBg; roundRect(ctx, x, y, w, hh, 12); ctx.fill();
    ctx.strokeStyle = v >= 0 || this.signed ? this._nodeColor(g, Math.max(0.25, hot)) : C.podStroke; ctx.lineWidth = 1 + 2 * hot; ctx.stroke();
    const pad = Math.max(5, w * 0.05), level = this.cam.s >= LOD.cardsHi ? 1 : 0;
    const th = hh * 0.52, neurons = this.expanded.get(g);
    if (neurons) {
      // An expanded cluster shows every neuron as a cell: activity at the current token, signed.
      const n = neurons.length, aw = w - 2 * pad, cols = Math.max(1, Math.ceil(Math.sqrt(n * aw / th))), rows = Math.ceil(n / cols);
      const cw = aw / cols, ch = th / rows;
      ctx.fillStyle = C.podFill; roundRect(ctx, x + pad, y + pad, aw, th, 7); ctx.fill();
      for (let k = 0; k < n; k++) {
        ctx.fillStyle = this._signedColor(neurons[k]);
        ctx.fillRect(x + pad + (k % cols) * cw, y + pad + Math.floor(k / cols) * ch, Math.max(cw - 0.6, 0.6), Math.max(ch - 0.6, 0.6));
      }
    } else {
      const key = `${g}:${level}`, img = this.thumbs.get(key) || this.thumbs.get(`${g}:0`);
      if (img) { ctx.imageSmoothingEnabled = false; ctx.save(); roundRect(ctx, x + pad, y + pad, w - 2 * pad, th, 7); ctx.clip(); ctx.drawImage(img, x + pad, y + pad, w - 2 * pad, th); ctx.restore(); }
      else { ctx.fillStyle = C.podFill; roundRect(ctx, x + pad, y + pad, w - 2 * pad, th, 7); ctx.fill(); }
      if (!this.thumbs.has(key)) this.cb.needThumb?.(g, level);
    }
    const fs = clamp(w * 0.085, 9, 15);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = C.ink; ctx.font = `600 ${fs}px Inter, system-ui, sans-serif`;
    ctx.fillText(`${u.kind === "head" ? "Head" : "Cluster"} ${u.i}`, x + pad, y + pad + th + fs * 1.35);
    ctx.fillStyle = C.ink2; ctx.font = `${fs * 0.82}px Inter, system-ui, sans-serif`;
    const sub = u.kind === "head" ? `L${u.layer} · ${u.size} dims` : `L${u.layer} · ${u.size} neurons`;
    ctx.fillText(sub, x + pad, y + pad + th + fs * 2.5);
    if (u.pos && w > 90) {
      ctx.fillStyle = C.muted; ctx.font = `${fs * 0.78}px "JetBrains Mono", ui-monospace, monospace`;
      ctx.save(); ctx.beginPath(); ctx.rect(x + pad, y, w - 2 * pad, hh); ctx.clip();
      ctx.fillText(`↑ ${u.pos.map(vis).join(" ")}`, x + pad, y + pad + th + fs * 3.6); ctx.restore();
    }
    if (v >= 0 && this.mode !== "norm") {  // activity bar
      ctx.fillStyle = C.podFill; roundRect(ctx, x + pad, y + hh - pad - 4, w - 2 * pad, 4, 2); ctx.fill();
      ctx.fillStyle = this._heatColor(Math.max(0.3, hot)); roundRect(ctx, x + pad, y + hh - pad - 4, (w - 2 * pad) * hot, 4, 2); ctx.fill();
    }
    if (this.edited.has(g)) { ctx.fillStyle = C.edited; ctx.beginPath(); ctx.arc(x + w - pad - 3, y + pad + 3, 4, 0, Math.PI * 2); ctx.fill(); }
  }

  _labels(ctx, s, wx0, wx1) {
    const C = this.C;
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    // layer numbers above the columns, thinned when zoomed out
    const every = s * COL >= 34 ? 1 : s * COL >= 16 ? 2 : s * COL >= 8 ? 4 : 8;
    ctx.font = `600 ${clamp(10 * Math.sqrt(s) + 3, 10, 15)}px Inter, system-ui, sans-serif`;
    for (const p of this.pods) {
      if (p.layer % every || p.x < wx0 - COL || p.x > wx1 + COL) continue;
      const x = this.sx(p.x), y = this.sy(this.top) + 2;
      ctx.fillStyle = C.ink2; ctx.fillText(`L${p.layer}`, x, y);
      if (s * COL >= 50 && p.type && p.type !== "attention") {
        ctx.font = `500 ${clamp(9 * Math.sqrt(s), 9, 12)}px Inter, system-ui, sans-serif`; ctx.fillStyle = C.muted;
        ctx.fillText(p.type === "linear_attention" ? "linear" : p.type === "full_attention" ? "full attn" : p.type, x, y + clamp(13 * Math.sqrt(s), 12, 18));
        ctx.font = `600 ${clamp(10 * Math.sqrt(s) + 3, 10, 15)}px Inter, system-ui, sans-serif`;
      }
    }
    // pod captions
    if (s >= 1.2 && s < LOD.cards) {
      ctx.font = `500 ${clamp(8 * Math.sqrt(s), 9, 12)}px Inter, system-ui, sans-serif`; ctx.fillStyle = C.muted;
      for (const p of this.pods) {
        if (p.x < wx0 - COL || p.x > wx1 + COL) continue;
        if (p.mlp) ctx.fillText(`MLP · ${p.mlp.n} clusters`, this.sx(p.x), this.sy(p.mlp.y1) + clamp(12 * Math.sqrt(s), 12, 18));
      }
    }
    // node labels
    if (s >= LOD.labels && s < LOD.cards) {
      const fs = clamp(3.4 * s, 9, 12);
      ctx.font = `500 ${fs}px Inter, system-ui, sans-serif`;
      for (let g = 0; g < this.U; g++) {
        const x = this.x[g]; if (x < wx0 - 20 || x > wx1 + 20) continue;
        const u = this.atlas.units[g], sx = this.sx(x), sy = this.sy(this.y[g]) + this.r[g] * s + fs + 2;
        if (sy < -20 || sy > this.H + 20) continue;
        ctx.fillStyle = C.ink2; ctx.fillText(`${u.kind === "head" ? "h" : "c"}${u.i}`, sx, sy);
        if (s >= LOD.labels * 1.35 && u.pos) { ctx.fillStyle = C.muted; ctx.fillText(vis(u.pos[0]), sx, sy + fs + 1); }
      }
    }
    // embedding and unembedding
    for (const end of this.ends) {
      const w = clamp(70 * s, 26, 180), hh = clamp(46 * s, 18, 110), x = this.sx(end.x) - w / 2, y = this.sy(0) - hh / 2;
      if (x > this.W || x + w < 0) continue;
      ctx.fillStyle = C.cardBg; roundRect(ctx, x, y, w, hh, Math.min(12, hh / 3)); ctx.fill();
      ctx.strokeStyle = end.id === "embed" ? C.busA : C.busB; ctx.lineWidth = 1.5; ctx.stroke();
      if (w >= 60) {
        ctx.fillStyle = C.ink; ctx.font = `600 ${clamp(w * 0.11, 9, 14)}px Inter, system-ui, sans-serif`; ctx.textAlign = "center";
        ctx.fillText(end.label, x + w / 2, y + hh / 2 + (end.sub ? -2 : 4));
        if (end.sub) { ctx.fillStyle = C.muted; ctx.font = `${clamp(w * 0.085, 8, 11)}px Inter, system-ui, sans-serif`; ctx.fillText(end.sub.join(" × "), x + w / 2, y + hh / 2 + clamp(w * 0.12, 10, 16)); }
      }
    }
    if (this.trace) this._drawTraceEnds(ctx, s);
  }

  /** A traced word reads left to right: the input token labelled START at the embedding, the output
   * token END at the unembedding, and the focused route's units numbered in order. */
  _drawTraceEnds(ctx, s) {
    const C = this.C, X = (x) => this.sx(x), Y = (y) => this.sy(y);
    ctx.save();
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    const clip = (t) => (t.length > 24 ? `${t.slice(0, 23)}…` : t);
    const label = (end, kicker, text) => {
      const w = clamp(70 * s, 26, 180), hh = clamp(46 * s, 18, 110), cx = X(end.x), nodeTop = Y(0) - hh / 2;
      ctx.strokeStyle = C.path; ctx.lineWidth = 2.5;
      roundRect(ctx, cx - w / 2, nodeTop, w, hh, Math.min(12, hh / 3)); ctx.stroke();
      ctx.font = "600 12.5px 'JetBrains Mono', ui-monospace, monospace";
      const shown = clip(text), cw = Math.max(ctx.measureText(shown).width + 20, 92), ch = 38, y = nodeTop - 8 - ch;
      if (cx > this.W + cw / 2 || cx < -cw / 2) return;  // the end is off screen
      const x = clamp(cx - cw / 2, 4, this.W - cw - 4);  // keep the label readable at the canvas edge
      ctx.fillStyle = C.cardBg; roundRect(ctx, x, y, cw, ch, 9); ctx.fill();
      ctx.strokeStyle = C.path; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = C.path; ctx.font = "700 9.5px Inter, system-ui, sans-serif";
      ctx.fillText(kicker, x + cw / 2, y + 13);
      ctx.fillStyle = C.ink; ctx.font = "600 12.5px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillText(shown, x + cw / 2, y + 29);
    };
    label(this.ends[0], "START · INPUT", this.trace.start);
    label(this.ends[1], "END · OUTPUT", this.trace.end);
    const fp = this.paths.find((p) => p.on && p.focus && p.ends);
    if (fp) {
      ctx.font = "700 10px Inter, system-ui, sans-serif"; ctx.textBaseline = "middle";
      fp.units.forEach((g, i) => {
        const r = Math.max(this.r[g] * s, 3) + 3, x = X(this.x[g]) + r, y = Y(this.y[g]) - r;
        ctx.fillStyle = C.pathFocus; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = C.dark ? "#1a1205" : "#fff"; ctx.fillText(String(i + 1), x, y + 0.5);
      });
    }
    ctx.restore();
  }

  _drawMinimap() {
    const mm = this.mm; if (!mm || !this.pods) return;
    const dpr = window.devicePixelRatio || 1, W = mm.clientWidth, H = mm.clientHeight;
    if (!W) return;
    if (mm.width !== Math.round(W * dpr)) { mm.width = Math.round(W * dpr); mm.height = Math.round(H * dpr); }
    const ctx = mm.getContext("2d"), C = this.C, b = this.bounds;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const pad = 6, sc = Math.min((W - 2 * pad) / (b.x1 - b.x0), (H - 2 * pad) / (b.y1 - b.y0));
    const ox = pad + ((W - 2 * pad) - (b.x1 - b.x0) * sc) / 2, oy = pad + ((H - 2 * pad) - (b.y1 - b.y0) * sc) / 2;
    const MX = (x) => ox + (x - b.x0) * sc, MY = (y) => oy + (y - b.y0) * sc;
    this._mm = { MX, MY, sc, ox, oy };
    ctx.fillStyle = C.busA; ctx.globalAlpha = 0.7; ctx.fillRect(MX(this.ends[0].x), MY(0) - 0.75, (this.ends[1].x - this.ends[0].x) * sc, 1.5); ctx.globalAlpha = 1;
    const h = this.heat;
    for (const p of this.pods) {
      for (const [pod, first, n] of [[p.heads, this.atlas.layers[p.layer].offset, p.heads?.n], [p.mlp, this.atlas.layers[p.layer].offset + (p.heads?.n || 0), p.mlp?.n]]) {
        if (!pod) continue;
        let m = 0;
        if (this.mode === "norm" && this.normT) for (let g = first; g < first + n; g++) m += this.normT[g];
        else if (h) for (let g = first; g < first + n; g++) m += h[g] >= 0 ? h[g] : 0;
        ctx.fillStyle = this._heatColor(clamp((m / n) * 2.2, 0, 1));
        ctx.fillRect(MX(pod.x0), MY(pod.y0), Math.max(1.5, (pod.x1 - pod.x0) * sc), (pod.y1 - pod.y0) * sc);
      }
    }
    const [vx0, vy0] = this.toWorld(0, 0), [vx1, vy1] = this.toWorld(this.W, this.H);
    ctx.strokeStyle = C.selectRing; ctx.lineWidth = 1.5;
    const rx = clamp(MX(vx0), 1, W - 1), ry = clamp(MY(vy0), 1, H - 1);
    ctx.strokeRect(rx, ry, clamp(MX(vx1), 1, W - 1) - rx, clamp(MY(vy1), 1, H - 1) - ry);
  }

  // ---------------------------------------------------------------- hit testing and input
  hit(sx, sy) {
    if (!this.x) return null;
    const [wx, wy] = this.toWorld(sx, sy), s = this.cam.s;
    for (const end of this.ends) {
      const w = clamp(70 * s, 26, 180) / s, hh = clamp(46 * s, 18, 110) / s;
      if (Math.abs(wx - end.x) <= w / 2 && Math.abs(wy) <= hh / 2) return { end: end.id };
    }
    const li = Math.round(wx / COL);
    const l = this.atlas.layers[li]; if (!l) return null;
    const cards = s >= LOD.cards;
    let best = null, bestD = Infinity;
    for (let g = l.offset; g < l.offset + l.heads + l.clusters; g++) {
      const dx = wx - this.x[g], dy = wy - this.y[g];
      const d = cards ? Math.max(Math.abs(dx) / (this._cardSize() / 2 / s), Math.abs(dy) / (this._cardSize() * 0.4 / s)) : Math.hypot(dx, dy) / Math.max(this.r[g] + 3 / s, 6 / s);
      if (d < 1 && d < bestD) { best = g; bestD = d; }
    }
    return best == null ? null : { g: best };
  }

  _input() {
    const cv = this.cv;
    let drag = null;
    cv.addEventListener("pointerdown", (ev) => {
      const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
      // Shift-drag draws a selection box; a plain drag pans.
      drag = { x: ev.clientX, y: ev.clientY, cx: this.target.x, cy: this.target.y, moved: false, box: ev.shiftKey, mx, my };
      cv.setPointerCapture(ev.pointerId);
    });
    cv.addEventListener("pointermove", (ev) => {
      const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
      if (drag) {
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        if (drag.moved && drag.box) {
          this.box = { x0: drag.mx, y0: drag.my, x1: mx, y1: my };
          cv.style.cursor = "crosshair"; this.cb.onHover?.(null); this.invalidate(); return;
        }
        if (drag.moved) {
          this.target.x = drag.cx - dx / this.target.s; this.target.y = drag.cy - dy / this.target.s;
          this.cam.x = this.target.x; this.cam.y = this.target.y; this.fitted = false;
          cv.style.cursor = "grabbing"; this.cb.onHover?.(null); this.invalidate(); return;
        }
      }
      const hit = this.hit(mx, my), g = hit?.g ?? null;
      cv.style.cursor = hit ? "pointer" : "grab";
      if (g !== this.hover) { this.hover = g; if (this.edgeMode === "selected") this._edgesDirty = true; this.invalidate(); }
      this.cb.onHover?.(hit, mx, my);
    });
    cv.addEventListener("pointerleave", () => { this.hover = null; this.cb.onHover?.(null); this.invalidate(); });
    cv.addEventListener("pointerup", (ev) => {
      const d = drag; drag = null; cv.style.cursor = "grab";
      if (d?.moved && d.box && this.box) {
        const units = this.unitsIn(this.box);
        this.box = null; this.invalidate();
        this.cb.onBoxSelect?.(units, ev.altKey);
        return;
      }
      if (d?.moved) return;
      const r = cv.getBoundingClientRect(), hit = this.hit(ev.clientX - r.left, ev.clientY - r.top);
      if (hit?.end) this.cb.onEnd?.(hit.end);
      else this.cb.onSelect?.(hit ? hit.g : null, { shift: ev.shiftKey || ev.metaKey || ev.ctrlKey });
    });
    cv.addEventListener("dblclick", (ev) => {
      const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top, hit = this.hit(mx, my);
      if (hit?.g != null) this.flyTo(hit.g, Math.max(LOD.cards * 1.3, this.target.s * 2)); else this.zoomAt(mx, my, 2.2);
    });
    cv.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
      if (ev.shiftKey && !ev.ctrlKey) { this.target.x += (ev.deltaY || ev.deltaX) / this.target.s; this.fitted = false; this.invalidate(); return; }
      const dy = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;
      this.zoomAt(mx, my, Math.exp(-dy * (ev.ctrlKey ? 0.012 : 0.0018)));
    }, { passive: false });
    if (this.mm) {
      let mmDrag = false;
      const go = (ev) => {
        if (!this._mm) return;
        const r = this.mm.getBoundingClientRect(), { MX, MY, sc } = this._mm;
        const wx = this.bounds.x0 + (ev.clientX - r.left - MX(this.bounds.x0)) / sc, wy = this.bounds.y0 + (ev.clientY - r.top - MY(this.bounds.y0)) / sc;
        this.target.x = wx; this.target.y = wy; this.fitted = false; this.invalidate();
      };
      this.mm.addEventListener("pointerdown", (ev) => { mmDrag = true; this.mm.setPointerCapture(ev.pointerId); go(ev); });
      this.mm.addEventListener("pointermove", (ev) => { if (mmDrag) go(ev); });
      this.mm.addEventListener("pointerup", () => { mmDrag = false; });
    }
  }
}

function vis(s) { return (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/ /g, "·") || "∅"; }

function roundRect(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

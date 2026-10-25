// Path search over the unit graph. Composition edges (each unit's strongest inputs and outputs, from
// /brainscan/anatomy/edges) form a layered graph: an edge a -> b means b's input weights read a's
// output directions, scored in multiples of chance. A path's score is the product of its edges'
// strength above chance, optionally times each unit's activity at the current token, so a path is
// strong only if every hop is.

/** Edge strength above chance: 0 at chance (1x), 1 at or above the model's 99th-percentile edge. */
export function edgeStrength(x, p99) {
  return Math.min(1, Math.max(0, (x - 1) / Math.max(p99 - 1, 1e-6)));
}

/**
 * Beam search for paths from `sources`.
 * opts.dir      "down" (follow outputs, toward later layers) or "up" (follow inputs, toward earlier)
 * opts.hops     maximum edges per path
 * opts.branch   edges tried per unit per hop
 * opts.beam     partial paths kept per hop
 * opts.weight   optional (g) => factor in [0, 1], e.g. activity at the current token
 * opts.targets  optional Set: keep only paths that end on one of these units
 * opts.limit    paths returned
 * Without targets, only complete routes are returned (ones that ran `hops` edges or could not go
 * further), not every prefix of them. Paths are returned in layer order (earliest unit first) with
 * their score and per-hop edge strengths in multiples of chance (xs).
 */
export function findPaths(edges, sources, opts = {}) {
  const { dir = "down", hops = 3, branch = 4, beam = 96, weight = null, targets = null, limit = 60 } = opts;
  if (!edges?.available || !sources?.length) return [];
  const k = edges.k, ids = dir === "down" ? edges.down : edges.up, xs = dir === "down" ? edges.down_x : edges.up_x;
  const w = (g) => (weight ? weight(g) : 1);
  const found = new Map();
  const keep = (p) => {
    if (p.units.length < 2) return;
    const units = dir === "up" ? [...p.units].reverse() : p.units, key = units.join(">");
    if (!found.has(key)) found.set(key, { units, score: p.score, xs: dir === "up" ? [...p.xs].reverse() : p.xs });
  };
  let frontier = sources.map((g) => ({ units: [g], score: 1, xs: [] }));
  for (let hop = 0; hop < hops && frontier.length; hop++) {
    const next = [];
    for (const p of frontier) {
      const g = p.units[p.units.length - 1], cand = [];
      for (let j = 0; j < k; j++) {
        const h = ids[g * k + j], x = xs[g * k + j];
        if (!(x > 0)) break;
        const s = edgeStrength(x, edges.p99) * w(h);
        if (s <= 0 || p.units.includes(h)) continue;
        cand.push({ h, s, x });
      }
      if (!cand.length && !targets) keep(p);  // a dead end is a complete route
      cand.sort((a, b) => b.s - a.s);
      for (const c of cand.slice(0, branch)) next.push({ units: [...p.units, c.h], score: p.score * c.s, xs: [...p.xs, c.x] });
    }
    next.sort((a, b) => b.score - a.score);
    next.length = Math.min(next.length, beam);
    if (targets) for (const p of next) if (targets.has(p.units[p.units.length - 1])) keep(p);
    frontier = next;
  }
  if (!targets) for (const p of frontier) keep(p);  // routes that used every hop
  return [...found.values()].sort((a, b) => b.score - a.score).slice(0, limit);
}

/**
 * The circuit behind a predicted token: start from the units that pushed the target's logit up the
 * most (direct attribution) and follow their strongest inputs upstream through active units. Each
 * path's score is weighted by its end unit's share of the attribution.
 */
export function explainPaths(edges, contrib, opts = {}) {
  const { roots = 8, ...rest } = opts;
  const order = [...contrib.keys()].filter((g) => contrib[g] > 0).sort((a, b) => contrib[b] - contrib[a]).slice(0, roots);
  if (!order.length) return [];
  const top = contrib[order[0]];
  const out = [];
  for (const g of order) {
    for (const p of findPaths(edges, [g], { ...rest, dir: "up" })) {
      out.push({ ...p, score: p.score * (contrib[g] / top), root: g });
    }
    out.push({ units: [g], score: contrib[g] / top, xs: [], root: g });  // the unit's own direct effect
  }
  const seen = new Set();
  return out.filter((p) => { const key = p.units.join(">"); if (seen.has(key)) return false; seen.add(key); return true; })
    .sort((a, b) => b.score - a.score).slice(0, rest.limit ?? 60);
}

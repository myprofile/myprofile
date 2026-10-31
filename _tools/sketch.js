// Usage: node sketch.js in.svg out.svg  — redraws rects/paths with rough.js, keeps text.
const fs = require('fs');
const rough = require('roughjs/bundled/rough.cjs.js');
const gen = rough.generator();
const [,, inp, out] = process.argv;
let svg = fs.readFileSync(inp, 'utf8');
const ACC = '#40045c';
const INK = '#1e1e1e';
// Excalidraw palette, solid fills (no hachure)
const STY = {
  'dg-data':  { stroke: INK, fill: '#ffec99', fillStyle: 'solid' },  // yellow: data and pipelines
  'dg-llm':   { stroke: INK, fill: '#ffc9c9', fillStyle: 'solid' },  // pink: LLM steps
  'dg-model': { stroke: INK, fill: '#b2f2bb', fillStyle: 'solid' },  // green: serving store, small models
  'dg-key':   { stroke: ACC, fill: ACC, fillStyle: 'solid' },        // purple: what the rep sees
  'dg-skip':  { stroke: '#adb5bd', fill: '#f1f3f5', fillStyle: 'solid' },
  'dg-line':  { stroke: '#495057', strokeWidth: 1.4 },
};
const attrs = s => Object.fromEntries([...s.matchAll(/([\w:-]+)="([^"]*)"/g)].map(m => [m[1], m[2]]));
let seed = 1;
const base = () => ({ roughness: 1.1, bowing: 1, strokeWidth: 1.3, hachureAngle: -41, hachureGap: 5, fillWeight: 0.8, seed: seed++ });
const draw = (d, o, dash) => gen.toPaths(gen.path(d, o)).map(p =>
  `<path d="${p.d.replace(/-?\d+\.\d+/g, n => (+n).toFixed(1))}" fill="${p.fill || 'none'}" stroke="${p.stroke || 'none'}" stroke-width="${p.strokeWidth}"${dash && p.stroke !== 'none' && p.fill === 'none' && p.stroke === o.stroke ? ` stroke-dasharray="${dash}"` : ''} stroke-linecap="round"/>`).join('');
function rr(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  if (!r) return `M${x} ${y} H${x + w} V${y + h} H${x} Z`;
  return `M${x + r} ${y} H${x + w - r} Q${x + w} ${y} ${x + w} ${y + r} V${y + h - r} Q${x + w} ${y + h} ${x + w - r} ${y + h} H${x + r} Q${x} ${y + h} ${x} ${y + h - r} V${y + r} Q${x} ${y} ${x + r} ${y} Z`;
}
function styleFor(a) {
  const cls = (a.class || '').split(/\s+/).find(c => STY[c]);
  const o = { ...base(), ...(cls ? STY[cls] : {}) };
  if (!cls) {
    const st = a.stroke || (a.style || '').match(/stroke:\s*([^;]+)/)?.[1];
    o.stroke = st && st.includes('var(--accent)') ? ACC : (st || '#8f8f8f');
    const f = a.fill;
    if (f && f !== 'none') { o.fill = f.includes('var(--accent)') ? ACC : f; o.fillStyle = 'solid'; }
  }
  if (a['stroke-width']) o.strokeWidth = +a['stroke-width'];
  return o;
}
// rects
svg = svg.replace(/<rect\b([^>]*)\/>/g, (m, s) => {
  const a = attrs(s);
  const [x, y, w, h] = ['x', 'y', 'width', 'height'].map(k => +a[k]);
  const o = styleFor(a);
  if (w < 20 && h < 20) { o.roughness = 0.6; o.hachureGap = 3; }   // legend chips
  return `<g>${draw(rr(x, y, w, h, +(a.rx || 0)), o, a['stroke-dasharray'])}</g>`;
});
// paths (lines, arrows, brackets) — skip arrowhead marker defs
svg = svg.replace(/<path\b([^>]*)\/>/g, (m, s) => {
  const a = attrs(s);
  if ((a.class || '') === 'dg-head') return m;
  const o = styleFor(a); o.roughness = 0.9; o.bowing = 0.6;
  let outp = draw(a.d, o, a['stroke-dasharray']);
  if (a['marker-end']) {
    // last point and direction from M/H/V/L commands
    let x = 0, y = 0, px = 0, py = 0;
    for (const [, c, n] of a.d.matchAll(/([MHVL])\s*([-\d.\s,]+)/g)) {
      const v = n.trim().split(/[\s,]+/).map(Number);
      px = x; py = y;
      if (c === 'M' || c === 'L') { x = v[0]; y = v[1]; } else if (c === 'H') x = v[0]; else y = v[0];
      if (c === 'M') { px = x; py = y; }
    }
    const ang = Math.atan2(y - py, x - px), L = 11, sp = 0.45;
    const p1 = [x - L * Math.cos(ang - sp), y - L * Math.sin(ang - sp)], p2 = [x - L * Math.cos(ang + sp), y - L * Math.sin(ang + sp)];
    const ho = { ...o, roughness: 0.5, seed: seed++ };
    outp += draw(`M${p1[0].toFixed(1)} ${p1[1].toFixed(1)} L${x} ${y} L${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`, ho);
  }
  return `<g>${outp}</g>`;
});
svg = svg.replace(/<defs>.*?<\/defs>\n?/s, '');
svg = svg.replace('<svg ', '<svg class="sketch" ');
fs.writeFileSync(out, svg);
console.log(out, svg.length);

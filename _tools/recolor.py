# Recolor Excalidraw PNG diagrams to the site palette.
import sys, numpy as np
from PIL import Image, ImageFilter
H = lambda s: np.array([int(s[i:i+2], 16) for i in (1, 3, 5)], np.float32)
W, K = H('#ffffff'), H('#1e1e1e')
Y, O, V, D = H('#fff9db'), H('#da77f2'), H('#7950f2'), H('#40045c')   # yellow, orchid, violet, dark purple
NONE = W  # "no fill"
INKS = {'#1971c2': D, '#2f9e44': D, '#f08c00': D, '#6741d9': D, '#e03131': V}
GREY = H('#868e96')
# per-image fill mapping: source fill -> target fill
MAPS = {
 'believable-vs-accurate': {'#ffd8a8': O, '#b2f2bb': V, '#a5d8ff': Y, '#e9ecef': NONE},
 'generative-agent-architecture': {'#ffec99': Y, '#a5d8ff': V, '#ffd8a8': O, '#d0bfff': O, '#b2f2bb': V, '#e9ecef': Y},
 'levels-of-analysis': {'#a5d8ff': Y, '#b2f2bb': O, '#d0bfff': V},
 'retrieval-scoring': {'#a5d8ff': Y, '#ffc9c9': O, '#b2f2bb': V, '#d0bfff': O, '#ffec99': Y, '#e9ecef': Y},
 'simulation-loop': {'#e9ecef': Y, '#ffec99': Y, '#a5d8ff': O, '#b2f2bb': V, '#d0bfff': O},
 'wicked-problem-loop': {'#a5d8ff': V, '#ffec99': Y, '#ffc9c9': O, '#b2f2bb': V, '#d0bfff': O, '#e9ecef': Y},
 'how-interview-studio-answers': {'#f6ead6': Y, '#d0c4fb': O, '#c9f0d3': V},
 'one-turn-of-trangs-simulation': {'#f6ead6': Y, '#d0c4fb': O, '#c9f0d3': V},
}
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
im = Image.open(src).convert('RGB'); h, w = im.size[1], im.size[0]
p = np.asarray(im).reshape(-1, 3).astype(np.float32)
# candidate segments: (a, b, a2, b2, tag) — pixel on a->b maps to a2->b2
segs = [(W, K, W, K, 'id')]
fills = [(H(s), t) for s, t in MAPS[name].items()]
for S_, T_ in fills:
    segs += [(W, S_, W, T_, 'fill'), (K, S_, K, T_, 'fillink')]
for c, T_ in INKS.items():
    C = H(c); segs.append((W, C, W, T_, 'ink'))
    for S_, TF in fills: segs.append((S_, C, TF, T_, 'ink'))
segs.append((W, GREY, W, V, 'grey'))
for S_, TF in fills: segs.append((S_, GREY, TF, V, 'grey'))
best = np.full(len(p), 1e9, np.float32); out = p.copy(); ident = np.zeros(len(p), bool)
solid = {i: np.zeros(len(p), bool) for i in range(len(fills))}
for k, (a, b, a2, b2, tag) in enumerate(segs):
    ab = b - a; t = ((p - a) @ ab) / (ab @ ab); tc = np.clip(t, 0, 1)
    r = np.linalg.norm(p - (a + tc[:, None] * ab), axis=1)
    ok = (r < best) & (t > -0.05) & (t < 1.12)
    best[ok] = r[ok]; out[ok] = a2 + tc[ok, None] * (b2 - a2); ident[ok] = tag == 'id'
    if tag == 'fill':
        i = (k - 1) // 2; solid[i] = (solid[i] & ~ok) | (ok & (t > 0.92) & (r < 4))
    else:
        for i in solid: solid[i] &= ~ok
out[best > 18] = p[best > 18]            # leave anything unexplained untouched
# white text on violet fills: inside the filled shape, darker pixels become white
for i, (S_, T_) in enumerate(fills):
    if not np.array_equal(T_, V): continue
    m = Image.fromarray((solid[i].reshape(h, w) * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(21)).filter(ImageFilter.MinFilter(21)).filter(ImageFilter.MinFilter(9))
    from scipy import ndimage
    region = np.asarray(m) > 0
    lab, n = ndimage.label(region); sol = solid[i].reshape(h, w)
    keep = np.zeros_like(region)
    for j in range(1, n + 1):                      # whiten only mostly-solid boxes, not hatched ones
        rj = lab == j
        if sol[rj].mean() > 0.5: keep |= rj
    inner = keep.reshape(-1) & ~solid[i]
    L = lambda c: c @ np.array([.299, .587, .114], np.float32)
    a_ = np.clip((L(S_) - L(p[inner])) / (L(S_) - L(K)), 0, 1)[:, None]
    out[inner] = a_ * 255 + (1 - a_) * V
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3)).save(dst, optimize=True)
print(name, 'unexplained px %.2f%%' % (100 * (best > 18).mean()))

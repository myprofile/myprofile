# Hand-drawn diagrams

Diagrams in `_includes/diagrams/` with `class="sketch"` are generated from the clean
sources in `_diagrams_src/` by `sketch.js` (Rough.js, Excalidraw-style).

To change one: edit the clean source, then regenerate.

```sh
npm install roughjs@4.6.6   # once, in the folder you run node from (or set NODE_PATH)
node _tools/sketch.js _diagrams_src/c360-architecture.svg _includes/diagrams/c360-architecture.svg
```

Colors (Excalidraw palette, solid fills): `dg-data` #fff9db, `dg-llm` #da77f2, `dg-model` #7950f2 (white text),
`dg-key` purple, `dg-skip` pale yellow. No grey.

Labels use the Virgil font (`assets/fonts/Virgil.woff2`, SIL Open Font License, from Excalidraw).

## Recoloring PNG diagrams

`recolor.py` maps the Excalidraw colors in the Generative Agents and Building Trang PNG
diagrams onto the same palette (#fff9db, #da77f2, #7950f2), turns text on solid violet
boxes white, and recolors colored or grey text. The per-image color mapping is in `MAPS`.
Needs numpy, pillow and scipy:

```sh
python _tools/recolor.py original.png out.png <image-name-without-.png>
```

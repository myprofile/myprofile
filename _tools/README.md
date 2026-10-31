# Hand-drawn diagrams

Diagrams in `_includes/diagrams/` with `class="sketch"` are generated from the clean
sources in `_diagrams_src/` by `sketch.js` (Rough.js, Excalidraw-style).

To change one: edit the clean source, then regenerate.

```sh
npm install roughjs@4.6.6   # once, in the folder you run node from (or set NODE_PATH)
node _tools/sketch.js _diagrams_src/c360-architecture.svg _includes/diagrams/c360-architecture.svg
```

Colors (Excalidraw palette, solid fills): `dg-data` yellow, `dg-llm` pink, `dg-model` green,
`dg-key` purple, `dg-skip` pale yellow. No grey.

Labels use the Virgil font (`assets/fonts/Virgil.woff2`, SIL Open Font License, from Excalidraw).

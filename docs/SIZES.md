# Bundle sizes by dialect variant

This document tracks how the WebAssembly bundle size changes as you enable different MLIR dialects.

What’s included in measurements:

- WASM (`wasm/mlir_parser.wasm`)
- JS loader (`wasm/mlir_parser.js`) + bindings (`wasm/bindings.js`)

We record raw bytes, gzip, and brotli. Use the helper script to generate rows:

```bash
npm run size:report
```

Paste the “Markdown row” output into the appropriate table below.

## Baseline variants

| Variant | WASM raw | WASM gzip | WASM brotli | JS gzip (loader+bindings) | Total gzip | Total brotli |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| builtin only | 1170531 B (~1143.1 KiB) | 399155 B (~389.8 KiB) | 290193 B (~283.4 KiB) | 18832 B (~18.4 KiB) | 417987 B (~408.2 KiB) | 306902 B (~299.7 KiB) |
| stablehlo+chlo+vhlo (wasm) | 18177039 B (~17751.0 KiB) | 4217451 B (~4118.6 KiB) | 2320774 B (~2266.4 KiB) | 19185 B (~18.7 KiB) | 4236636 B (~4137.3 KiB) | 2337800 B (~2283.0 KiB) |

## Proposed variants

Add new variants as you register additional dialects and rebuild. Suggested common combinations:

- builtin + func
- builtin + func + arith
- builtin + llvm + cf

Document the exact CMake and registration changes for each variant in `docs/ADD-DIALECTS.md`.

Notes:

- “JS gzip” is `mlir_parser.js` + `bindings.js` compressed.
- “Total” includes WASM gzip + JS gzip. Brotli totals likewise.
- Sizes will vary by toolchain versions and build flags.

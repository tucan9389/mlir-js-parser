# mlir-js-parser

Tiny, WebAssembly-powered MLIR parsing for JavaScript. Use the prebuilt WASM to parse MLIR text in Node or the browser. If you need more dialects, you can add them and rebuild.

• Primary goal: parse MLIR from JS and get structured JSON.
• Secondary goal: let developers add desired dialects and build their own bundle.

## What you can do (right now)

- Use the prebuilt WASM artifacts in `wasm/` directly from Node or a browser.
- Call a tiny API: `parseMlirJson(mlirText)` → `{ ok, json? , error? }`.
- See working samples in `wasm/sample/`.

Artifact sizes (current build):

- `wasm/mlir_parser.js`: ~62 KB (raw), ~18 KB (gzip)
- `wasm/mlir_parser.wasm`: ~1.1 MB (raw), ~398 KB (gzip)
- `wasm/bindings.js`: ~3.0 KB (raw), ~0.8 KB (gzip)

Tip: To recompute sizes (raw, gzip, brotli), run:

```bash
npm run size:report
```

Sizes will grow when you add dialects or enable more features. Use the table below to track.

## Quick start

Prerequisite: a recent Node.js (v18+) for the Node sample. For the browser sample, any static file server will do.

1. Clone and run the Node sample

```bash
git clone <this-repo-url>
cd mlir-js-parser
node wasm/sample/node.mjs
```

You should see an object with `ok: true` and a `json` field describing the IR structure.

1. Try the browser sample

Serve `wasm/sample/` with any static file server (e.g., your favorite dev server). Then open `index.html` and click Parse.

Tip: If you need an example, `npx http-server wasm/sample` is a quick option, but any static server works.

## Example

A single, builtin-safe example that works with the minimal bundle and shows structured JSON output.

Input:

```mlir
module attributes { test.meta = { a = 1 : i32, msg = "hi" } } {}
```

Output (JSON):

```json
{
  "ok": true,
  "json": {
    "attributes": {
      "test.meta": {
        "a": "1",
        "msg": "hi"
      }
    },
    "name": "builtin.module",
    "operands": [],
    "regions": [
      {
        "blocks": [
          {
            "arguments": [],
            "operations": []
          }
        ]
      }
    ],
    "results": []
  }
}
```

Note: The attribute name uses a dialect-style prefix (`test.meta`), which the builtin module accepts in this minimal build. Non-prefixed attribute names may be rejected by the builtin verifier.

## How to use in your project

Copy these files into your project (or reference them directly):

- `wasm/mlir_parser.js`
- `wasm/mlir_parser.wasm`
- `wasm/bindings.js`

Then:

Node (ESM):

```js
import { createParserModule } from './bindings.js';
import ModuleFactory from './mlir_parser.js';

const { parseMlirJson } = await createParserModule(ModuleFactory);
const res = parseMlirJson('module {}');
console.log(res);
```

Browser:

```html
<script type="module">
  import { createParserModule } from './bindings.js';
  import ModuleFactory from './mlir_parser.js';

  const { parseMlirJson } = await createParserModule(ModuleFactory);
  const res = parseMlirJson('module {}');
  console.log(res);
}</script>
```

## API

- `createParserModule(ModuleFactory): Promise<{ Module, parseMlirJson, parseMlir? }>`
  - `ModuleFactory`: the Emscripten-generated module factory exported by `mlir_parser.js`.
  - Returns an object with:
    - `Module`: the Emscripten module (advanced use).
    - `parseMlirJson(text: string)`: synchronously parses MLIR text and returns structured JSON. This is the primary API.
      - On success: `{ ok: true, json: object }`.
      - On failure: `{ ok: false, error: string }`.
    - `parseMlir(text: string)` (optional, debug-only): returns canonical MLIR text as a string. Prefer `parseMlirJson` for all programmatic use; this may be removed in a future version.

## Add dialects or build your own

If you need additional dialects, you can register them on the C++ side and rebuild the WASM bundle. To keep this README environment-agnostic, the full instructions live in separate docs:

- Building from source: see `docs/BUILD.md`
- Adding dialects: see `docs/ADD-DIALECTS.md`

## Project structure (short)

- `cpp/` — tiny C++ core and C API
- `wasm/` — generated loader/wasm + JS bindings and samples
- `scripts/` — helper scripts (optional, for local builds)

## Troubleshooting (common)

- If Node reports ESM import issues, ensure Node v18+ and that your script runs as an ES module.
- In browsers, load via a web server (not `file://`) so the `.wasm` can be fetched.
- Builtin-only by default: This bundle currently registers only the builtin dialect. Ops from other dialects (e.g., `func`, `arith`) will fail to parse until those dialects are registered and linked. See `docs/ADD-DIALECTS.md`.

## License

TBD.

---

## Netron integration checklist

If you plan to use this parser inside Netron (or similar viewers), consider:

- Loading strategy: bundle `mlir_parser.wasm` and `mlir_parser.js` via Netron’s asset pipeline; confirm ESM support or use a wrapper.
- Transfer size: check gz/brotli; prefer brotli over gzip when hosting (most CDNs support it).
- Worker isolation: instantiate the parser inside a Web Worker to avoid blocking the UI thread.
- Timeouts & cancellation: validate large-model inputs; budget parse times and add cancellation hooks.
- Error surfaces: return `{ ok: false, error }` consistently; handle unknown dialects gracefully.
- Dialect coverage: decide which dialects to ship; build multiple variants if needed (see table below).
- Caching: cache `.wasm` with long max-age + immutable; version via file hashes.

## Dialect variants and sizes

Track how bundle size changes as you enable dialects. Use `npm run size:report` after each build and paste a row here.

| Variant | WASM raw | WASM gzip | WASM brotli | JS gzip (loader+bindings) | Total gzip | Total brotli |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| builtin only | ~1.1 MB | ~398 KB | ~289 KB | ~18.6 KB | ~416.6 KB | ~306 KB |

Notes:

- “JS gzip” is `mlir_parser.js` + `bindings.js` compressed.
- “Total” includes `.wasm` gzip + JS gzip; brotli totals likewise.

# mlir-js-parser (minimal)

Tiny, WebAssembly-powered MLIR parsing for JavaScript. Use the prebuilt WASM to parse MLIR text in Node or the browser. If you need more dialects, you can add them and rebuild.

• Primary goal: parse MLIR from JS with a simple API.
• Secondary goal: let developers add desired dialects and build their own bundle.

## What you can do (right now)

- Use the prebuilt WASM artifacts in `wasm/` directly from Node or a browser.
- Call a tiny API: `parseMlir(mlirText)` → `{ ok, module? , error? }`.
- See working samples in `wasm/sample/`.

Artifact sizes (current build):

- `wasm/mlir_parser.js`: ~62 KB (raw), ~18 KB (gzip)
- `wasm/mlir_parser.wasm`: ~1.1 MB (raw), ~378–390 KB (gzip)

Sizes vary when you add dialects or change build flags.

## Quick start

Prerequisite: a recent Node.js (v18+) for the Node sample. For the browser sample, any static file server will do.

1. Clone and run the Node sample

```bash
git clone <this-repo-url>
cd mlir-js-parser
node wasm/sample/node.mjs
```

You should see an object like `{ ok: true, module: 'module {\n}\n' }`.

2. Try the browser sample

Serve `wasm/sample/` with any static file server (e.g., your favorite dev server). Then open `index.html` and click Parse.

Tip: If you need an example, `npx http-server wasm/sample` is a quick option, but any static server works.

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

const { parseMlir } = await createParserModule(ModuleFactory);
const res = parseMlir('module {}');
console.log(res);
```

Browser:

```html
<script type="module">
  import { createParserModule } from './bindings.js';
  import ModuleFactory from './mlir_parser.js';

  const { parseMlir } = await createParserModule(ModuleFactory);
  const res = parseMlir('module {}');
  console.log(res);
}</script>
```

## API

- `createParserModule(ModuleFactory): Promise<{ Module, parseMlir }>`
  - `ModuleFactory`: the Emscripten-generated module factory exported by `mlir_parser.js`.
  - Returns an object with:
    - `Module`: the Emscripten module (advanced use).
    - `parseMlir(text: string)`: synchronously parses MLIR text.
      - On success: `{ ok: true, module: string }` (canonical MLIR text).
      - On failure: `{ ok: false, error: string }`.

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

## License

TBD.

# mlir-js-parser

WebAssembly MLIR parser for JavaScript (Node & browsers). Parse MLIR text into structured JSON and build custom bundles by enabling only the dialects you need.

## Use in your project (JS first)

Copy or reference these files:

- `wasm/mlir_parser.js`
- `wasm/mlir_parser.wasm`
- `wasm/bindings.js`

Node (ESM):

```js
import { createParserModule } from './bindings.js';
import ModuleFactory from './mlir_parser.js';

const { parseMlirJson } = await createParserModule(ModuleFactory);
const res = parseMlirJson('module {}');
console.log(res); // { ok: true, json: {...} } or { ok: false, error: '...' }
```

Browser:

```html
<script type="module">
  import { createParserModule } from './bindings.js';
  import ModuleFactory from './mlir_parser.js';

  const { parseMlirJson } = await createParserModule(ModuleFactory);
  const res = parseMlirJson('module {}');
  console.log(res);
</script>
```

See `wasm/sample/` for ready-to-run examples.

### Example

Input MLIR:

```mlir
module attributes { test.meta = { a = 1 : i32, msg = "hi" } } {}
```

Output (summary):

```json
{
  "ok": true,
  "json": {
    "name": "builtin.module",
    "attributes": { "test.meta": { "a": "1", "msg": "hi" } },
    "operands": [],
    "regions": [ { "blocks": [ { "arguments": [], "operations": [] } ] } ],
    "results": []
  }
}
```

Note: The minimal bundle registers only Builtin. Ops from other dialects will fail until those dialects are enabled and linked.

## Dialect support

The minimal bundle registers only the Builtin dialect. Additional dialects can be enabled by registering them in C++ and linking their MLIR libraries.

| Dialect | Status | How to enable (summary) |
| --- | --- | --- |
| Builtin | Supported (default) | No action required |
| func | Planned | Register `mlir::func::FuncDialect`; link `MLIRFuncDialect` |
| arith | Planned | Register `mlir::arith::ArithDialect`; link `MLIRArithDialect` |
| scf | Planned | Register `mlir::scf::SCFDialect`; link `MLIRSCFDialect` |
| tensor | Planned | Register `mlir::tensor::TensorDialect`; link `MLIRTensorDialect` |

See `docs/ADD-DIALECTS.md` for step-by-step instructions.

## Status

- Default: Minimal bundle with only the Builtin dialect registered
- Ready today: Use the prebuilt artifacts in `wasm/` directly in Node or browsers
- Extensible: Add common dialects like `func`, `arith`, `scf`, `tensor` as needed by following the docs

For build/extension guides:

- Build guide: `docs/BUILD.md`
- Add dialects: `docs/ADD-DIALECTS.md`

Approximate bundle sizes (vary by toolchain and flags):

- `wasm/mlir_parser.js`: ~62 KB (raw) / ~18 KB (gzip)
- `wasm/mlir_parser.wasm`: ~1.14 MB (raw) / ~390 KB (gzip)
- `wasm/bindings.js`: ~3.4 KB (raw) / ~0.8 KB (gzip)

Generate a fresh size report (raw/gzip/brotli) and update `docs/SIZES.md`:

```bash
npm run size:report
```

Enabling more dialects increases size. Track variants in `docs/SIZES.md`.

## Quick start

Prereqs: Node v18+ (ESM). For the browser sample, serve files over HTTP(s).

1. Clone and run the Node sample

```bash
git clone <this-repo-url>
cd mlir-js-parser
npm run sample:node
```

You should see `{ ok: true, ... }` with parsed output.

1. (Optional) Try JSON output locally

```bash
npm run sample:json
```

You should see a structured JSON object for a module with attributes.

1. Open the browser sample

Serve `wasm/sample/` and open `index.html`:

```bash
npm run serve
```

Click Parse to see results.

## API

- `createParserModule(ModuleFactory): Promise<{ Module, parseMlirJson, parseMlir }>`
  - `ModuleFactory`: The Emscripten module factory from `mlir_parser.js`.
  - Returns:
    - `Module`: The Emscripten module (advanced use)
    - `parseMlirJson(text: string)`: Synchronous parse → `{ ok: true, json } | { ok: false, error }`
    - `parseMlir(text: string)`: Returns canonical MLIR text (useful for debugging)
  - Errors: When parsing fails, `error` includes detailed diagnostics with `file:line:column: message` where available.

## Add dialects

1. Register dialects in C++

- File: `cpp/src/parser.cc`, extend `registerDialects(MLIRContext&)`:

```cpp
// Example: add FuncDialect
#include "mlir/Dialect/Func/IR/FuncOps.h"

void registerDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::BuiltinDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
}
```

1. Link MLIR dialect libraries in CMake

- File: `cpp/CMakeLists.txt`
- Add dialect libraries to `mlir_libs` (e.g., `MLIRFuncDialect`):

```cmake
set(mlir_libs
  MLIRIR
  MLIRParser
  MLIRSupport
  # Add: required dialect libraries
  MLIRFuncDialect
)
```

1. Build the WASM bundle

```bash
# Ensure Emscripten env is active (e.g., emsdk_env.sh)
LLVM_DIR=...</path/to/wasm/llvm> MLIR_DIR=...</path/to/wasm/llvm> \
  npm run build:wasm
```

Artifacts will be written to `wasm/mlir_parser.{js,wasm}`. See `docs/ADD-DIALECTS.md` for details and common dialect sets.

## Docs & sizes

- Build guide: `docs/BUILD.md`
- Add dialects: `docs/ADD-DIALECTS.md`
- Bundle size tracking: `docs/SIZES.md` (run `npm run size:report` to update tables)

## Troubleshooting

- Node ESM issues: use Node v18+ and ensure your script runs as an ES module.
- In browsers, serve over HTTP(s) (not `file://`) so the `.wasm` can be fetched.
- Minimal bundle registers Builtin only. If `func`/`arith` ops fail, add those dialects and rebuild.

## License

TBD

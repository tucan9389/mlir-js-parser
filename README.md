# mlir-js-parser (minimal)

A minimal, WebAssembly-powered MLIR parser exposed to JavaScript. The goal is to extract only the essential pieces from an existing, larger setup (e.g., `../mlir-js-parser-big`), wire up a tiny C++ core (import/register dialects + parse), and surface a simple JS API for parsing MLIR text. Over time, we’ll expand dialect coverage by registering more dialects on the C++ side.

## Core objectives
- Minimal C++ core that:
  - Registers a small, configurable set of MLIR dialects
  - Parses MLIR text into an `mlir::ModuleOp`
  - Optionally re-prints canonical MLIR text for now (round-trip as validation)
- Build to WebAssembly using Emscripten and the LLVM/MLIR toolchain
- Provide a tiny JS API: `parseMlir(mlirText: string): { ok: boolean, error?: string, module?: string }`
- Keep the project small and focused. Prefer linking against existing builds rather than rebuilding LLVM/MLIR from scratch when possible.

## Assumptions and corrections
- We assume `../mlir-js-parser-big` contains prior work or helpful scripts and maybe hints to a working LLVM/MLIR + dialect setup.
- We will not vendor or fork full LLVM/MLIR here; instead, we’ll rely on a local LLVM build and CMake toolchain configuration that we can reuse for both native and WASM builds.
- WebAssembly is mandatory: we target a browser-compatible build (and Node.js support if trivial).
- The C++ implementation stays minimal: create an MLIR context, register selected dialects, parse text, and print the module (for now).

## High-level plan
1. Discover existing LLVM/MLIR setup
   - Find local LLVM/MLIR install or build hints from `../mlir-js-parser-big`.
   - Identify minimal MLIR libs needed: IR, Parser, Support, and selected dialects.
2. Establish build system
   - Author a minimal `CMakeLists.txt` to build our small C++ library/executable.
   - Confirm a successful native build first (optional but useful for diagnostics).
3. Emscripten + WebAssembly build
   - Configure Emscripten toolchain (`emcmake`, `emcc`) to build the same target to WebAssembly.
   - Produce `mlir_parser.wasm` + `mlir_parser.js` (or `mlir_parser.mjs`) with an exported C API.
4. JS bridge and sample
   - Wrap the exported C function in a tiny JS function `parseMlir(text)`.
   - Add a basic sample (Node and/or browser) that parses a tiny MLIR snippet.
5. Iterate on dialect coverage
   - Start with builtin/standard dialects.
   - Gradually add more dialect registrations as needed.

## Repository layout (proposed)
- `cpp/`
  - `CMakeLists.txt` — minimal build rules
  - `src/`
    - `parser.cc` — creates context, registers dialects, parses, prints
  - `include/`
    - `parser.h` — C-compatible header for exported functions
- `wasm/`
  - `CMakeLists.txt` — Emscripten build setup or helper script
  - `bindings.js` — thin JS wrapper (`parseMlir`)
  - `sample/`
    - `index.html` — browser example
    - `node.mjs` — Node example
- `scripts/`
  - `find-llvm.sh` — heuristic to locate LLVM/MLIR builds (optional)
  - `build-native.sh` — native build (sanity check)
  - `build-wasm.sh` — WASM build using Emscripten

This layout is a starting point. We will prune or adjust as we discover constraints from your environment.

## Toolchain requirements
- LLVM/MLIR builds available locally (headers + libs)
  - Native build: any compatible LLVM/MLIR install with CMake config packages
  - WebAssembly build: LLVM/MLIR must be cross-compiled with Emscripten (you cannot link native macOS/Linux libs into a WASM module). Build or provide an Emscripten-targeted LLVM/MLIR with the same version used natively.
  - RTTI often required by MLIR; ensure it’s enabled consistently.
  - Path hints expected in `../mlir-js-parser-big`.
- CMake >= 3.20
- Emscripten SDK (latest stable)
  - Activate via `emsdk activate <version>` and `source emsdk_env.sh`
- Node.js (for local testing), or a modern browser

## Minimal C API (target)
We expose a very small C interface (exported via Emscripten):

```c
// parser.h
#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success; non-zero on error. On success, writes a canonical
// MLIR string into the provided buffer (UTF-8). If the buffer is too small,
// returns a negative value with the required size.
int mlir_parse_to_string(const char* mlir_text,
                         char* out_buffer,
                         int out_capacity,
                         char* err_buffer,
                         int err_capacity);

#ifdef __cplusplus
}
#endif
```

- JS side will call this via `cwrap`/`embind` or direct exports and convert it into a Promise-based API if needed.

## Build steps (sketch)

### 1) Locate LLVM/MLIR
If you already know where your LLVM build is installed, set:

```bash
export LLVM_DIR=/path/to/llvm/install
export MLIR_DIR=$LLVM_DIR
```

Otherwise, consult `../mlir-js-parser-big` and scripts we’ll add (`scripts/find-llvm.sh`).

### 2) Native build (optional sanity check)
```bash
mkdir -p build/native
cd build/native
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm -DMLIR_DIR=$MLIR_DIR/lib/cmake/mlir ../../cpp
cmake --build . --target mlir_parser_min_cli
./mlir_parser_min_cli || true
```

### 3) WebAssembly build
Important: Ensure your LLVM/MLIR is built for WebAssembly using Emscripten (separate from native). Set `LLVM_DIR`/`MLIR_DIR` to that cross-compiled install.
```bash
# Ensure Emscripten env is active
source /path/to/emsdk/emsdk_env.sh

mkdir -p build/wasm
cd build/wasm
emcmake cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm \
  -DMLIR_DIR=$MLIR_DIR/lib/cmake/mlir \
  ../../cpp
cmake --build . --target mlir_parser
```

This generates `wasm/mlir_parser.js` and `wasm/mlir_parser.wasm`.

Tip: In the Emscripten build, pass `-DLLVM_DIR` and `-DMLIR_DIR` explicitly to CMake (as shown). Setting shell variables alone is not sufficient for `find_package(LLVM/MLIR)`.

### Building LLVM/MLIR for WebAssembly (one-time)

To produce `LLVM_DIR`/`MLIR_DIR` that work under Emscripten, build LLVM/MLIR with the Emscripten toolchain:

```bash
# 0) Ensure emsdk is installed and active
source /path/to/emsdk/emsdk_env.sh

# 1) Fetch llvm-project (if you don’t already have it)
git clone https://github.com/llvm/llvm-project.git ../llvm-project

# 2) Configure minimal MLIR for WASM
emcmake cmake -S ../llvm-project/llvm -B ../llvm-project/build-wasm -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD=WebAssembly \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

# 3) Build minimal MLIR libs (parser + IR + support + asm/bytecode readers)
cmake --build ../llvm-project/build-wasm --target MLIRParser MLIRAsmParser MLIRBytecodeReader MLIRIR MLIRSupport

# 4) Use these as your WASM cmake packages
export LLVM_DIR=$PWD/../llvm-project/build-wasm
export MLIR_DIR=$PWD/../llvm-project/build-wasm

# 5) Build this project for WASM
emcmake cmake -S cpp -B build/wasm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$MLIR_DIR/lib/cmake/mlir"
cmake --build build/wasm --target mlir_parser
```

### 4) JS wrapper and test
For Node:

```bash
node wasm/sample/node.mjs
```

For browser:

- Open `wasm/sample/index.html` via a local server (due to WASM fetch requirements):

```bash
npx http-server wasm/sample
# then navigate to http://localhost:8080
```

## Risks and notes

- Linking MLIR to WebAssembly can be heavy; we must be aggressive about minimizing linked libs and enabling dead code elimination.
- Some dialects may require additional dependencies; we’ll start with builtin/standard ops only.
- If size becomes an issue, consider building a specialized MLIR with only required components.
- For WASM builds, you must use an LLVM/MLIR toolchain compiled with Emscripten; native libraries will not link into a WASM module.

## Troubleshooting

- Node ESM error (e.g., "Unexpected token import"): use a recent Node that supports ES modules out of the box. Node v18+ is recommended; v20+ tested. This repo sets `"type": "module"` and the Emscripten output uses ES6 modules when `-sEXPORT_ES6=1` is enabled.
- `UTF8ToString is not a function` at runtime: ensure the Emscripten link options include `-sEXPORTED_RUNTIME_METHODS=['cwrap','ccall','UTF8ToString']`. This is already configured in `cpp/CMakeLists.txt` under the EMSCRIPTEN branch. Rebuild the WASM target after changing flags.
- Accessing `HEAPU8` throws/undefined: avoid direct heap access. Use `Module.cwrap` and `Module.UTF8ToString` as done in `wasm/bindings.js`. The wrapper handles allocation (`_malloc`/`_free`) and UTF-8 conversion for you.
- Binaryen version mismatch warning during link: generally benign, but updating Emscripten to a matching version for your binaryen toolchain will silence the warning.
- Linking errors about missing MLIR libraries: make sure your Emscripten-built LLVM/MLIR includes the specific libs required. For this project we need at least: `MLIRParser`, `MLIRAsmParser`, `MLIRBytecodeReader`, `MLIRIR`, and `MLIRSupport`. The bootstrap script `scripts/bootstrap-llvm-wasm.sh` builds these explicitly.
- CMake cannot find LLVM/MLIR: pass `-DLLVM_DIR=/path/to/.../lib/cmake/llvm` and `-DMLIR_DIR=/path/to/.../lib/cmake/mlir` explicitly on the CMake command line (environment variables alone are not enough for `find_package`).

Tip: After tweaking CMake flags or Emscripten options, do a clean rebuild of the WASM build directory to ensure flags are applied consistently.

## Next steps

- Add the minimal `CMakeLists.txt` and C++ stub (`parser.h/.cc`).
- Add Emscripten export config and a tiny JS wrapper.
- Wire up sample inputs and smoke tests.
- Iterate on dialect registration in C++ as needed.

## Quick scripts

If you use the provided scripts and `package.json`:

```bash
# Try to locate LLVM/MLIR and export env (prints export commands)
bash scripts/find-llvm.sh

# Build a minimal LLVM/MLIR toolchain for WASM (one-time, heavy)
npm run bootstrap:wasm-llvm

# Native build (sanity)
LLVM_DIR=...</path/to/native/llvm> MLIR_DIR=...</path/to/native/llvm> npm run build:native

# WASM build (Emscripten + LLVM/MLIR for wasm)
source /path/to/emsdk/emsdk_env.sh
LLVM_DIR=...</path/to/wasm/llvm> MLIR_DIR=...</path/to/wasm/llvm> npm run build:wasm

# Node sample
npm run sample:node

# Browser sample
npm run serve
```

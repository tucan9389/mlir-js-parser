# Build from source

This guide explains how to build the WASM bundle yourself and how to run a native sanity check. It is intentionally environment-agnostic:

- Use any platform with CMake (>= 3.20) and Ninja.
- Use any LLVM/MLIR installation that exposes CMake packages.
- For WebAssembly, you must use an LLVM/MLIR toolchain built with Emscripten.

If you don't need to customize dialects, you can skip this guide and use the prebuilt files in `wasm/`.
CI note: Snapshot tests build the WASM artifacts first (with Emscripten and the required MLIR dialect libs) and then run Vitest snapshots against those artifacts.

## Prerequisites

- CMake >= 3.20
- Ninja (recommended)
- Node.js (optional, for samples)
- LLVM/MLIR with CMake packages available
  - Native build: an ordinary host build of LLVM/MLIR
  - WASM build: an Emscripten-targeted build of LLVM/MLIR
- Emscripten SDK on your PATH for the WASM build (provides `emcc`, `emcmake`)

Note: LLVM/MLIR must be built twice if you want both native and WASM: once for your host, and once for Emscripten. You cannot link native libs into a WASM module.

## Directory overview

- `cpp/` contains the C++ sources and CMake config
- `wasm/` is where the generated `mlir_parser.{js,wasm}` land
- `scripts/` has convenience scripts that wrap the steps below

## Configure paths

CMake needs to know where LLVM and MLIR CMake packages live. Wherever your install is, define:

```bash
export LLVM_DIR=/path/to/llvm/install
export MLIR_DIR=$LLVM_DIR
```

Then pass these to CMake as `-DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" -DMLIR_DIR="$MLIR_DIR/lib/cmake/mlir"`.

## Native build (sanity check)

```bash
# From repo root
LLVM_DIR=...</path/to/host/llvm> MLIR_DIR=...</path/to/host/llvm> \
  bash scripts/build-native.sh
```

This builds and runs `build/native/mlir_parser_min_cli` which parses a trivial `module {}`.

## WebAssembly build

```bash
# Ensure Emscripten is active (so `emcc` exists on PATH)
# e.g., source your emsdk_env.sh before continuing

LLVM_DIR=...</path/to/wasm/llvm> MLIR_DIR=...</path/to/wasm/llvm> \
  bash scripts/build-wasm.sh
```

Outputs:

- `wasm/mlir_parser.js` (Emscripten loader)
- `wasm/mlir_parser.wasm` (the module)

Open `wasm/sample/` with a static server to try the browser sample or run the Node sample:

```bash
node wasm/sample/node.mjs
```

### Optional: Add StableHLO/CHLO/VHLO dialects (WASM)

StableHLO family dialects are not part of LLVM/MLIR core. To parse StableHLO IR without preprocessing, build StableHLO for wasm and link it:

1. Build StableHLO for wasm (minimal dialects only)

```bash
# Uses ../stablehlo by default; set STABLEHLO_SRC_DIR to override
LLVM_DIR=$LLVM_DIR MLIR_DIR=$MLIR_DIR \
  bash scripts/build-stablehlo-wasm.sh
```

This produces static archives under:

- `../stablehlo/build-wasm/stablehlo/dialect/` (e.g., `libStablehloOps.a`, `libVhloOps.a`, `libChloOps.a`, plus support libs like `libStablehloBase.a`, `libStablehloTypeInference.a`, `libVhloTypes.a`)

1. Rebuild our wasm linking StableHLO libs

```bash
STABLEHLO_LIB_DIR="$PWD/../stablehlo/build-wasm/stablehlo/dialect" \
STABLEHLO_INCLUDE_DIR="$PWD/../stablehlo" \
  bash scripts/build-wasm.sh
```

If the libraries are found, the build will define `HAVE_STABLEHLO_DIALECT`, `HAVE_CHLO_DIALECT`, and/or `HAVE_VHLO_DIALECT`, and the parser will register them automatically.

See `docs/STABLEHLO_COVERAGE.md` for end-to-end coverage results using this configuration and guidance on scanning a StableHLO corpus.

## Building LLVM/MLIR for WebAssembly

If you don’t already have an Emscripten-targeted LLVM/MLIR, the repository includes a helper script that configures a minimal build:

```bash
npm run bootstrap:wasm-llvm
```

This will:

- Clone `llvm-project` under `../llvm-project` if missing
- Configure a WASM build in `../llvm-project/build-wasm`
- Build the minimal set of MLIR libraries this project needs

After it completes, set:

```bash
export LLVM_DIR=$PWD/../llvm-project/build-wasm
export MLIR_DIR=$PWD/../llvm-project/build-wasm
```

Then run the WebAssembly build steps above.

## Troubleshooting

- Could not find LLVM/MLIR CMake packages: pass `-DLLVM_DIR=.../lib/cmake/llvm` and `-DMLIR_DIR=.../lib/cmake/mlir` explicitly on the configure command.
- Emscripten not found: ensure `emcc` is on PATH by activating your SDK.
- Large output size: adding dialects increases size; enable Release builds and allow memory growth (already configured).

#!/usr/bin/env bash
set -euo pipefail

: "${LLVM_DIR:?Set LLVM_DIR (e.g., /path/to/wasm/llvm/install)}"
: "${MLIR_DIR:?Set MLIR_DIR (usually same as LLVM_DIR)}"

if ! command -v emcc >/dev/null 2>&1; then
  echo "Emscripten compiler 'emcc' not found on PATH. Please source emsdk_env.sh or install via Homebrew." >&2
  exit 1
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build/wasm"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

emcmake cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$MLIR_DIR/lib/cmake/mlir" \
  "$ROOT_DIR/cpp"

cmake --build . --target mlir_parser

# Output expected at wasm/mlir_parser.{js,wasm}
ls -lh "$ROOT_DIR/wasm" || true

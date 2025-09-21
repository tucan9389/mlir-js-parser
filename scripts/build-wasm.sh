#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Try to locate LLVM/MLIR if not provided in env
if [[ -z "${LLVM_DIR:-}" || -z "${MLIR_DIR:-}" ]]; then
  if exports=$(bash "$ROOT_DIR/scripts/find-llvm.sh" 2>/dev/null); then
    eval "$exports"
    echo "Using auto-detected LLVM_DIR=$LLVM_DIR"
    echo "Using auto-detected MLIR_DIR=$MLIR_DIR"
  else
    echo "LLVM_DIR/MLIR_DIR not set and auto-detection failed."
    echo "Set LLVM_DIR (e.g., /path/to/llvm/build) and MLIR_DIR or run scripts/bootstrap-llvm-wasm.sh" >&2
    exit 1
  fi
fi

if ! command -v emcc >/dev/null 2>&1; then
  echo "Emscripten compiler 'emcc' not found on PATH. Please source emsdk_env.sh or install via Homebrew." >&2
  exit 1
fi

BUILD_DIR="$ROOT_DIR/build/wasm"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

emcmake cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$MLIR_DIR/lib/cmake/mlir" \
  ${STABLEHLO_LIB_DIR:+-DSTABLEHLO_LIB_DIR="$STABLEHLO_LIB_DIR"} \
  ${STABLEHLO_INCLUDE_DIR:+-DSTABLEHLO_INCLUDE_DIR="$STABLEHLO_INCLUDE_DIR"} \
  "$ROOT_DIR/cpp"

cmake --build . --target mlir_parser

# Output expected at wasm/mlir_parser.{js,wasm}
ls -lh "$ROOT_DIR/wasm" || true

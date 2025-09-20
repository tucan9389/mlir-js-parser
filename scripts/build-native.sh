#!/usr/bin/env bash
set -euo pipefail

: "${LLVM_DIR:?Set LLVM_DIR (e.g., /path/to/llvm/install)}"
: "${MLIR_DIR:?Set MLIR_DIR (usually same as LLVM_DIR)}"

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build/native"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$MLIR_DIR/lib/cmake/mlir" \
  "$ROOT_DIR/cpp"

cmake --build . --target mlir_parser_min_cli

"$BUILD_DIR/mlir_parser_min_cli" || true

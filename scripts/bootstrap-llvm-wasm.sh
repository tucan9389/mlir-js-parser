#!/usr/bin/env bash
set -euo pipefail

# This script builds a minimal LLVM/MLIR toolchain for WebAssembly using Emscripten.
# It configures and builds only the MLIR libs we need (IR/Parser/Support) for our WASM parser.

if ! command -v emcc >/dev/null 2>&1; then
  echo "Emscripten 'emcc' not found. Please install emsdk and source emsdk_env.sh before continuing." >&2
  echo "  https://emscripten.org/docs/getting_started/downloads.html" >&2
  exit 1
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SRC_DIR="${ROOT_DIR}/../llvm-project"
BUILD_DIR="${SRC_DIR}/build-wasm"

if [[ ! -d "$SRC_DIR/llvm" ]]; then
  echo "Cloning llvm-project into: $SRC_DIR"
  git clone --depth 1 https://github.com/llvm/llvm-project.git "$SRC_DIR"
fi

echo "Configuring LLVM/MLIR (WASM) at: $BUILD_DIR"
emcmake cmake -S "$SRC_DIR/llvm" -B "$BUILD_DIR" -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD=WebAssembly \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_THREADS=OFF -DLLVM_ENABLE_PTHREADS=OFF \
  -DMLIR_ENABLE_THREADING=OFF -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

echo "Building MLIR libs (this will take a while)"
cmake --build "$BUILD_DIR" --target \
  MLIRParser MLIRAsmParser MLIRBytecodeReader MLIRIR MLIRSupport \
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect \
  MLIRMemRefDialect MLIRTensorDialect MLIRMathDialect MLIRDLTIDialect \
  MLIRVectorDialect MLIRLinalgDialect MLIRLLVMDialect MLIRSPIRVDialect \
  MLIRTransformDialect MLIRBufferizationDialect MLIRSparseTensorDialect \
  MLIROpenMPDialect MLIRGPUDialect MLIRTosaDialect MLIRAsyncDialect \
  MLIREmitCDialect MLIRShapeDialect

echo
echo "Success. Use these settings for WASM builds:"
echo "  export LLVM_DIR=$BUILD_DIR"
echo "  export MLIR_DIR=$BUILD_DIR"
echo
echo "Then run:"
echo "  emcmake cmake -S $ROOT_DIR/cpp -B $ROOT_DIR/build/wasm -G Ninja \\\" \
     -DCMAKE_BUILD_TYPE=Release \\\" \
     -DLLVM_DIR=\"$BUILD_DIR/lib/cmake/llvm\" \\\" \
     -DMLIR_DIR=\"$BUILD_DIR/lib/cmake/mlir\""
echo "  cmake --build $ROOT_DIR/build/wasm --target mlir_parser"

echo
echo "Optional: Build StableHLO (external dialects) for WASM and pass paths to our CMake via STABLEHLO_* vars"
echo "  git clone https://github.com/openxla/stablehlo.git $SRC_DIR/../stablehlo"
echo "  emcmake cmake -S $SRC_DIR/../stablehlo -B $SRC_DIR/../stablehlo/build-wasm -G Ninja \\\" \
  -DCMAKE_BUILD_TYPE=Release \\\" \
  -DLLVM_DIR=$BUILD_DIR/lib/cmake/llvm \\\" \
  -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir"
echo "  cmake --build $SRC_DIR/../stablehlo/build-wasm"
echo
echo "Then rebuild our WASM with:"
echo "  emcmake cmake -S $ROOT_DIR/cpp -B $ROOT_DIR/build/wasm -G Ninja \\\" \
  -DCMAKE_BUILD_TYPE=Release \\\" \
  -DLLVM_DIR=$BUILD_DIR/lib/cmake/llvm \\\" \
  -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir \\\" \
  -DSTABLEHLO_LIB_DIR=$SRC_DIR/../stablehlo/build-wasm/lib \\\" \
  -DSTABLEHLO_INCLUDE_DIR=$SRC_DIR/../stablehlo"
echo "  cmake --build $ROOT_DIR/build/wasm --target mlir_parser"

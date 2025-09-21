#!/usr/bin/env bash
set -euo pipefail

# Build the StableHLO project (Stablehlo/Chlo/Vhlo dialects) for WebAssembly using Emscripten.
# This produces static archives we can link into our wasm parser.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SRC_DIR_DEFAULT="${ROOT_DIR}/../stablehlo"
SRC_DIR="${STABLEHLO_SRC_DIR:-$SRC_DIR_DEFAULT}"
BUILD_DIR="${SRC_DIR}/build-wasm"
# Overlay CMake that builds only the dialect libraries (no integrations/tools/tests)
OVERLAY_DIR="${ROOT_DIR}/scripts/cmake-overlays/stablehlo-wasm-min"
NATIVE_LLVM_BUILD_DIR_DEFAULT="${ROOT_DIR}/../llvm-project/build-native"
NATIVE_LLVM_BUILD_DIR="${NATIVE_LLVM_BUILD_DIR:-$NATIVE_LLVM_BUILD_DIR_DEFAULT}"

if ! command -v emcc >/dev/null 2>&1; then
  echo "Emscripten 'emcc' not found. Please install emsdk and source emsdk_env.sh before continuing." >&2
  exit 1
fi

# Try to locate LLVM/MLIR if not provided in env
if [[ -z "${LLVM_DIR:-}" || -z "${MLIR_DIR:-}" ]]; then
  if exports=$(bash "$ROOT_DIR/scripts/find-llvm.sh" 2>/dev/null); then
    eval "$exports"
    echo "Using auto-detected LLVM_DIR=$LLVM_DIR"
    echo "Using auto-detected MLIR_DIR=$MLIR_DIR"
  else
    echo "LLVM_DIR/MLIR_DIR not set and auto-detection failed." >&2
    echo "Set LLVM_DIR (e.g., /path/to/llvm/build) and MLIR_DIR or run scripts/bootstrap-llvm-wasm.sh" >&2
    exit 1
  fi
fi

# Clone StableHLO if missing (shallow)
if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Cloning StableHLO into: ${SRC_DIR}"
  git clone --depth 1 https://github.com/openxla/stablehlo.git "${SRC_DIR}"
fi

# Ensure native MLIR tools are available (needed during cross-compilation):
# - mlir-pdll
# - mlir-tblgen
# - llvm-tblgen
NEED_TO_BUILD_NATIVE_TOOLS=0
if [[ -z "${MLIR_NATIVE_TOOL_DIR:-}" ]]; then
  # Try common locations first
  if [[ -x "${NATIVE_LLVM_BUILD_DIR}/bin/mlir-pdll" && -x "${NATIVE_LLVM_BUILD_DIR}/bin/mlir-tblgen" && -x "${NATIVE_LLVM_BUILD_DIR}/bin/llvm-tblgen" ]]; then
    export MLIR_NATIVE_TOOL_DIR="${NATIVE_LLVM_BUILD_DIR}/bin"
  elif [[ -x "${LLVM_DIR}/bin/mlir-pdll" && -x "${LLVM_DIR}/bin/mlir-tblgen" && -x "${LLVM_DIR}/bin/llvm-tblgen" ]]; then
    export MLIR_NATIVE_TOOL_DIR="${LLVM_DIR}/bin"
  else
    NEED_TO_BUILD_NATIVE_TOOLS=1
  fi
fi

if [[ ${NEED_TO_BUILD_NATIVE_TOOLS} -eq 1 ]]; then
  echo "Building native MLIR tools at: ${NATIVE_LLVM_BUILD_DIR}"
  SRC_LLVM_DIR="${ROOT_DIR}/../llvm-project/llvm"
  if [[ ! -d "${SRC_LLVM_DIR}" ]]; then
    echo "Cloning llvm-project into: ${ROOT_DIR}/../llvm-project"
    git clone --depth 1 https://github.com/llvm/llvm-project.git "${ROOT_DIR}/../llvm-project"
  fi
  cmake -S "${SRC_LLVM_DIR}" -B "${NATIVE_LLVM_BUILD_DIR}" -G Ninja \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD=Native \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF
  cmake --build "${NATIVE_LLVM_BUILD_DIR}" --target mlir-pdll mlir-tblgen llvm-tblgen
  export MLIR_NATIVE_TOOL_DIR="${NATIVE_LLVM_BUILD_DIR}/bin"
fi

if [[ -z "${MLIR_NATIVE_TOOL_DIR:-}" ]]; then
  echo "ERROR: Could not locate native MLIR tools (mlir-pdll, mlir-tblgen, llvm-tblgen)." >&2
  echo "Set MLIR_NATIVE_TOOL_DIR to a directory containing these binaries and retry." >&2
  exit 1
else
  echo "Using MLIR_NATIVE_TOOL_DIR=${MLIR_NATIVE_TOOL_DIR}"
fi

# Workaround: Some projects look for native tools under ${MLIR_DIR}/NATIVE/bin when cross-compiling.
# Create that path and symlink required tools if missing.
NATIVE_BIN_DIR="${MLIR_DIR}/NATIVE/bin"
mkdir -p "${NATIVE_BIN_DIR}"
for tool in mlir-pdll mlir-tblgen llvm-tblgen; do
  if [[ ! -x "${NATIVE_BIN_DIR}/${tool}" ]]; then
    if [[ -x "${MLIR_NATIVE_TOOL_DIR}/${tool}" ]]; then
      ln -sf "${MLIR_NATIVE_TOOL_DIR}/${tool}" "${NATIVE_BIN_DIR}/${tool}"
    fi
  fi
done

mkdir -p "${BUILD_DIR}"

echo "Configuring StableHLO (WASM, minimal dialects only) at: ${BUILD_DIR}"
emcmake cmake -S "${OVERLAY_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="${LLVM_DIR}/lib/cmake/llvm" \
  -DMLIR_DIR="${MLIR_DIR}/lib/cmake/mlir" \
  -DMLIR_NATIVE_TOOL_DIR="${MLIR_NATIVE_TOOL_DIR}" \
  -DMLIR_TABLEGEN_EXE="${MLIR_NATIVE_TOOL_DIR}/mlir-tblgen" \
  -DLLVM_TABLEGEN_EXE="${MLIR_NATIVE_TOOL_DIR}/llvm-tblgen" \
  -DSTABLEHLO_SOURCE_DIR="${SRC_DIR}"

# Build only the dialect static libraries (Ops) to minimize dependencies.
# Use explicit targets discovered via 'ninja -t targets': they live under stablehlo/dialect/.
cmake --build "${BUILD_DIR}" --target \
  stablehlo/dialect/libStablehloOps.a \
  stablehlo/dialect/libVhloOps.a \
  stablehlo/dialect/libVhloTypes.a \
  stablehlo/dialect/libStablehloBase.a \
  stablehlo/dialect/libStablehloTypeInference.a \
  stablehlo/dialect/libStablehloAssemblyFormat.a \
  stablehlo/dialect/libStablehloBroadcastUtils.a \
  stablehlo/dialect/libVersion.a \
  stablehlo/dialect/libChloOps.a

# Print hints for linking
echo
echo "StableHLO wasm build complete. To link with our parser, pass:"
echo "  -DSTABLEHLO_LIB_DIR=${BUILD_DIR}/stablehlo/dialect \" \
  -DSTABLEHLO_INCLUDE_DIR=${SRC_DIR}"

echo "Then rebuild our wasm:"
echo "  emcmake cmake -S $ROOT_DIR/cpp -B $ROOT_DIR/build/wasm -G Ninja \\\" \
     -DCMAKE_BUILD_TYPE=Release \\\" \
     -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm \\\" \
     -DMLIR_DIR=$MLIR_DIR/lib/cmake/mlir \\\" \
     -DSTABLEHLO_LIB_DIR=${BUILD_DIR}/lib \\\" \
     -DSTABLEHLO_INCLUDE_DIR=${SRC_DIR}"
echo "  cmake --build $ROOT_DIR/build/wasm --target mlir_parser"

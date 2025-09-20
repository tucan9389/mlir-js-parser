#!/usr/bin/env bash
set -euo pipefail

# Heuristically locate LLVM/MLIR build/install directories.
# Prints export commands for LLVM_DIR and MLIR_DIR on success.

LLVM_DIR_VAL="${LLVM_DIR:-}"
MLIR_DIR_VAL="${MLIR_DIR:-}"

candidates=(
  "$LLVM_DIR_VAL"
  "$MLIR_DIR_VAL"
  "/usr/local/opt/llvm"
  "/usr/local/llvm"
  "/opt/llvm"
  "$HOME/llvm/install"
  "$HOME/.local/llvm"
)

best_llvm=""
for c in "${candidates[@]}"; do
  [[ -z "${c}" ]] && continue
  if [[ -d "$c/lib/cmake/llvm" ]]; then
    best_llvm="$c"
    break
  fi
  if [[ -d "$c" ]]; then
    # Maybe user passed cmake root already
    if [[ -f "$c/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
      best_llvm="$c"
      break
    fi
  fi
  # Check sibling of big project
  if [[ -d "$(pwd)/../mlir-js-parser-big" ]]; then
    while IFS= read -r -d '' d; do
      if [[ -f "$d/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
        best_llvm="${d%/lib/cmake/llvm}"
        break
      fi
    done < <(find "$(pwd)/../mlir-js-parser-big" -type d -path '*/lib/cmake/llvm' -print0 2>/dev/null)
  fi

  # Check sibling llvm-project/build-wasm (our bootstrap default)
  if [[ -z "$best_llvm" && -d "$(pwd)/../llvm-project/build-wasm/lib/cmake/llvm" ]]; then
    best_llvm="$(pwd)/../llvm-project/build-wasm"
  fi
  [[ -n "$best_llvm" ]] && break
done

if [[ -z "$best_llvm" ]]; then
  echo "Could not locate LLVM cmake config. Set LLVM_DIR and MLIR_DIR manually." >&2
  exit 1
fi

echo "export LLVM_DIR=$best_llvm"
echo "export MLIR_DIR=$best_llvm"

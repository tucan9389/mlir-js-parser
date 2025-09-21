# Add MLIR dialects

The default bundle includes Builtin + Func + Arith + SCF. You can extend the set of registered dialects and rebuild the WASM to parse additional MLIR constructs.

This document explains:

- Where to register dialects in C++
- What libraries to link in CMake
- A minimal example (Builtin + one more)

## 1) Register dialects in C++

Edit `cpp/src/parser.cc`. The helper `registerDialects(MLIRContext&)` is the place to add more dialects:

```cpp
#include "mlir/IR/BuiltinDialect.h"
// Example: #include "mlir/Dialect/Func/IR/FuncOps.h"

namespace {
void registerDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::BuiltinDialect>();
  // Example: ctx.getOrLoadDialect<mlir::func::FuncDialect>();
}
}
```

Notes:

- Include the dialect header that provides the dialect type.
- Use `getOrLoadDialect<YourDialect>()` to ensure it is registered with the context.

## 2) Link required MLIR libraries

Open `cpp/CMakeLists.txt` and add the corresponding MLIR dialect libraries to the `mlir_libs` list. For example, to add `FuncDialect`:

```cmake
set(mlir_libs
  MLIRIR
  MLIRParser
  MLIRSupport
  # Add dialect libs below
  MLIRFuncDialect
)
```

The specific library names depend on your LLVM/MLIR version. Common examples:

- `MLIRFuncDialect`
- `MLIRArithDialect`
- `MLIRSCFDialect`
- `MLIRTensorDialect`

If you see link errors, search your LLVM/MLIR build tree for `libMLIR*Dialect*` to determine the correct targets.

## 3) Rebuild

- Native sanity (optional):

```bash
LLVM_DIR=...</path/to/host/llvm> MLIR_DIR=...</path/to/host/llvm> \
  bash scripts/build-native.sh
```

- WebAssembly bundle:

```bash
# Ensure Emscripten is active
LLVM_DIR=...</path/to/wasm/llvm> MLIR_DIR=...</path/to/wasm/llvm> \
  bash scripts/build-wasm.sh
```

Generated files will appear in `wasm/mlir_parser.{js,wasm}`.

## Tips

- Adding dialects increases output size; keep only what you need.
- Many dialects also depend on others (e.g., `Tensor` on `Arith`); link all needed libs.
- If you need advanced passes or serialization (bytecode/asm), you may need to link `MLIRAsmParser` or `MLIRBytecodeReader` as well. This project already links the minimal set for parsing text.

- Source locations (loc): The bundle includes `loc` for operations and block arguments in the JSON output. To test block-argument `loc`, parse a `func.func` with parameters (Func dialect is included by default).

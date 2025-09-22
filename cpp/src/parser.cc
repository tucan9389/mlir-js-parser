#include "parser.h"

#include <cstring>
#include <memory>
#include <string>

// MLIR headers
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Pass/PassManager.h"

// Dialects: func, arith, scf (+ core infra)
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#ifdef HAVE_MLIR_CF_DIALECT
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#endif
#ifdef HAVE_MLIR_MEMREF_DIALECT
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#endif
#ifdef HAVE_MLIR_TENSOR_DIALECT
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#endif
#ifdef HAVE_MLIR_MATH_DIALECT
#include "mlir/Dialect/Math/IR/Math.h"
#endif
// Index dialect (optional)
#ifdef HAVE_MLIR_INDEX_DIALECT
#  if __has_include("mlir/Dialect/Index/IR/IndexDialect.h")
#    include "mlir/Dialect/Index/IR/IndexDialect.h"
#    define HAVE_INDEX_HEADER 1
#  elif __has_include("mlir/Dialect/Index/IR/IndexOps.h")
#    include "mlir/Dialect/Index/IR/IndexOps.h"
#    define HAVE_INDEX_HEADER 1
#  endif
#endif
// Optional/common attrs
// Some environments may have the DLTI library but not expose headers in the include paths.
// Use __has_include to avoid hard errors when headers are missing.
#if defined(HAVE_MLIR_DLTI_DIALECT) && __has_include("mlir/Dialect/DLTI/IR/DLTI.h")
#define HAVE_DLTI_HEADER 1
#include "mlir/Dialect/DLTI/IR/DLTI.h"
#endif
// More dialects (conditionally included)
#ifdef HAVE_MLIR_VECTOR_DIALECT
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#endif
#ifdef HAVE_MLIR_LINALG_DIALECT
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#endif
#ifdef HAVE_MLIR_LLVM_DIALECT
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#endif
#ifdef HAVE_MLIR_SPIRV_DIALECT
#if __has_include("mlir/Dialect/SPIRV/IR/SPIRVDialect.h")
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#else
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#endif
#endif
#ifdef HAVE_MLIR_TRANSFORM_DIALECT
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#endif
// Transform dialect extensions: register structured transform ops (if available).
#if __has_include("mlir/Dialect/Transform/Transforms/TransformDialectExtensions.h")
#include "mlir/Dialect/Transform/Transforms/TransformDialectExtensions.h"
#define HAVE_TRANSFORM_EXTENSIONS_HEADER 1
#elif __has_include("mlir/Dialect/Transform/Transforms/TransformDialectExtension.h")
#include "mlir/Dialect/Transform/Transforms/TransformDialectExtension.h"
#define HAVE_TRANSFORM_EXTENSIONS_HEADER 1
#endif
// Individually packaged Transform dialect extensions: prefer function-based
// registration APIs when headers are present.
#if __has_include("mlir/Dialect/Transform/DebugExtension/DebugExtension.h")
#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"
#define HAVE_TRANSFORM_DEBUG_FN 1
#endif
#if __has_include("mlir/Dialect/Transform/LoopExtension/LoopExtension.h")
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#define HAVE_TRANSFORM_LOOP_FN 1
#endif
#if __has_include("mlir/Dialect/Transform/PDLExtension/PDLExtension.h")
#include "mlir/Dialect/Transform/PDLExtension/PDLExtension.h"
#define HAVE_TRANSFORM_PDL_FN 1
#endif
#if __has_include("mlir/Dialect/Transform/IRDLExtension/IRDLExtension.h")
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtension.h"
#define HAVE_TRANSFORM_IRDL_FN 1
#endif
#if __has_include("mlir/Dialect/Transform/TuneExtension/TuneExtension.h")
#include "mlir/Dialect/Transform/TuneExtension/TuneExtension.h"
#define HAVE_TRANSFORM_TUNE_FN 1
#endif
// IRDL dialect itself (needed when using Transform IRDL extension).
#if defined(HAVE_MLIR_IRDL_DIALECT) && __has_include("mlir/Dialect/IRDL/IR/IRDLDialect.h")
#include "mlir/Dialect/IRDL/IR/IRDLDialect.h"
#define HAVE_IRDL_HEADER 1
#endif
// Dialect-specific TransformOps extension headers (conditionally include if present)
#if __has_include("mlir/Dialect/Linalg/TransformOps/DialectExtension.h")
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#define HAVE_LINALG_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h")
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#define HAVE_TENSOR_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/SCF/TransformOps/SCFTransformOps.h")
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#define HAVE_SCF_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h")
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#define HAVE_MEMREF_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/Func/TransformOps/FuncTransformOps.h")
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#define HAVE_FUNC_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h")
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h"
#define HAVE_DLTI_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h")
#include "mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h"
#define HAVE_SPARSETENSOR_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/GPU/TransformOps/GPUTransformOps.h")
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#define HAVE_GPU_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h")
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#define HAVE_NVGPU_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/Vector/TransformOps/VectorTransformOps.h")
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#define HAVE_VECTOR_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/Affine/TransformOps/AffineTransformOps.h")
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#define HAVE_AFFINE_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h")
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#define HAVE_BUFFERIZATION_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.h")
#include "mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.h"
#define HAVE_ARMNEON_TRANSFORM_EXT 1
#endif
#if __has_include("mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h")
#include "mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h"
#define HAVE_ARMSVE_TRANSFORM_EXT 1
#endif
#ifdef HAVE_MLIR_BUFFERIZATION_DIALECT
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#endif
#ifdef HAVE_MLIR_SPARSE_TENSOR_DIALECT
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#endif
#ifdef HAVE_MLIR_OMP_DIALECT
// Guard OpenMP include as availability varies between builds.
#if defined(HAVE_MLIR_OMP_DIALECT) && __has_include("mlir/Dialect/OpenMP/OpenMPDialect.h")
#define HAVE_OPENMP_HEADER 1
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#endif
#endif
#if defined(HAVE_MLIR_OPENACC_DIALECT) && __has_include("mlir/Dialect/OpenACC/OpenACC.h")
#define HAVE_OPENACC_HEADER 1
#include "mlir/Dialect/OpenACC/OpenACC.h"
#endif
#ifdef HAVE_MLIR_GPU_DIALECT
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#endif
#ifdef HAVE_MLIR_TOSA_DIALECT
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#endif
#ifdef HAVE_MLIR_ASYNC_DIALECT
#include "mlir/Dialect/Async/IR/Async.h"
#endif
#ifdef HAVE_MLIR_EMITC_DIALECT
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#endif
#ifdef HAVE_MLIR_SHAPE_DIALECT
#include "mlir/Dialect/Shape/IR/Shape.h"
#endif
// Affine & Complex
#ifdef HAVE_MLIR_AFFINE_DIALECT
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#endif
#ifdef HAVE_MLIR_COMPLEX_DIALECT
#include "mlir/Dialect/Complex/IR/Complex.h"
#endif
#ifdef HAVE_MLIR_UB_DIALECT
#include "mlir/Dialect/UB/IR/UBOps.h"
#endif
#ifdef HAVE_MLIR_ARMSME_DIALECT
#if __has_include("mlir/Dialect/ArmSME/IR/ArmSME.h")
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#define HAVE_ARMSME_HEADER 1
#endif
#endif
#ifdef HAVE_MLIR_X86VECTOR_DIALECT
#if __has_include("mlir/Dialect/X86Vector/IR/X86VectorOps.h")
#include "mlir/Dialect/X86Vector/IR/X86VectorOps.h"
#define HAVE_X86VECTOR_HEADER 1
#endif
#endif
#ifdef HAVE_MLIR_AMX_DIALECT
#if __has_include("mlir/Dialect/AMX/IR/AMX.h")
#include "mlir/Dialect/AMX/IR/AMX.h"
#define HAVE_AMX_HEADER 1
#endif
#endif

// Register conversion-to-LLVM transform interfaces for common dialects used by
// transform.apply_conversion_patterns.dialect_to_llvm.
#if __has_include("mlir/Conversion/ArithToLLVM/ArithToLLVM.h")
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#define HAVE_ARITH_TO_LLVM_HEADER 1
#endif
#if __has_include("mlir/Conversion/MathToLLVM/MathToLLVM.h")
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#define HAVE_MATH_TO_LLVM_HEADER 1
#endif
#if __has_include("mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h")
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#define HAVE_MEMREF_TO_LLVM_HEADER 1
#endif
#if __has_include("mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h")
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#define HAVE_CF_TO_LLVM_HEADER 1
#endif

// PDL / PDLInterp (pattern description languages)
#ifdef HAVE_MLIR_PDL_DIALECT
#include "mlir/Dialect/PDL/IR/PDL.h"
#endif
#ifdef HAVE_MLIR_PDL_INTERP_DIALECT
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#endif

// MLProgram dialect (light infra used in some corpora)
#ifdef HAVE_MLIR_ML_PROGRAM_DIALECT
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#endif

// Optional testing dialects (register only if available)
#ifdef HAVE_MLIR_TEST_DIALECT
#  if __has_include("mlir/Dialect/Test/IR/TestDialect.h")
#    include "mlir/Dialect/Test/IR/TestDialect.h"
#    define HAVE_TEST_HEADER 1
#  elif __has_include("mlir/IR/TestDialect.h")
#    include "mlir/IR/TestDialect.h"
#    define HAVE_TEST_HEADER 1
#  endif
#endif

// StableHLO external dialects (optional)
#if defined(HAVE_STABLEHLO_DIALECT)
#  if __has_include("stablehlo/dialect/StablehloOps.h")
#    include "stablehlo/dialect/StablehloOps.h"
#    define HAVE_STABLEHLO_HEADER 1
#  endif
#endif
#if defined(HAVE_CHLO_DIALECT)
#  if __has_include("stablehlo/dialect/ChloOps.h")
#    include "stablehlo/dialect/ChloOps.h"
#    define HAVE_CHLO_HEADER 1
#  endif
#endif
#if defined(HAVE_VHLO_DIALECT)
#  if __has_include("stablehlo/dialect/VhloOps.h")
#    include "stablehlo/dialect/VhloOps.h"
#    define HAVE_VHLO_HEADER 1
#  endif
#endif

// LLVM support headers used by the parser
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {
// Fallback stub dialects: register a dialect namespace so the parser accepts
// generic operations like 'stablehlo.foo' even without linking real dialects.
// This significantly reduces "dialect not found" parse errors during scans.
struct StablehloStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "stablehlo"; }
  explicit StablehloStubDialect(MLIRContext *ctx)
      : Dialect("stablehlo", ctx, TypeID::get<StablehloStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct ChloStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "chlo"; }
  explicit ChloStubDialect(MLIRContext *ctx)
      : Dialect("chlo", ctx, TypeID::get<ChloStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct VhloStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "vhlo"; }
  explicit VhloStubDialect(MLIRContext *ctx)
      : Dialect("vhlo", ctx, TypeID::get<VhloStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
// Testing corpora often include `check.*` and `test.*` operations. When the real
// dialects aren't linked, provide permissive stubs to avoid hard parse fails.
struct CheckStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "check"; }
  explicit CheckStubDialect(MLIRContext *ctx)
      : Dialect("check", ctx, TypeID::get<CheckStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct TestStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "test"; }
  explicit TestStubDialect(MLIRContext *ctx)
      : Dialect("test", ctx, TypeID::get<TestStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
// Additional permissive stubs for external or tutorial dialects commonly found
// in corpora, but typically not linked in production builds.
struct ToyStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "toy"; }
  explicit ToyStubDialect(MLIRContext *ctx)
      : Dialect("toy", ctx, TypeID::get<ToyStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct SMTStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "smt"; }
  explicit SMTStubDialect(MLIRContext *ctx)
      : Dialect("smt", ctx, TypeID::get<SMTStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct XEGPUStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "xegpu"; }
  explicit XEGPUStubDialect(MLIRContext *ctx)
      : Dialect("xegpu", ctx, TypeID::get<XEGPUStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct XEVMStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "xevm"; }
  explicit XEVMStubDialect(MLIRContext *ctx)
      : Dialect("xevm", ctx, TypeID::get<XEVMStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct WasmSSAStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "wasmssa"; }
  explicit WasmSSAStubDialect(MLIRContext *ctx)
      : Dialect("wasmssa", ctx, TypeID::get<WasmSSAStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct MPIStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "mpi"; }
  explicit MPIStubDialect(MLIRContext *ctx)
      : Dialect("mpi", ctx, TypeID::get<MPIStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct CUDAFortranStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "cuf"; }
  explicit CUDAFortranStubDialect(MLIRContext *ctx)
      : Dialect("cuf", ctx, TypeID::get<CUDAFortranStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct PtrStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "ptr"; }
  explicit PtrStubDialect(MLIRContext *ctx)
      : Dialect("ptr", ctx, TypeID::get<PtrStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct StandaloneStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "standalone"; }
  explicit StandaloneStubDialect(MLIRContext *ctx)
      : Dialect("standalone", ctx, TypeID::get<StandaloneStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
struct ShardStubDialect : public Dialect {
  static StringRef getDialectNamespace() { return "shard"; }
  explicit ShardStubDialect(MLIRContext *ctx)
      : Dialect("shard", ctx, TypeID::get<ShardStubDialect>()) {
    allowUnknownOperations();
    allowUnknownTypes();
  }
};
// Note: We intentionally do not stub 'check' or 'test' dialects because many
// of their ops rely on custom assembly/attribute parsing; a stub would cause
// harder-to-diagnose parse errors. If available in the MLIR build, we load the
// real dialects below; otherwise we rely on allowUnregisteredDialects.
// Helper to register a minimal set of dialects. Expand here over time.
void registerDialects(MLIRContext &ctx) {
  // If available, wire up Transform dialect extensions via a DialectRegistry.
  // Upstream MLIR often builds these into the main Transform lib; there is
  // no separate lib in many builds, so rely on header presence.
  #if defined(HAVE_MLIR_TRANSFORM_DIALECT)
  {
    DialectRegistry reg;
    #if defined(HAVE_TRANSFORM_EXTENSIONS_HEADER)
    mlir::transform::registerTransformDialectExtension(reg);
    #endif
  // Register optional Transform extensions when available in this MLIR build.
  #if defined(HAVE_TRANSFORM_DEBUG_FN)
  mlir::transform::registerDebugExtension(reg);
  #endif
  #if defined(HAVE_TRANSFORM_LOOP_FN)
  mlir::transform::registerLoopExtension(reg);
  #endif
  #if defined(HAVE_TRANSFORM_PDL_FN)
  mlir::transform::registerPDLExtension(reg);
  #endif
  #if defined(HAVE_TRANSFORM_IRDL_FN)
  mlir::transform::registerIRDLExtension(reg);
  #endif
  #if defined(HAVE_TRANSFORM_TUNE_FN)
  mlir::transform::registerTuneExtension(reg);
  #endif
    // Dialect-specific Transform extensions (conditionally registered)
    #if defined(HAVE_LINALG_TRANSFORM_EXT)
    mlir::linalg::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_TENSOR_TRANSFORM_EXT)
    mlir::tensor::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_SCF_TRANSFORM_EXT)
    mlir::scf::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_MEMREF_TRANSFORM_EXT)
    mlir::memref::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_FUNC_TRANSFORM_EXT)
    mlir::func::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_DLTI_TRANSFORM_EXT)
    mlir::dlti::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_SPARSETENSOR_TRANSFORM_EXT)
    mlir::sparse_tensor::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_GPU_TRANSFORM_EXT)
    mlir::gpu::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_NVGPU_TRANSFORM_EXT)
    mlir::nvgpu::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_VECTOR_TRANSFORM_EXT)
    mlir::vector::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_AFFINE_TRANSFORM_EXT)
    mlir::affine::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_BUFFERIZATION_TRANSFORM_EXT)
    mlir::bufferization::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_ARMNEON_TRANSFORM_EXT)
    mlir::arm_neon::registerTransformDialectExtension(reg);
    #endif
    #if defined(HAVE_ARMSVE_TRANSFORM_EXT)
    mlir::arm_sve::registerTransformDialectExtension(reg);
    #endif

    // Register ConvertToLLVMTransformInterfaces so transform.apply_conversion_patterns
    // can find the providers for common dialects.
    #if defined(HAVE_ARITH_TO_LLVM_HEADER)
    mlir::arith::registerConvertArithToLLVMInterface(reg);
    #endif
  // ConvertToLLVM transform interfaces: register when headers are present.
    #if defined(HAVE_CF_TO_LLVM_HEADER)
    mlir::cf::registerConvertControlFlowToLLVMInterface(reg);
    #endif
  #if defined(HAVE_ARITH_TO_LLVM_HEADER)
  mlir::arith::registerConvertArithToLLVMInterface(reg);
  #endif
  #if defined(HAVE_MATH_TO_LLVM_HEADER)
  mlir::registerConvertMathToLLVMInterface(reg);
  #endif
  #if defined(HAVE_MEMREF_TO_LLVM_HEADER)
  mlir::registerConvertMemRefToLLVMInterface(reg);
  #endif

    ctx.appendDialectRegistry(reg);
  }
  #endif

  ctx.getOrLoadDialect<BuiltinDialect>();
  // Common dialects
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  // Core infrastructure dialects frequently encountered in real IR
  #ifdef HAVE_MLIR_CF_DIALECT
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  #endif
  #ifdef HAVE_MLIR_MEMREF_DIALECT
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  #endif
  #ifdef HAVE_MLIR_TENSOR_DIALECT
  ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();
  #endif
  #ifdef HAVE_MLIR_MATH_DIALECT
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  #endif
  #ifdef HAVE_INDEX_HEADER
  ctx.getOrLoadDialect<mlir::index::IndexDialect>();
  #endif
  // Data layout common attribute dialect (safe to register; no heavy deps)
  #if defined(HAVE_DLTI_HEADER)
  ctx.getOrLoadDialect<mlir::dlti::DLTIDialect>();
  #endif

  #if defined(HAVE_INDEX_HEADER)
  ctx.getOrLoadDialect<mlir::index::IndexDialect>();
  #endif
  #ifdef HAVE_MLIR_VECTOR_DIALECT
  ctx.getOrLoadDialect<mlir::vector::VectorDialect>();
  #endif
  #ifdef HAVE_MLIR_LINALG_DIALECT
  ctx.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  #endif
  #ifdef HAVE_MLIR_LLVM_DIALECT
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  #endif
  #ifdef HAVE_MLIR_SPIRV_DIALECT
  ctx.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
  #endif
  // Load the Transform dialect when available (even if extensions are not
  // linked). Extensions are conditionally registered above.
  #if defined(HAVE_MLIR_TRANSFORM_DIALECT)
  ctx.getOrLoadDialect<mlir::transform::TransformDialect>();
  #endif
  #ifdef HAVE_MLIR_BUFFERIZATION_DIALECT
  ctx.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  #endif
  #ifdef HAVE_MLIR_SPARSE_TENSOR_DIALECT
  ctx.getOrLoadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  #endif
  #ifdef HAVE_MLIR_OMP_DIALECT
  #if defined(HAVE_OPENMP_HEADER)
  ctx.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  #endif
  #endif
  #if defined(HAVE_OPENACC_HEADER)
  ctx.getOrLoadDialect<mlir::acc::OpenACCDialect>();
  #endif
  #ifdef HAVE_MLIR_GPU_DIALECT
  ctx.getOrLoadDialect<mlir::gpu::GPUDialect>();
  #endif
  #ifdef HAVE_MLIR_TOSA_DIALECT
  ctx.getOrLoadDialect<mlir::tosa::TosaDialect>();
  #endif
  #ifdef HAVE_MLIR_ASYNC_DIALECT
  ctx.getOrLoadDialect<mlir::async::AsyncDialect>();
  #endif
  #ifdef HAVE_MLIR_EMITC_DIALECT
  ctx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  #endif
  #ifdef HAVE_MLIR_SHAPE_DIALECT
  ctx.getOrLoadDialect<mlir::shape::ShapeDialect>();
  #endif
  #ifdef HAVE_MLIR_AFFINE_DIALECT
  ctx.getOrLoadDialect<mlir::affine::AffineDialect>();
  #endif
  #ifdef HAVE_MLIR_COMPLEX_DIALECT
  ctx.getOrLoadDialect<mlir::complex::ComplexDialect>();
  #endif
  #ifdef HAVE_MLIR_UB_DIALECT
  ctx.getOrLoadDialect<mlir::ub::UBDialect>();
  #endif
  #ifdef HAVE_IRDL_HEADER
  ctx.getOrLoadDialect<mlir::irdl::IRDLDialect>();
  #endif
  #ifdef HAVE_ARMSME_HEADER
  ctx.getOrLoadDialect<mlir::arm_sme::ArmSMEDialect>();
  #endif
  #ifdef HAVE_X86VECTOR_HEADER
  ctx.getOrLoadDialect<mlir::x86vector::X86VectorDialect>();
  #endif
  #ifdef HAVE_AMX_HEADER
  ctx.getOrLoadDialect<mlir::amx::AMXDialect>();
  #endif

  // PDL dialects (optional)
  #ifdef HAVE_MLIR_PDL_DIALECT
  ctx.getOrLoadDialect<mlir::pdl::PDLDialect>();
  #endif
  #ifdef HAVE_MLIR_PDL_INTERP_DIALECT
  ctx.getOrLoadDialect<mlir::pdl_interp::PDLInterpDialect>();
  #endif

  // MLProgram dialect (optional)
  #ifdef HAVE_MLIR_ML_PROGRAM_DIALECT
  ctx.getOrLoadDialect<mlir::ml_program::MLProgramDialect>();
  #endif

  // External StableHLO family
  #if defined(HAVE_STABLEHLO_HEADER)
  ctx.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  #endif
  #if defined(HAVE_CHLO_HEADER)
  ctx.getOrLoadDialect<mlir::chlo::ChloDialect>();
  #endif
  #if defined(HAVE_VHLO_HEADER)
  ctx.getOrLoadDialect<mlir::vhlo::VhloDialect>();
  #endif
  // If we don't have the real dialects linked, fallback to stub dialects
  #if !defined(HAVE_STABLEHLO_HEADER)
  ctx.getOrLoadDialect<StablehloStubDialect>();
  #endif
  #if !defined(HAVE_CHLO_HEADER)
  ctx.getOrLoadDialect<ChloStubDialect>();
  #endif
  #if !defined(HAVE_VHLO_HEADER)
  ctx.getOrLoadDialect<VhloStubDialect>();
  #endif
  // Provide permissive stubs for `check` and `test` dialect namespaces if the
  // real ones aren't linked in this build. These stubs only make parsing
  // succeed; they don't implement custom asm/attrs.
  ctx.getOrLoadDialect<CheckStubDialect>();
  #if !defined(HAVE_TEST_HEADER)
  ctx.getOrLoadDialect<TestStubDialect>();
  #endif

  // Test dialect: load only if available in the linked MLIR build.
  #if defined(HAVE_TEST_HEADER)
  ctx.getOrLoadDialect<mlir::test::TestDialect>();
  #endif

  // Fallback stub dialects for various external/tutorial dialects that appear
  // in corpora but are typically not present in production builds.
  ctx.getOrLoadDialect<ToyStubDialect>();
  ctx.getOrLoadDialect<SMTStubDialect>();
  ctx.getOrLoadDialect<XEGPUStubDialect>();
  ctx.getOrLoadDialect<XEVMStubDialect>();
  ctx.getOrLoadDialect<WasmSSAStubDialect>();
  ctx.getOrLoadDialect<MPIStubDialect>();
  ctx.getOrLoadDialect<CUDAFortranStubDialect>();
  ctx.getOrLoadDialect<PtrStubDialect>();
  ctx.getOrLoadDialect<StandaloneStubDialect>();
  ctx.getOrLoadDialect<ShardStubDialect>();
}

// Serialize a Type to a string (best-effort).
static std::string typeToString(Type ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  os.flush();
  return s;
}

// Convert an Attribute to a JSON value (best-effort stringification where complex).
static llvm::json::Value attrToJson(Attribute attr) {
  if (!attr)
    return llvm::json::Value(nullptr);
  if (auto i = llvm::dyn_cast<IntegerAttr>(attr)) {
    // Use decimal string (unsigned) to avoid precision issues; represent as string.
    llvm::SmallString<64> buf;
    i.getValue().toString(buf, /*Radix=*/10, /*Signed=*/false);
    return llvm::json::Value(std::string(buf));
  }
  if (auto f = llvm::dyn_cast<FloatAttr>(attr)) {
    std::string s;
    llvm::raw_string_ostream os(s);
    f.getValue().print(os);
    os.flush();
    return llvm::json::Value(s);
  }
  if (auto sattr = llvm::dyn_cast<StringAttr>(attr)) {
    return llvm::json::Value(sattr.getValue().str());
  }
  if (auto ta = llvm::dyn_cast<TypeAttr>(attr)) {
    return llvm::json::Value(typeToString(ta.getValue()));
  }
  if (auto da = llvm::dyn_cast<DictionaryAttr>(attr)) {
    llvm::json::Object obj;
    for (auto it : da) {
      obj[it.getName().str()] = attrToJson(it.getValue());
    }
    return llvm::json::Value(std::move(obj));
  }
  if (auto arr = llvm::dyn_cast<ArrayAttr>(attr)) {
    llvm::json::Array a;
    for (auto v : arr)
      a.push_back(attrToJson(v));
    return llvm::json::Value(std::move(a));
  }
  // Fallback to printed form for unknown/complex attrs.
  std::string s;
  llvm::raw_string_ostream os(s);
  attr.print(os);
  os.flush();
  return llvm::json::Value(s);
}

static llvm::json::Object valueToJson(Value v) {
  llvm::json::Object obj;
  obj["type"] = typeToString(v.getType());
  return obj;
}

// Convert a Location to a minimal JSON object.
// - FileLineColLoc => { file, line, column }
// - Others => { unknown: true }
static llvm::json::Object locToJson(Location loc) {
  llvm::json::Object o;
  if (auto fl = mlir::dyn_cast<FileLineColLoc>(loc)) {
    o["file"] = fl.getFilename().str();
    o["line"] = static_cast<int64_t>(fl.getLine());
    o["column"] = static_cast<int64_t>(fl.getColumn());
    return o;
  }
  o["unknown"] = true;
  return o;
}

static llvm::json::Object opToJson(Operation *op) {
  llvm::json::Object obj;
  // name/opcode
  obj["name"] = op->getName().getStringRef().str();

  // source location (best-effort)
  obj["loc"] = locToJson(op->getLoc());

  // attributes
  llvm::json::Object attrs;
  for (auto &it : op->getAttrs()) {
    attrs[it.getName().str()] = attrToJson(it.getValue());
  }
  obj["attributes"] = std::move(attrs);

  // operands
  llvm::json::Array operands;
  for (auto operand : op->getOperands()) {
    operands.push_back(valueToJson(operand));
  }
  obj["operands"] = std::move(operands);

  // results (types only)
  llvm::json::Array results;
  for (auto res : op->getResults()) {
    llvm::json::Object r;
    r["type"] = typeToString(res.getType());
    results.push_back(std::move(r));
  }
  obj["results"] = std::move(results);

  // regions -> blocks -> arguments/ops
  llvm::json::Array regions;
  for (auto &region : op->getRegions()) {
    llvm::json::Object rj;
    llvm::json::Array blocks;
    for (auto &block : region) {
      llvm::json::Object bj;
      // block arguments
      llvm::json::Array args;
      for (auto arg : block.getArguments()) {
        llvm::json::Object aj = valueToJson(arg);
        aj["loc"] = locToJson(arg.getLoc());
        args.push_back(std::move(aj));
      }
      bj["arguments"] = std::move(args);
      // operations in block
      llvm::json::Array ops;
      for (auto &innerOp : block)
        ops.push_back(opToJson(&innerOp));
      bj["operations"] = std::move(ops);
      blocks.push_back(std::move(bj));
    }
    rj["blocks"] = std::move(blocks);
    regions.push_back(std::move(rj));
  }
  obj["regions"] = std::move(regions);

  return obj;
}
} // namespace

static int mlir_parse_to_string_impl(const char *mlir_text,
                                     bool allowUnregistered,
                                     char *out_buffer,
                                     int out_capacity,
                                     char *err_buffer,
                                     int err_capacity) {
  if (!mlir_text) {
    const char *msg = "input text is null";
    if (err_buffer && err_capacity > 0) {
      std::strncpy(err_buffer, msg, err_capacity - 1);
      err_buffer[err_capacity - 1] = '\0';
    }
    return 1;
  }

  MLIRContext ctx;
  registerDialects(ctx);
  if (allowUnregistered) ctx.allowUnregisteredDialects();

  // Prepare to capture diagnostics with file:line:col.
  std::string diagStr;
  llvm::raw_string_ostream diagOS(diagStr);
  auto handlerId = ctx.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    // Try to prefix with file:line:col if available
    mlir::Location loc = diag.getLocation();
    if (mlir::isa<mlir::FileLineColLoc>(loc)) {
      auto fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
      auto filename = fileLoc.getFilename().str();
      diagOS << filename << ":" << fileLoc.getLine() << ":" << fileLoc.getColumn() << ": ";
    }
    diag.print(diagOS);
    diagOS << '\n';
  });

  // Parse the module from the provided text.
  llvm::SourceMgr sourceMgr;
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(mlir_text, "<input>", false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  // Note: We rely on DiagnosticEngine printing locations from the parsed IR.

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    diagOS.flush();
    ctx.getDiagEngine().eraseHandler(handlerId);
    // Return detailed diagnostics if available.
    const std::string &msg = diagStr.empty() ? std::string("failed to parse MLIR") : diagStr;
    int needed = static_cast<int>(msg.size()) + 1;
    if (err_capacity < needed)
      return -needed; // signal insufficient error buffer capacity
    if (err_buffer && err_capacity > 0) {
      std::memcpy(err_buffer, msg.c_str(), needed);
    }
    return 2;
  }

  ctx.getDiagEngine().eraseHandler(handlerId);

  // Print canonical text to a string.
  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os);
  os.flush();

  int needed = static_cast<int>(result.size()) + 1; // include NUL
  if (out_capacity < needed) {
    // Indicate insufficient capacity by returning negative required size.
    return -needed;
  }

  std::memcpy(out_buffer, result.c_str(), needed);
  return 0;
}

extern "C" int mlir_parse_to_string(const char *mlir_text,
                                     char *out_buffer,
                                     int out_capacity,
                                     char *err_buffer,
                                     int err_capacity) {
  return mlir_parse_to_string_impl(mlir_text, /*allowUnregistered=*/false,
                                   out_buffer, out_capacity, err_buffer, err_capacity);
}

// Extended entrypoint with options (for wasm scanning etc.)
extern "C" int mlir_parse_to_string_opts(const char *mlir_text,
                                          int allow_unregistered,
                                          char *out_buffer,
                                          int out_capacity,
                                          char *err_buffer,
                                          int err_capacity) {
  return mlir_parse_to_string_impl(mlir_text, allow_unregistered != 0,
                                   out_buffer, out_capacity, err_buffer, err_capacity);
}

static int mlir_parse_to_json_impl(const char *mlir_text,
                                   bool allowUnregistered,
                                   char *out_buffer,
                                   int out_capacity,
                                   char *err_buffer,
                                   int err_capacity) {
  if (!mlir_text) {
    const char *msg = "input text is null";
    if (err_buffer && err_capacity > 0) {
      std::strncpy(err_buffer, msg, err_capacity - 1);
      err_buffer[err_capacity - 1] = '\0';
    }
    return 1;
  }

  MLIRContext ctx;
  registerDialects(ctx);
  if (allowUnregistered) ctx.allowUnregisteredDialects();

  // Prepare to capture diagnostics with file:line:col.
  std::string diagStr;
  llvm::raw_string_ostream diagOS(diagStr);
  auto handlerId = ctx.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    mlir::Location loc = diag.getLocation();
    if (mlir::isa<mlir::FileLineColLoc>(loc)) {
      auto fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
      auto filename = fileLoc.getFilename().str();
      diagOS << filename << ":" << fileLoc.getLine() << ":" << fileLoc.getColumn() << ": ";
    }
    diag.print(diagOS);
    diagOS << '\n';
  });

  // Parse the module from the provided text.
  llvm::SourceMgr sourceMgr;
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(mlir_text, "<input>", false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  // Note: We rely on DiagnosticEngine printing locations from the parsed IR.

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    diagOS.flush();
    ctx.getDiagEngine().eraseHandler(handlerId);
    const std::string &msg = diagStr.empty() ? std::string("failed to parse MLIR") : diagStr;
    int needed = static_cast<int>(msg.size()) + 1;
    if (err_capacity < needed)
      return -needed; // signal insufficient error buffer capacity
    if (err_buffer && err_capacity > 0) {
      std::memcpy(err_buffer, msg.c_str(), needed);
    }
    return 2;
  }

  ctx.getDiagEngine().eraseHandler(handlerId);

  // Build JSON from IR (root op is the module itself)
  llvm::json::Object root = opToJson(module.get());
  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);
  os << llvm::formatv("{0}", llvm::json::Value(std::move(root)));
  os.flush();

  int needed = static_cast<int>(jsonStr.size()) + 1; // include NUL
  if (out_capacity < needed) {
    return -needed;
  }
  std::memcpy(out_buffer, jsonStr.c_str(), needed);
  return 0;
}

extern "C" int mlir_parse_to_json(const char *mlir_text,
                                   char *out_buffer,
                                   int out_capacity,
                                   char *err_buffer,
                                   int err_capacity) {
  return mlir_parse_to_json_impl(mlir_text, /*allowUnregistered=*/false,
                                 out_buffer, out_capacity, err_buffer, err_capacity);
}

extern "C" int mlir_parse_to_json_opts(const char *mlir_text,
                                        int allow_unregistered,
                                        char *out_buffer,
                                        int out_capacity,
                                        char *err_buffer,
                                        int err_capacity) {
  return mlir_parse_to_json_impl(mlir_text, allow_unregistered != 0,
                                 out_buffer, out_capacity, err_buffer, err_capacity);
}

extern "C" int mlir_parse_check(const char *mlir_text,
                                 int allow_unregistered,
                                 char *err_buffer,
                                 int err_capacity) {
  if (!mlir_text) {
    const char *msg = "input text is null";
    if (err_buffer && err_capacity > 0) {
      std::strncpy(err_buffer, msg, err_capacity - 1);
      err_buffer[err_capacity - 1] = '\0';
    }
    return 1;
  }

  MLIRContext ctx;
  registerDialects(ctx);
  if (allow_unregistered) ctx.allowUnregisteredDialects();

  std::string diagStr;
  llvm::raw_string_ostream diagOS(diagStr);
  auto handlerId = ctx.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    mlir::Location loc = diag.getLocation();
    if (mlir::isa<mlir::FileLineColLoc>(loc)) {
      auto fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
      diagOS << fileLoc.getFilename().str() << ":" << fileLoc.getLine() << ":" << fileLoc.getColumn() << ": ";
    }
    diag.print(diagOS);
    diagOS << '\n';
  });

  llvm::SourceMgr sourceMgr;
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(mlir_text, "<input>", false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    diagOS.flush();
    ctx.getDiagEngine().eraseHandler(handlerId);
    const std::string &msg = diagStr.empty() ? std::string("failed to parse MLIR") : diagStr;
    int needed = static_cast<int>(msg.size()) + 1;
    if (err_capacity < needed)
      return -needed;
    if (err_buffer && err_capacity > 0)
      std::memcpy(err_buffer, msg.c_str(), needed);
    return 2;
  }
  ctx.getDiagEngine().eraseHandler(handlerId);
  return 0;
}

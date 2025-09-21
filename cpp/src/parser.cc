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

// LLVM support headers used by the parser
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {
// Helper to register a minimal set of dialects. Expand here over time.
void registerDialects(MLIRContext &ctx) {
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
  // Data layout common attribute dialect (safe to register; no heavy deps)
  #if defined(HAVE_DLTI_HEADER)
  ctx.getOrLoadDialect<mlir::dlti::DLTIDialect>();
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
  #ifdef HAVE_MLIR_TRANSFORM_DIALECT
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

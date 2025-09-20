#include "parser.h"

#include <cstring>
#include <memory>
#include <string>

// MLIR headers
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Pass/PassManager.h"

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
  // TODO: add more dialect registrations as needed.
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

static llvm::json::Object opToJson(Operation *op) {
  llvm::json::Object obj;
  // name/opcode
  obj["name"] = op->getName().getStringRef().str();

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
      for (auto arg : block.getArguments())
        args.push_back(valueToJson(arg));
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

extern "C" int mlir_parse_to_string(const char *mlir_text,
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

  // Parse the module from the provided text.
  llvm::SourceMgr sourceMgr;
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(mlir_text, "<input>", false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    const char *msg = "failed to parse MLIR";
    if (err_buffer && err_capacity > 0) {
      std::strncpy(err_buffer, msg, err_capacity - 1);
      err_buffer[err_capacity - 1] = '\0';
    }
    return 2;
  }

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

extern "C" int mlir_parse_to_json(const char *mlir_text,
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

  // Parse the module from the provided text.
  llvm::SourceMgr sourceMgr;
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(mlir_text, "<input>", false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    const char *msg = "failed to parse MLIR";
    if (err_buffer && err_capacity > 0) {
      std::strncpy(err_buffer, msg, err_capacity - 1);
      err_buffer[err_capacity - 1] = '\0';
    }
    return 2;
  }

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

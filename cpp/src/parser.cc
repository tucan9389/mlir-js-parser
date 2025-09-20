#include "parser.h"

#include <cstring>
#include <memory>
#include <string>

// MLIR headers
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Pass/PassManager.h"

// LLVM support headers used by the parser
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

namespace {
// Helper to register a minimal set of dialects. Expand here over time.
void registerDialects(MLIRContext &ctx) {
  ctx.getOrLoadDialect<BuiltinDialect>();
  // TODO: add more dialect registrations as needed.
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

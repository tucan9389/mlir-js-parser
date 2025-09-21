cmake_minimum_required(VERSION 3.20)
project(mlir_parser_min C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM/MLIR config
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(mlir_parser_min STATIC
  src/parser.cc
)

target_include_directories(mlir_parser_min
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(mlir_parser_min PRIVATE
  MLIR_DISABLE_THREADING
)

# Reduce size and require RTTI where needed.
set_property(TARGET mlir_parser_min PROPERTY CXX_STANDARD 17)

# MLIR and LLVM include paths
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Link minimal MLIR components; expand as needed.
llvm_map_components_to_libnames(llvm_libs Support)

# Base required libs
set(mlir_libs
  MLIRIR
  MLIRParser
  MLIRSupport
)

get_filename_component(_mlir_cmake_dir ${MLIR_DIR} DIRECTORY)       # .../lib/cmake
get_filename_component(MLIR_LIB_DIR ${_mlir_cmake_dir} DIRECTORY)   # .../lib

function(append_dialect_if_available lib macro)
  set(_lib_path "${MLIR_LIB_DIR}/lib${lib}.a")
  if(EXISTS ${_lib_path})
    list(APPEND mlir_libs ${lib})
    set(mlir_libs "${mlir_libs}" PARENT_SCOPE)
    target_compile_definitions(mlir_parser_min PRIVATE ${macro}=1)
    message(STATUS "Linking MLIR lib: ${_lib_path} (define ${macro})")
  else()
    message(STATUS "Skipping MLIR lib (not found): ${_lib_path}")
  endif()
endfunction()

# Common dialects (link if present in user's MLIR build), also define macros for conditional registration
append_dialect_if_available(MLIRFuncDialect           HAVE_MLIR_FUNC_DIALECT)
append_dialect_if_available(MLIRArithDialect          HAVE_MLIR_ARITH_DIALECT)
append_dialect_if_available(MLIRSCFDialect            HAVE_MLIR_SCF_DIALECT)
append_dialect_if_available(MLIRControlFlowDialect    HAVE_MLIR_CF_DIALECT)
append_dialect_if_available(MLIRMemRefDialect         HAVE_MLIR_MEMREF_DIALECT)
append_dialect_if_available(MLIRTensorDialect         HAVE_MLIR_TENSOR_DIALECT)
append_dialect_if_available(MLIRMathDialect           HAVE_MLIR_MATH_DIALECT)
append_dialect_if_available(MLIRDLTIDialect           HAVE_MLIR_DLTI_DIALECT)

# Additional common dialects seen in StableHLO corpora
append_dialect_if_available(MLIRVectorDialect         HAVE_MLIR_VECTOR_DIALECT)
append_dialect_if_available(MLIRLinalgDialect         HAVE_MLIR_LINALG_DIALECT)
append_dialect_if_available(MLIRLLVMDialect           HAVE_MLIR_LLVM_DIALECT)
append_dialect_if_available(MLIRSPIRVDialect          HAVE_MLIR_SPIRV_DIALECT)
append_dialect_if_available(MLIRTransformDialect      HAVE_MLIR_TRANSFORM_DIALECT)
append_dialect_if_available(MLIRBufferizationDialect  HAVE_MLIR_BUFFERIZATION_DIALECT)
append_dialect_if_available(MLIRSparseTensorDialect   HAVE_MLIR_SPARSE_TENSOR_DIALECT)
append_dialect_if_available(MLIROpenMPDialect         HAVE_MLIR_OMP_DIALECT)
append_dialect_if_available(MLIRGPUDialect            HAVE_MLIR_GPU_DIALECT)
append_dialect_if_available(MLIRTosaDialect           HAVE_MLIR_TOSA_DIALECT)
append_dialect_if_available(MLIRAsyncDialect          HAVE_MLIR_ASYNC_DIALECT)
append_dialect_if_available(MLIREmitCDialect          HAVE_MLIR_EMITC_DIALECT)
append_dialect_if_available(MLIRShapeDialect          HAVE_MLIR_SHAPE_DIALECT)

# Link libraries
# Note: order matters on some toolchains; MLIR first, then LLVM support

target_link_libraries(mlir_parser_min
  PUBLIC
    ${mlir_libs}
    ${llvm_libs}
)

# Native CLI for sanity checks (only for native builds).
if(NOT EMSCRIPTEN)
  add_executable(mlir_parser_min_cli
    src/cli.cc
  )
  target_link_libraries(mlir_parser_min_cli PRIVATE mlir_parser_min)
endif()

# WASM target (built when using Emscripten toolchain)
if(EMSCRIPTEN)
  add_executable(mlir_parser src/wasm_entry.c)
  target_link_libraries(mlir_parser PRIVATE mlir_parser_min)

  # Emscripten-specific settings
  target_link_options(mlir_parser PRIVATE
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sENVIRONMENT=web,worker,node"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sSTACK_SIZE=5242880" # 5MB stack to tolerate deep recursion/large inputs
  "-sEXPORTED_FUNCTIONS=['_mlir_parse_to_string','_mlir_parse_to_json','_mlir_parse_to_string_opts','_mlir_parse_to_json_opts','_mlir_parse_check','_malloc','_free']"
    "-sEXPORTED_RUNTIME_METHODS=['cwrap','ccall','UTF8ToString']"
    "-sERROR_ON_UNDEFINED_SYMBOLS=0"
  )

  # Provide stronger diagnostics in Debug builds to chase OOB causes
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(mlir_parser PRIVATE
      "-sASSERTIONS=2"
      "-sSAFE_HEAP=1"
      "-sSTACK_OVERFLOW_CHECK=2"
      "-sDEMANGLE_SUPPORT=1"
    )
  endif()

  # Place generated js/wasm next to source wasm/ folder for easy consumption
  set(WASM_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../wasm")
  set_target_properties(mlir_parser PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${WASM_OUT_DIR}"
    OUTPUT_NAME "mlir_parser"
  )
endif()

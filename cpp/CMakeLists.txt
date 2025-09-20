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

set(mlir_libs
  MLIRIR
  MLIRParser
  MLIRSupport
)

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
    "-sEXPORTED_FUNCTIONS=['_mlir_parse_to_string','_mlir_parse_to_json','_malloc','_free']"
    "-sEXPORTED_RUNTIME_METHODS=['cwrap','ccall','UTF8ToString']"
    "-sERROR_ON_UNDEFINED_SYMBOLS=0"
  )

  # Place generated js/wasm next to source wasm/ folder for easy consumption
  set(WASM_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../wasm")
  set_target_properties(mlir_parser PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${WASM_OUT_DIR}"
    OUTPUT_NAME "mlir_parser"
  )
endif()

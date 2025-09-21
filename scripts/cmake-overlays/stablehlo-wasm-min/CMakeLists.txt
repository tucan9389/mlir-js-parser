cmake_minimum_required(VERSION 3.15)

project(stablehlo_wasm_min C CXX)

# Expect MLIR/LLVM to be discoverable via provided config packages.
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Make MLIR/LLVM CMake modules (AddMLIR, AddLLVM, TableGen) available
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# MLIR/LLVM helpers
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Common includes/definitions like upstream does
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories(${STABLEHLO_SOURCE_DIR})
# Prepend binary include paths so generated headers like
# stablehlo/dialect/*.inc are discoverable by all targets.
include_directories(BEFORE ${STABLEHLO_BINARY_DIR})
include_directories(BEFORE ${CMAKE_BINARY_DIR})
include_directories(BEFORE ${CMAKE_BINARY_DIR}/stablehlo)

# Required by StableHLO dialect/CMake
if(NOT DEFINED STABLEHLO_SOURCE_DIR)
  message(FATAL_ERROR "STABLEHLO_SOURCE_DIR must be provided and point to the root of the stablehlo repository")
endif()

# Mirror key variables expected by stablehlo/stablehlo/dialect CMake
set(STABLEHLO_BINARY_DIR ${CMAKE_BINARY_DIR})
set(STABLEHLO_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

# Ensure a headers aggregation target exists; some add_dependencies() refer to it.
if(NOT TARGET mlir-headers)
  add_custom_target(mlir-headers)
endif()

# Provide a minimal stub for StablehloPasses to satisfy dependencies from
# stablehlo/stablehlo/dialect/Serialization.cpp without bringing in transforms.
if(NOT TARGET StablehloPasses)
  add_library(StablehloPasses INTERFACE)
endif()

# Only add the dialects directory; this avoids pulling in integrations/builders, tools, tests, etc.
add_subdirectory(${STABLEHLO_SOURCE_DIR}/stablehlo/dialect ${CMAKE_BINARY_DIR}/stablehlo/dialect)

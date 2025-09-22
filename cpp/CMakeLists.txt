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
# Link essential LLVM components. Support is mandatory; Core and DebugInfoDWARF provide
# DataLayout parsing and DWARF enum utilities referenced by a few dialect utilities.
llvm_map_components_to_libnames(llvm_libs Support Core DebugInfoDWARF)

# Base required libs
set(mlir_libs
  MLIRSupport
)

get_filename_component(_mlir_cmake_dir ${MLIR_DIR} DIRECTORY)       # .../lib/cmake
get_filename_component(MLIR_LIB_DIR ${_mlir_cmake_dir} DIRECTORY)   # .../lib

function(append_dialect_if_available lib macro)
  set(_lib_path "${MLIR_LIB_DIR}/lib${lib}.a")
  if(EXISTS ${_lib_path})
    # WASM builds may aggressively strip unused code; retain critical libs entirely.
    set(use_whole_archive FALSE)
    if(EMSCRIPTEN)
      # Keep interface helpers and utils to ensure parser entrypoints are present.
    if("${lib}" STREQUAL "MLIRFunctionInterfaces" OR
         "${lib}" STREQUAL "MLIRFuncUtils" OR
         "${lib}" STREQUAL "MLIRCallInterfaces" OR
         "${lib}" STREQUAL "MLIRCastInterfaces" OR
         "${lib}" STREQUAL "MLIRSideEffectInterfaces" OR
         "${lib}" STREQUAL "MLIRLoopLikeInterface" OR
         "${lib}" STREQUAL "MLIRViewLikeInterface" OR
         "${lib}" STREQUAL "MLIRDestinationStyleOpInterface" OR
         "${lib}" STREQUAL "MLIRSubsetOpInterface" OR
         "${lib}" STREQUAL "MLIRShapedOpInterfaces" OR
         "${lib}" STREQUAL "MLIRIndexingMapOpInterface" OR
         "${lib}" STREQUAL "MLIRMaskableOpInterface" OR
         "${lib}" STREQUAL "MLIRValueBoundsOpInterface" OR
         "${lib}" STREQUAL "MLIRParallelCombiningOpInterface" OR
         "${lib}" STREQUAL "MLIROpenACCMPCommon" OR
         "${lib}" STREQUAL "MLIRVectorInterfaces" OR
         "${lib}" STREQUAL "MLIRVectorUtils" OR
         "${lib}" STREQUAL "MLIRDialectUtils" OR
         "${lib}" STREQUAL "MLIRDialect" OR
         "${lib}" STREQUAL "MLIRArithUtils" OR
         "${lib}" STREQUAL "MLIRParser" OR
         "${lib}" STREQUAL "MLIRAsmParser" OR
      "${lib}" STREQUAL "MLIRPresburger" OR
         "${lib}" STREQUAL "MLIRTensorUtils" OR
         "${lib}" STREQUAL "MLIRLinalgUtils" OR
         "${lib}" STREQUAL "MLIRIR" OR
         "${lib}" STREQUAL "MLIRTensorDialect" OR
         # Transform dialect core and extensions (needed for registration)
         "${lib}" STREQUAL "MLIRTransformDialect" OR
         "${lib}" STREQUAL "MLIRTransformDialectUtils" OR
         "${lib}" STREQUAL "MLIRTransformDialectInterfaces" OR
         "${lib}" STREQUAL "MLIRTransformDialectTransforms" OR
         "${lib}" STREQUAL "MLIRTransformDebugExtension" OR
         "${lib}" STREQUAL "MLIRTransformLoopExtension" OR
         "${lib}" STREQUAL "MLIRTransformPDLExtension" OR
         "${lib}" STREQUAL "MLIRTransformDialectIRDLExtension" OR
         "${lib}" STREQUAL "MLIRTransformTuneExtension" OR
         # Dialect-specific TransformOps often referenced only via registries
         "${lib}" STREQUAL "MLIRLinalgTransformOps" OR
         "${lib}" STREQUAL "MLIRTensorTransformOps" OR
         "${lib}" STREQUAL "MLIRSCFTransformOps" OR
         "${lib}" STREQUAL "MLIRGPUTransformOps" OR
         "${lib}" STREQUAL "MLIRMemRefTransformOps" OR
         "${lib}" STREQUAL "MLIRFuncTransformOps" OR
         "${lib}" STREQUAL "MLIRDLTITransformOps" OR
         "${lib}" STREQUAL "MLIRSparseTensorTransformOps" OR
         "${lib}" STREQUAL "MLIRNVGPUTransformOps" OR
         "${lib}" STREQUAL "MLIRVectorTransformOps" OR
         "${lib}" STREQUAL "MLIRAffineTransformOps" OR
         "${lib}" STREQUAL "MLIRArmNeonVectorTransformOps" OR
         "${lib}" STREQUAL "MLIRArmSVEVectorTransformOps")
        set(use_whole_archive TRUE)
      endif()
    endif()

    if(use_whole_archive)
      # Wrap the archive with whole-archive flags to prevent dead stripping.
      list(APPEND mlir_libs "-Wl,--whole-archive" ${_lib_path} "-Wl,--no-whole-archive")
      message(STATUS "Linking MLIR lib with --whole-archive: ${_lib_path} (define ${macro})")
    else()
      # Use absolute path to ensure correct linking with wasm-ld/emscripten.
      list(APPEND mlir_libs ${_lib_path})
      message(STATUS "Linking MLIR lib: ${_lib_path} (define ${macro})")
    endif()

  # Propagate updated list back to parent scope without collapsing list semantics.
  set(mlir_libs ${mlir_libs} PARENT_SCOPE)
    target_compile_definitions(mlir_parser_min PRIVATE ${macro}=1)
  else()
    message(STATUS "Skipping MLIR lib (not found): ${_lib_path}")
  endif()
endfunction()

# Common dialects (link if present in user's MLIR build), also define macros for conditional registration
append_dialect_if_available(MLIRParser               HAVE_MLIR_PARSER_LIB)
append_dialect_if_available(MLIRAsmParser            HAVE_MLIR_ASM_PARSER)
append_dialect_if_available(MLIRFuncDialect           HAVE_MLIR_FUNC_DIALECT)
append_dialect_if_available(MLIRArithDialect          HAVE_MLIR_ARITH_DIALECT)
append_dialect_if_available(MLIRSCFDialect            HAVE_MLIR_SCF_DIALECT)
append_dialect_if_available(MLIRControlFlowDialect    HAVE_MLIR_CF_DIALECT)
append_dialect_if_available(MLIRMemRefDialect         HAVE_MLIR_MEMREF_DIALECT)
append_dialect_if_available(MLIRTensorDialect         HAVE_MLIR_TENSOR_DIALECT)
append_dialect_if_available(MLIRMathDialect           HAVE_MLIR_MATH_DIALECT)
append_dialect_if_available(MLIRDLTIDialect           HAVE_MLIR_DLTI_DIALECT)
append_dialect_if_available(MLIRIndexDialect          HAVE_MLIR_INDEX_DIALECT)
append_dialect_if_available(MLIRIRDL                  HAVE_MLIR_IRDL_DIALECT)

# Additional common dialects seen in StableHLO corpora
append_dialect_if_available(MLIRVectorDialect         HAVE_MLIR_VECTOR_DIALECT)
append_dialect_if_available(MLIRLinalgDialect         HAVE_MLIR_LINALG_DIALECT)
append_dialect_if_available(MLIRLLVMDialect           HAVE_MLIR_LLVM_DIALECT)
append_dialect_if_available(MLIRSPIRVDialect          HAVE_MLIR_SPIRV_DIALECT)
append_dialect_if_available(MLIRTransformDialect      HAVE_MLIR_TRANSFORM_DIALECT)
# Transform dialect extensions provide many ops like transform.structured.*
append_dialect_if_available(MLIRTransformDialectExtensions HAVE_MLIR_TRANSFORM_DIALECT_EXTENSIONS)
# Transform dialect support/utilities and individual extension libs (present in recent MLIR builds)
append_dialect_if_available(MLIRTransformDialectUtils           HAVE_MLIR_TRANSFORM_DIALECT_UTILS)
append_dialect_if_available(MLIRTransformDialectInterfaces      HAVE_MLIR_TRANSFORM_DIALECT_INTERFACES)
append_dialect_if_available(MLIRTransformDialectTransforms      HAVE_MLIR_TRANSFORM_DIALECT_TRANSFORMS)
append_dialect_if_available(MLIRTransformDebugExtension         HAVE_MLIR_TRANSFORM_DEBUG_EXTENSION)
append_dialect_if_available(MLIRTransformLoopExtension          HAVE_MLIR_TRANSFORM_LOOP_EXTENSION)
append_dialect_if_available(MLIRTransformPDLExtension           HAVE_MLIR_TRANSFORM_PDL_EXTENSION)
append_dialect_if_available(MLIRTransformDialectIRDLExtension   HAVE_MLIR_TRANSFORM_IRDL_EXTENSION)
append_dialect_if_available(MLIRTransformTuneExtension          HAVE_MLIR_TRANSFORM_TUNE_EXTENSION)

# Dialect-specific Transform Ops (registerable as extensions)
append_dialect_if_available(MLIRLinalgTransformOps              HAVE_MLIR_LINALG_TRANSFORM_OPS)
append_dialect_if_available(MLIRTensorTransformOps              HAVE_MLIR_TENSOR_TRANSFORM_OPS)
append_dialect_if_available(MLIRSCFTransformOps                 HAVE_MLIR_SCF_TRANSFORM_OPS)
append_dialect_if_available(MLIRGPUTransformOps                 HAVE_MLIR_GPU_TRANSFORM_OPS)
append_dialect_if_available(MLIRMemRefTransformOps              HAVE_MLIR_MEMREF_TRANSFORM_OPS)
append_dialect_if_available(MLIRFuncTransformOps                HAVE_MLIR_FUNC_TRANSFORM_OPS)
append_dialect_if_available(MLIRDLTITransformOps                HAVE_MLIR_DLTI_TRANSFORM_OPS)
append_dialect_if_available(MLIRSparseTensorTransformOps        HAVE_MLIR_SPARSE_TENSOR_TRANSFORM_OPS)
append_dialect_if_available(MLIRNVGPUTransformOps               HAVE_MLIR_NVGPU_TRANSFORM_OPS)
append_dialect_if_available(MLIRVectorTransformOps              HAVE_MLIR_VECTOR_TRANSFORM_OPS)
append_dialect_if_available(MLIRAffineTransformOps              HAVE_MLIR_AFFINE_TRANSFORM_OPS)
append_dialect_if_available(MLIRArmNeonVectorTransformOps       HAVE_MLIR_ARMNEON_VECTOR_TRANSFORM_OPS)
append_dialect_if_available(MLIRArmSVEVectorTransformOps        HAVE_MLIR_ARMSVE_VECTOR_TRANSFORM_OPS)
# Additional libs required by Transform extensions
append_dialect_if_available(MLIRLinalgTransforms               HAVE_MLIR_LINALG_TRANSFORMS)
append_dialect_if_available(MLIRVectorTransforms               HAVE_MLIR_VECTOR_TRANSFORMS)
append_dialect_if_available(MLIRBufferizationTransformOps      HAVE_MLIR_BUFFERIZATION_TRANSFORM_OPS)
append_dialect_if_available(MLIRBufferizationTransforms        HAVE_MLIR_BUFFERIZATION_TRANSFORMS)
append_dialect_if_available(MLIRBufferizationDialect  HAVE_MLIR_BUFFERIZATION_DIALECT)
append_dialect_if_available(MLIRSparseTensorDialect   HAVE_MLIR_SPARSE_TENSOR_DIALECT)
append_dialect_if_available(MLIROpenMPDialect         HAVE_MLIR_OMP_DIALECT)
append_dialect_if_available(MLIRGPUDialect            HAVE_MLIR_GPU_DIALECT)
append_dialect_if_available(MLIRNVGPUDialect          HAVE_MLIR_NVGPU_DIALECT)
append_dialect_if_available(MLIRAMDGPUDialect         HAVE_MLIR_AMDGPU_DIALECT)
append_dialect_if_available(MLIRNVVMDialect           HAVE_MLIR_NVVM_DIALECT)
append_dialect_if_available(MLIRROCDLDialect          HAVE_MLIR_ROCDL_DIALECT)
append_dialect_if_available(MLIRTosaDialect           HAVE_MLIR_TOSA_DIALECT)
append_dialect_if_available(MLIRAsyncDialect          HAVE_MLIR_ASYNC_DIALECT)
append_dialect_if_available(MLIREmitCDialect          HAVE_MLIR_EMITC_DIALECT)
append_dialect_if_available(MLIRShapeDialect          HAVE_MLIR_SHAPE_DIALECT)
append_dialect_if_available(MLIRAffineDialect         HAVE_MLIR_AFFINE_DIALECT)
append_dialect_if_available(MLIRComplexDialect        HAVE_MLIR_COMPLEX_DIALECT)
append_dialect_if_available(MLIRUBDialect             HAVE_MLIR_UB_DIALECT)
append_dialect_if_available(MLIRArmNeonDialect        HAVE_MLIR_ARMNEON_DIALECT)
append_dialect_if_available(MLIRArmSVEDialect         HAVE_MLIR_ARMSVE_DIALECT)
append_dialect_if_available(MLIRArmSMEDialect         HAVE_MLIR_ARMSME_DIALECT)

# Additional light-weight infra dialects often present in corpora
append_dialect_if_available(MLIRPDLDialect            HAVE_MLIR_PDL_DIALECT)
append_dialect_if_available(MLIRPDLInterpDialect      HAVE_MLIR_PDL_INTERP_DIALECT)
append_dialect_if_available(MLIRMLProgramDialect      HAVE_MLIR_ML_PROGRAM_DIALECT)

# Optional: Index dialect (core MLIR) if present
append_dialect_if_available(MLIRIndexDialect          HAVE_MLIR_INDEX_DIALECT)

# Optional: MLIR Test dialects (present only when MLIR built with tests)
append_dialect_if_available(MLIRTestDialect           HAVE_MLIR_TEST_DIALECT)
append_dialect_if_available(MLIRCheckDialect          HAVE_MLIR_CHECK_DIALECT)
append_dialect_if_available(MLIRSMTDialect            HAVE_MLIR_SMT_DIALECT)
append_dialect_if_available(MLIROpenACCDialect        HAVE_MLIR_OPENACC_DIALECT)
append_dialect_if_available(MLIRAMXDialect            HAVE_MLIR_AMX_DIALECT)
append_dialect_if_available(MLIRX86VectorDialect      HAVE_MLIR_X86VECTOR_DIALECT)

# Optional: External StableHLO family (provide STABLEHLO_LIB_DIR and STABLEHLO_INCLUDE_DIR)
if(STABLEHLO_LIB_DIR)
  function(append_extlib_if_available lib macro)
    set(_lib_path "${STABLEHLO_LIB_DIR}/lib${lib}.a")
    if(EXISTS ${_lib_path})
      # Link via absolute path to avoid relying on -l and link directories.
      list(APPEND mlir_libs ${_lib_path})
      set(mlir_libs "${mlir_libs}" PARENT_SCOPE)
      if(NOT "${macro}" STREQUAL "")
        target_compile_definitions(mlir_parser_min PRIVATE ${macro}=1)
      endif()
      if(STABLEHLO_INCLUDE_DIR)
        target_include_directories(mlir_parser_min PUBLIC ${STABLEHLO_INCLUDE_DIR})
      endif()
      # Also include the StableHLO build lib dir so generated headers (e.g., stablehlo/dialect/*.inc) are found.
      if(STABLEHLO_LIB_DIR)
        # Add lib dir and its parents so includes like "stablehlo/dialect/*.inc" resolve.
        get_filename_component(_shlo_parent "${STABLEHLO_LIB_DIR}" DIRECTORY)      # .../stablehlo
        get_filename_component(_shlo_grandparent "${_shlo_parent}" DIRECTORY)      # build root
        target_include_directories(mlir_parser_min PUBLIC ${STABLEHLO_LIB_DIR} ${_shlo_parent} ${_shlo_grandparent})
      endif()
      message(STATUS "Linking external lib: ${_lib_path} (define ${macro})")
    else()
      message(STATUS "Skipping external lib (not found): ${_lib_path}")
    endif()
  endfunction()

  # StableHLO dialect libraries (actual names in upstream CMake)
  append_extlib_if_available(StablehloOps        HAVE_STABLEHLO_DIALECT)
  append_extlib_if_available(ChloOps             HAVE_CHLO_DIALECT)
  append_extlib_if_available(VhloOps             HAVE_VHLO_DIALECT)

  # StableHLO support libraries frequently required by the above (no macros needed)
  append_extlib_if_available(StablehloTypeInference  "")
  append_extlib_if_available(StablehloAssemblyFormat "")
  append_extlib_if_available(StablehloBase           "")
  append_extlib_if_available(StablehloBroadcastUtils "")
  append_extlib_if_available(VhloTypes               "")
  append_extlib_if_available(Version                 "")
endif()

# Extra MLIR libs commonly referenced by StableHLO
append_dialect_if_available(MLIRQuantDialect           HAVE_MLIR_QUANT_DIALECT)
append_dialect_if_available(MLIRDataLayoutInterfaces   HAVE_MLIR_DLI)
append_dialect_if_available(MLIRTransformUtils         HAVE_MLIR_TRANSFORM_UTILS)
append_dialect_if_available(MLIRBytecodeWriter         HAVE_MLIR_BYTECODE_WRITER)
append_dialect_if_available(MLIRBytecodeReader         HAVE_MLIR_BYTECODE_READER)
append_dialect_if_available(MLIRControlFlowInterfaces  HAVE_MLIR_CF_INTERFACES)
append_dialect_if_available(MLIRInferTypeOpInterface   HAVE_MLIR_INFER_TYPE_OP_IFACE)
append_dialect_if_available(MLIRFuncUtils              HAVE_MLIR_FUNC_UTILS)
append_dialect_if_available(MLIRCallInterfaces         HAVE_MLIR_CALL_INTERFACES)
append_dialect_if_available(MLIRFunctionInterfaces     HAVE_MLIR_FUNCTION_INTERFACES)
append_dialect_if_available(MLIRCastInterfaces         HAVE_MLIR_CAST_INTERFACES)
append_dialect_if_available(MLIRSideEffectInterfaces   HAVE_MLIR_SIDE_EFFECT_INTERFACES)
append_dialect_if_available(MLIRLoopLikeInterface      HAVE_MLIR_LOOP_LIKE_INTERFACE)
append_dialect_if_available(MLIRViewLikeInterface      HAVE_MLIR_VIEW_LIKE_INTERFACE)
append_dialect_if_available(MLIRDestinationStyleOpInterface HAVE_MLIR_DESTINATION_STYLE_OP_INTERFACE)
append_dialect_if_available(MLIRSubsetOpInterface      HAVE_MLIR_SUBSET_OP_INTERFACE)
append_dialect_if_available(MLIRShapedOpInterfaces     HAVE_MLIR_SHAPED_OP_INTERFACES)
append_dialect_if_available(MLIRTensorUtils            HAVE_MLIR_TENSOR_UTILS)
append_dialect_if_available(MLIRLinalgUtils            HAVE_MLIR_LINALG_UTILS)
append_dialect_if_available(MLIRIndexingMapOpInterface HAVE_MLIR_INDEXING_MAP_OP_INTERFACE)
append_dialect_if_available(MLIRMaskableOpInterface    HAVE_MLIR_MASKABLE_OP_INTERFACE)
append_dialect_if_available(MLIRValueBoundsOpInterface HAVE_MLIR_VALUE_BOUNDS_OP_INTERFACE)
append_dialect_if_available(MLIRParallelCombiningOpInterface HAVE_MLIR_PARALLEL_COMBINING_OP_INTERFACE)
append_dialect_if_available(MLIRVectorInterfaces       HAVE_MLIR_VECTOR_INTERFACES)
append_dialect_if_available(MLIRVectorUtils            HAVE_MLIR_VECTOR_UTILS)
append_dialect_if_available(MLIRDialectUtils           HAVE_MLIR_DIALECT_UTILS)
append_dialect_if_available(MLIRDialect                HAVE_MLIR_DIALECT_LIB)
append_dialect_if_available(MLIRPresburger             HAVE_MLIR_PRESBURGER)
append_dialect_if_available(MLIRAnalysis               HAVE_MLIR_ANALYSIS)
append_dialect_if_available(MLIRIR                     HAVE_MLIR_IR_LIB)
append_dialect_if_available(MLIRArithUtils             HAVE_MLIR_ARITH_UTILS)

# Conversion libraries for ConvertToLLVMTransformInterfaces used by Transform dialect
append_dialect_if_available(MLIRArithToLLVM            HAVE_MLIR_ARITH_TO_LLVM)
append_dialect_if_available(MLIRMathToLLVM             HAVE_MLIR_MATH_TO_LLVM)
append_dialect_if_available(MLIRControlFlowToLLVM      HAVE_MLIR_CF_TO_LLVM)
append_dialect_if_available(MLIRMemRefToLLVM           HAVE_MLIR_MEMREF_TO_LLVM)

# SPIR-V image/sampling interfaces used by some corpora
append_dialect_if_available(MLIRSPIRVImageInterfaces   HAVE_MLIR_SPIRV_IMAGE_INTERFACES)

# OpenACC/OpenMP common atomic interfaces
append_dialect_if_available(MLIROpenACCMPCommon HAVE_MLIR_OPENACCMP_COMMON)

# Optional: expose STABLEHLO_LIB_DIR and STABLEHLO_INCLUDE_DIR as cache vars for convenience
set(STABLEHLO_LIB_DIR "${STABLEHLO_LIB_DIR}" CACHE PATH "Path to StableHLO build lib directory")
set(STABLEHLO_INCLUDE_DIR "${STABLEHLO_INCLUDE_DIR}" CACHE PATH "Path to StableHLO include root (contains stablehlo/dialect/*)")

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

### Testing Strategy Overview

The testing strategy should be structured around four main pillars:

1. **Functional Testing:** Does valid MLIR code get converted into the correct Abstract Syntax Tree (AST) according to the specification?
2. **Robustness Testing:** Does the parser behave stably and report appropriate errors for invalid or exceptional inputs?
3. **Regression Prevention:** Do code changes break existing functionality?
4. **Non-Functional Testing:** Are quality requirements such as performance and usability met?

---

### P0: Core Functionality and Foundation (Critical)

These tests are directly tied to the project's viability. They ensure the most basic operations of the parser and establish a foundation for efficiently adding future tests.

### 1. Test Infrastructure and CI Setup (P0) ✅

- **Description:** Set up a testing framework (e.g., Jest, Vitest) and establish an automated testing pipeline via GitHub Actions.
- **Reason (P0):** Building an automated testing environment from the early stages of development is crucial for continuously maintaining code quality and preventing regressions.

### 2. Unit Tests for Basic MLIR Structure Parsing (P0) ✅

- **Description:** Verify that the core components of MLIR—Operations, Blocks, Regions, and SSA Values—are parsed correctly.
- **Example Scenarios:**
    - Parsing `module` and `func.func` definitions.
    - Parsing the standard Operation format: `%result = "dialect.op"(%arg0) : (type) -> type`.
    - Parsing Block definitions (`^bb0(%arg1: i32):`) and Terminator operations (e.g., `return`).
- **Reason (P0):** This forms the skeleton of MLIR; if this part doesn't work, the parser is useless.

Implemented (initial):

- Added Vitest with a foundational test at `tests/unit/core/basic-structure.test.js`.
- [x] Added a unit test for round-trip text API at `tests/unit/core/parse-mlir-api.test.js`.
- What it checks:
    - Parses `module {}` successfully and asserts minimal JSON shape: name, regions/blocks, operands/results arrays.
    - Reports a readable error for malformed input (negative case).
    - `parseMlir` returns canonical MLIR text for valid input and a non-empty error message for malformed input.
- How to run locally:
    - `npm install`
    - `npm run test:run` (or `npm test` for watch/interactive mode)
- CI:
    - GitHub Actions workflow `tests-node` runs on Node 18/20/22 via `.github/workflows/ci-tests-node.yml`.

Note: Priority labels (P0–P3) are planning tools in this document and issue tracker. They are intentionally not reflected in code or folder names.

### 3. Introduction of Snapshot Testing Framework (P0) ✅

- **Description:** Take valid MLIR code as input, save the generated AST as a JSON snapshot, and compare future outputs against this snapshot when the code changes.
- **Reason (P0):** The AST output by the parser is complex and large. Snapshot testing is the most efficient way to verify the parser's overall behavior and detect unintended changes.

### 4. Verification of Basic Tokenization and Identifier Handling (P0)

Implemented (initial):

- Added Vitest snapshot config at `vitest.snap.config.mjs`.
- Created initial snapshot test: `tests/snapshot/core/module-attributes.test.js`.
- NPM scripts:
    - `npm run test:snap` (watch/local)
    - `npm run test:snap:run` (CI/one-shot)
- CI workflow: `.github/workflows/ci-tests-snapshots.yml` runs on Node 18/20/22.

Notes:

- Tests are conditionally skipped if `wasm/mlir_parser.js` is absent (e.g., on forks without artifacts) to keep CI green.
- Snapshots will evolve as dialect support expands; update snapshots intentionally when parser output changes.

- **Description:** Ensure the Lexer (or equivalent logic) correctly distinguishes MLIR's various identifiers and keywords, and properly handles comments and whitespace.
- **Example Scenarios:**
    - Distinguishing SSA values (`%0`, `%ssa_name`), Block IDs (`^bb0`), and Attribute aliases (`#map`).
    - Verifying that comments (`// ...`, `/* ... */`) inserted in the code are correctly ignored.
- **Reason (P0):** Accurate tokenization is the first step of parsing; an error here will cause the entire parsing process to fail.

---

### P1: Expanding Grammar Coverage and Ensuring Robustness (High Priority)

This phase focuses on supporting the main parts of the MLIR specification and ensuring the parser operates stably even in exceptional situations.

### 5. Type System Verification (P1)

- **Description:** Verify the accurate parsing of MLIR's diverse and complex type system.
- **Example Scenarios:**
    - Built-in types (`i32`, `f64`, `index`).
    - Tensor types (static and dynamic shapes: `tensor<1x?xf32>`, `tensor<*xf32>`).
    - MemRef types (including layout and memory space: `memref<?xi8, strided<[?], offset: ?>>`).
    - Function types (`(i32) -> (f32)`).
- **Reason (P1):** MLIR is a strongly typed IR, and type information is essential for understanding the semantics of operations.

### 6. Attributes Verification (P1)

- **Description:** Parse the various attribute formats that define the static information of an Operation.
- **Example Scenarios:**
    - Basic literals (integer, string, boolean).
    - Parsing `DenseElementsAttribute` (`dense<...>`) (Very important).
    - Array and Dictionary attributes (`{key = "value", list = [1, 2]}`).
- **Reason (P1):** Real-world MLIR code uses attributes extensively, so supporting them is necessary to increase the parser's utility.

### 7. Negative Testing and Error Reporting (P1)

- **Description:** Verify that when syntactically incorrect MLIR code is input, the parser does not crash and reports the expected errors.
- **Example Scenarios:**
    - Mismatched parentheses (`{ ... )`).
    - Missing required tokens (e.g., missing the `:` for type information after an Operation).
    - Unexpected EOF (End Of File).
- **Reason (P1):** Essential for guaranteeing parser stability and providing useful feedback to the user.

### 8. Integration Tests for Major Dialects (P1)

- **Description:** Parse realistic MLIR code composed of combinations of widely used Dialects and verify them with snapshot tests.
- **Example Scenarios:** Testing sample code from `affine` (affine maps, `affine.for`), `scf` (control flow `scf.if`), `memref`, `tensor`, and `llvm` Dialects.
- **Reason (P1):** This can verify interactions between Dialects or complex structural parsing issues that are difficult to detect with unit tests alone.

---

### P2: Improving Usability and Preparing for Future Features (Medium Priority)

These tests enhance the quality of the parser and lay the groundwork for implementing planned visualization or editor features.

### 9. Source Location Tracking Verification (P2)

- **Description:** Verify that all nodes in the generated AST contain accurate location information (line, column) from the original source code.
- **Reason (P2):** Essential for highlighting specific code in visualization tools or reporting the exact location of errors.

### 10. Generic vs. Custom Operation Syntax Testing (P2)

- **Description:** MLIR provides two ways to represent operations (Custom Form and Generic Form). Verify that the parser handles both formats correctly.
- **Example Scenarios:** Ensuring `arith.addi %0, %1 : i32` (Custom) and `"arith.addi"(%0, %1) : (i32, i32) -> i32` (Generic) are parsed into the same AST structure.
- **Reason (P2):** Necessary to fully comply with the MLIR specification.

### 11. AST Traversal and API Usability Testing (P2)

- **Description:** Verify that the public APIs provided by the parser and the utilities for traversing the AST (e.g., Visitor pattern implementations) work correctly.
- **Reason (P2):** The parser's output will be used by other tools (visualizers, analyzers), so the AST structure must be stable and easy to navigate.

### 12. MLIR Official Examples Verification (Conformance Test) (P2)

- **Description:** Collect example files included in the official documentation or test suites of the LLVM/MLIR project and perform parsing tests on them.
- **Reason (P2):** The most reliable way to ensure compatibility with the MLIR specification.

---

### P3: Advanced Features and Optimization (Low Priority)

These tests maximize the completeness of the parser and maintain long-term quality.

### 13. Performance Benchmarking (P3)

- **Description:** Measure the time taken and memory usage required to parse very large MLIR files (thousands of lines or more).
- **Reason (P3):** Responsiveness is important when used in a web environment, and a baseline is needed to detect performance degradation.

### 14. Round-trip Testing (P3)

- **Description:** A necessary test when implementing a Printer (Serializer) feature in the future, which outputs the AST back into MLIR text. Verify that the input and output are semantically identical after going through the `MLIR -> Parser -> AST -> Printer -> MLIR` process.
- **Reason (P3):** The most powerful way to verify that the parser has generated the AST completely without losing information.

### 15. Fuzz Testing (P3)

- **Description:** Inject randomly generated inputs into the parser to find unexpected crashes, infinite loops, or security vulnerabilities.
- **Reason (P3):** Helps discover extreme edge-case bugs that are difficult to find through manual testing.

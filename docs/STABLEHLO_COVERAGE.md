# StableHLO MLIR Coverage Tracking

Top-level doc to run large-scale parsing against a StableHLO MLIR corpus, track coverage over time, and drive dialect enablement.

## Coverage status

| Date | Total | OK | Failed | Success Rate | Size | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-09-20 | 5,449 | 19 | 5,430 | 0.35% | — | Dialects: builtin, func, arith, scf; allowUnregisteredDialects: false |
| 2025-09-20 | 5,449 | 58 | 5,391 | 1.06% | js: 62 KB, wasm: 3.5 MB | allowUnregistered: true (scan mode); Dialects: builtin, func, arith, scf, cf, memref, tensor |
| 2025-09-20 | 5,449 | 515 | 4,934 | 9.46% | js: 62 KB, wasm: 3.5 MB | allowUnregistered: true; stack=5MB; Dialects: builtin, func, arith, scf, cf, memref, tensor |
| 2025-09-20 | 5,449 | 1,473 | 3,976 | 27.03% | js: 62 KB, wasm: 15 MB | allowUnregistered: true; stack=5MB; Dialects: builtin, func, arith, scf, cf, memref, tensor, math, dlti, vector, linalg, llvm, spirv, transform, bufferization, sparse_tensor, omp, gpu, tosa, async, emitc, shape |
| 2025-09-21 | 5,449 | 4,348 | 1,101 | 79.83% | js: 18.7 KB (gz), wasm: 4.12 MB (gz) | StableHLO/CHLO/VHLO linked; allowUnregistered: true; stack=5MB; Dialects: core MLIR + StableHLO family |
| 2025-09-21 | 5,449 | 4,348 | 1,101 | 79.83% | js: 17.7 KB (gz), wasm: 4.02 MB (gz) | Transform extensions wired (header-registry); allowUnregistered: true; stack=5MB |

Update this table after each scan to see progress as dialect coverage improves.

## Practical setup recap (what worked for us)

Goal: parse StableHLO/CHLO/VHLO in WASM without preprocessing.

High-level:

- Build LLVM/MLIR for WASM (Emscripten) and this project’s parser.
- Build StableHLO “ops-only” for WASM using an overlay (avoids integrations/tools).
- Link StableHLO archives by absolute path and add include paths so generated headers resolve.
- Increase WASM stack (we use 5MB) and allow memory growth.

Steps we used end-to-end (bash):

```bash
# 1) Build LLVM/MLIR for WASM (once)
npm run bootstrap:wasm-llvm

export LLVM_DIR=$PWD/../llvm-project/build-wasm
export MLIR_DIR=$PWD/../llvm-project/build-wasm

# 2) Build StableHLO for WASM using the minimal overlay
LLVM_DIR=$LLVM_DIR MLIR_DIR=$MLIR_DIR \
  bash scripts/build-stablehlo-wasm.sh

# Output archives (examples) live under:
#   ../stablehlo/build-wasm/stablehlo/dialect/
# e.g., libStablehloOps.a, libChloOps.a, libVhloOps.a, libStablehloBase.a,
#       libStablehloTypeInference.a, libStablehloAssemblyFormat.a,
#       libStablehloBroadcastUtils.a, libVhloTypes.a, libVersion.a

# 3) Build our WASM and link StableHLO by absolute paths
STABLEHLO_LIB_DIR="$PWD/../stablehlo/build-wasm/stablehlo/dialect" \
STABLEHLO_INCLUDE_DIR="$PWD/../stablehlo" \
  bash scripts/build-wasm.sh

# Artifacts: wasm/mlir_parser.{js,wasm}

# 4) Run snapshot tests to sanity check
npm run test:snap:run

# 5) Scan StableHLO corpus (allow unregistered for broad triage)
node scripts/stablehlo-scan.mjs --dir ../stablehlo --allow-unregistered
```

Notes that mattered:

- We link StableHLO archives via absolute paths (not `-l...`), which was more robust under Emscripten.
- Include paths must cover the StableHLO build tree so generated headers like `stablehlo/dialect/*.inc` resolve. We add:
  - `STABLEHLO_INCLUDE_DIR` (source root), `STABLEHLO_LIB_DIR` (build lib dir), and parent dirs.
- VHLO bytecode references `VhloTypes` — we needed `libVhloTypes.a` to resolve TypeID symbols.
- Emscripten link flags we use: `-sALLOW_MEMORY_GROWTH=1` and `-sSTACK_SIZE=5242880` (5MB) for stability.

## Troubleshooting StableHLO + WASM

- “stablehlo/dialect/... .inc not found”: Ensure StableHLO build output directories are on the include path (we add `STABLEHLO_LIB_DIR` and its parents).
- wasm-ld cannot find `-lStablehloOps`: Link archives by absolute path (handled by CMake logic in `cpp/CMakeLists.txt`).
- Undefined VHLO TypeID or bytecode symbols: Link `libVhloTypes.a`.
- Too many “unknown dialect/op” errors in scans: Run with `--allow-unregistered`, and consider avoiding registration of optional dialects that you won’t link (so unknown ops are accepted instead of hard-failing).
- Runtime “memory access out of bounds”: Often alleviated by allowing unregistered dialects and increasing stack size; also ensure threads are disabled in Emscripten.

## Quick start

Prereqs: wasm artifacts exist (`wasm/mlir_parser.{js,wasm}`), Node 18+.

1) Generate file list (list-only)

```bash
node scripts/stablehlo-scan.mjs --list-only --dir ../stablehlo
```

Outputs:

- `tmp/stablehlo-files.txt` — Absolute paths of all discovered `.mlir` files

1) Run full scan and collect stats

```bash
# Default (strict):
node scripts/stablehlo-scan.mjs --dir ../stablehlo

# Broad triage (allow unknown/unregistered dialects to parse):
node scripts/stablehlo-scan.mjs --dir ../stablehlo --allow-unregistered
```

Outputs:

- `tmp/stablehlo-parse-report.json` — Summary (totals, top errors, top unknown dialects)
- `tmp/stablehlo-error-stats.json` — Error messages with counts (sorted)
- `tmp/stablehlo-dialect-stats.json` — Extracted unknown dialects with counts (sorted)
- `tmp/stablehlo-parse-results.jsonl` — Per-file result lines: `{ file, ok }` or `{ file, ok: false, error }`

## Iteration cycle (dialect coverage)

1) Prepare/build (first time or after dialect changes)

```bash
# Build LLVM/MLIR for wasm (emscripten) once (cached in CI)
bash scripts/bootstrap-llvm-wasm.sh

# Build the wasm parser module (outputs to wasm/)
bash scripts/build-wasm.sh
```

1) Collect corpus and run scan

```bash
# List files only (optional sanity)
node scripts/stablehlo-scan.mjs --list-only --dir ../stablehlo

# Full parse + stats
node scripts/stablehlo-scan.mjs --dir ../stablehlo
```

1) Inspect outputs and decide next actions

- Read `tmp/stablehlo-parse-report.json` for high-level totals and top errors
- Read `tmp/stablehlo-error-stats.json` to identify the most frequent error messages
- Read `tmp/stablehlo-dialect-stats.json` to see which dialects are missing (from messages like `dialect 'X' not found`)
- Review a few failing entries in `tmp/stablehlo-parse-results.jsonl` (search by interesting error keywords)

1) Choose dialects to enable next

From the initial run, notable items include (examples):

- Unknown dialects in corpus: `gpu`, `omp`, `memref`, `fir`, `cf` (counts vary)
- Many files failed with a generic error (e.g., `memory access out of bounds`). This may be a side-effect of halting on unregistered dialects or a wasm runtime limitation.

Recommended next dialects to consider (incrementally):

- Core infra: `cf`, `memref`, `tensor`, `math` (plus `func`, `arith`, `scf` already enabled)
- If present in corpus: `llvm`, `dlti` (common attrs), `gpu`, `omp`, `fir`
- StableHLO/MHLO-specific: if you plan to parse those dialects explicitly, you’ll need their dialect libs (outside LLVM/MLIR core) and to link/register them. Otherwise, consider enabling `allowUnregisteredDialects` (see below) during scanning to gather structure without full registration.

1) Implement dialect enablement

- Register in `cpp/src/parser.cc` (extend `registerDialects`)
- Link libraries in `cpp/CMakeLists.txt` (`mlir_libs` list)
- Rebuild wasm: `bash scripts/build-wasm.sh`
- Commit, re-run scan, and update the coverage table at the top

1) Optional: allow unregistered dialects for scanning mode

For broad corpus triage, allowing unregistered dialects can increase parse progress and improve error fidelity:

- In code: call `ctx.allowUnregisteredDialects();` on the `MLIRContext`
- The scan script also supports `--allow-unregistered` which enables this mode in the wasm parser (no rebuild needed)
- You can gate this via an env var or build flag if you prefer not to change default behavior for production

## Artifacts at a glance

- `tmp/stablehlo-files.txt` — file list
- `tmp/stablehlo-parse-report.json` — summary report (totals, top errors/dialects)
- `tmp/stablehlo-error-stats.json` — errors sorted by frequency
- `tmp/stablehlo-dialect-stats.json` — unknown dialects sorted by frequency
- `tmp/stablehlo-parse-results.jsonl` — per-file results (line-delimited JSON)



## Known notes

- `pthread_create failed`: benign in Node + Emscripten minimal flows; we’ll keep monitoring. We link with threading disabled where possible.
- `memory access out of bounds`: Often correlates with unregistered dialects halting parsing. Enabling key dialects or `allowUnregisteredDialects` typically reduces these.

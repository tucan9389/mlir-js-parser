# StableHLO MLIR Coverage Tracking

Top-level doc to run large-scale parsing against a StableHLO MLIR corpus, track coverage over time, and drive dialect enablement.

## Coverage status

| Date | Total | OK | Failed | Success Rate | Size | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-09-20 | 5,449 | 1,473 | 3,976 | 27.03% | js: 62 KB, wasm: 15 MB | allowUnregistered: true; stack=5MB; Dialects: builtin, func, arith, scf, cf, memref, tensor, math, dlti, vector, linalg, llvm, spirv, transform, bufferization, sparse_tensor, omp, gpu, tosa, async, emitc, shape |
| 2025-09-20 | 5,449 | 515 | 4,934 | 9.46% | js: 62 KB, wasm: 3.5 MB | allowUnregistered: true; stack=5MB; Dialects: builtin, func, arith, scf, cf, memref, tensor |
| 2025-09-20 | 5,449 | 58 | 5,391 | 1.06% | js: 62 KB, wasm: 3.5 MB | allowUnregistered: true (scan mode); Dialects: builtin, func, arith, scf, cf, memref, tensor |
| 2025-09-20 | 5,449 | 19 | 5,430 | 0.35% | — | Dialects: builtin, func, arith, scf; allowUnregisteredDialects: false |

Update this table after each scan to see progress as dialect coverage improves.

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

## CI tie-in (optional)

You can add a CI job that:

1) Builds wasm artifacts (as in snapshot CI)
2) Runs the scan (with a smaller representative subset if needed)
3) Uploads `tmp/*.json` as artifacts

This helps track regressions/improvements across PRs.

## Known notes

- `pthread_create failed`: benign in Node + Emscripten minimal flows; we’ll keep monitoring. We link with threading disabled where possible.
- `memory access out of bounds`: Often correlates with unregistered dialects halting parsing. Enabling key dialects or `allowUnregisteredDialects` typically reduces these.

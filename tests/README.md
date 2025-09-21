# Tests layout

This repository organizes tests by type and runtime, avoiding internal priority labels in the codebase. Priorities (e.g., P0–P3) are tracked in documentation and issues—not in folder names.

Folders & conventions:

- `unit/` — Fast tests for isolated behavior.
  - Example: `unit/core/basic-structure.test.js` (parses `module {}` and basic error handling)
- `integration/` — Cross-component scenarios and multi-dialect cases (often with snapshots).
- `fixtures/` — Reusable MLIR inputs.
- `snapshots/` — Snapshot outputs (when snapshot testing is introduced).
- `performance/` — Benchmarks and size tracking (optional, separate CI job recommended).
- `regression/` — Reproducers for historical bugs with links to issues/PRs.

Runtime scopes:

- Node tests are executed by the `tests-node` workflow (`.github/workflows/ci-tests-node.yml`).
- If browser-based tests are introduced later, use `web/`-prefixed folders under `integration/` or a sibling tree, and a workflow like `ci-tests-web.yml`.

Naming:

- Test files: `*.test.js` (ESM). Prefer descriptive names, e.g., `tokenization-identifiers.test.js`.
- Keep tests deterministic, small, and runtime-agnostic where possible.

Run locally:

- `npm run test:run` — one-off run
- `npm test` — watch/interactive mode

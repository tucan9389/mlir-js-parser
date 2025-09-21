import { describe, it, expect } from 'vitest';
import { createParserModule } from '../../../wasm/bindings.js';

let ModuleFactory = null;
try {
  const mod = await import('../../../wasm/mlir_parser.js');
  ModuleFactory = mod.default ?? mod;
} catch (e) {
  // swallow, will skip when not present
}

// Snapshot: ensure stable AST for a module with dict attributes

describe('Snapshot: module with attributes', () => {
  it('matches snapshot for attributes dictionary', async () => {
    if (!ModuleFactory) return; // skip if wasm artifact not available
    const { parseMlirJson } = await createParserModule(ModuleFactory);
    const input = 'module attributes { test.meta = { a = 1 : i32, msg = "hi" } } {}';
    const res = parseMlirJson(input);

    expect(res.ok).toBe(true);
    // Ensure keys we care about exist before snapshotting
    expect(res.json).toBeTypeOf('object');
    expect(res.json.name).toBe('builtin.module');
    // Snapshot entire JSON to catch structural changes early
    expect(res.json).toMatchSnapshot();
  });
});

import { describe, it, expect } from 'vitest';
import { createParserModule } from '../../../wasm/bindings.js';

let ModuleFactory = null;
try {
  const mod = await import('../../../wasm/mlir_parser.js');
  ModuleFactory = mod.default ?? mod;
} catch (_) {
  // no wasm artifact -> tests will be skipped
}

function normalizeLoc(obj) {
  // Make snapshot stable: ensure key order and only keep known loc shape
  if (obj && typeof obj === 'object') {
    if (obj.loc && typeof obj.loc === 'object') {
      const { file, line, column, unknown } = obj.loc;
      obj.loc = unknown ? { unknown: true } : { file, line, column };
    }
    if (Array.isArray(obj.regions)) {
      for (const r of obj.regions || []) {
        for (const b of r.blocks || []) {
          for (const arg of b.arguments || []) normalizeLoc(arg);
          for (const op of b.operations || []) normalizeLoc(op);
        }
      }
    }
  }
  return obj;
}

describe('Snapshot: func.func with block-arg loc', () => {
  it('parses a simple function and captures argument loc', async () => {
    if (!ModuleFactory) return; // skip
    const { parseMlirJson } = await createParserModule(ModuleFactory);
    const src = `module {
      func.func @foo(%x: i32, %y: i32) {
        %0 = arith.addi %x, %y : i32
        func.return
      }
    }`;
    const res = parseMlirJson(src);
    expect(res.ok).toBe(true);
    const json = normalizeLoc(res.json);

    // Quick asserts before snapshot
    expect(json.name).toBe('builtin.module');
    const block0 = json.regions[0].blocks[0];
    // Inside module body: expect one operation (func.func)
    expect(block0.operations.length).toBeGreaterThanOrEqual(1);
    const funcOp = block0.operations.find(o => o.name === 'func.func');
    expect(funcOp).toBeTruthy();
    const entry = funcOp.regions[0].blocks[0];
    expect(entry.arguments.length).toBe(2);
    // Both block args should have loc
    expect(entry.arguments[0].loc).toBeTypeOf('object');
    expect(entry.arguments[1].loc).toBeTypeOf('object');

    // Snapshot entire JSON (normalized) for stability
    expect(json).toMatchSnapshot();
  });
});

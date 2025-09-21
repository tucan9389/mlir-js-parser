import { describe, it, expect } from 'vitest';
import { createParserModule } from '../../../wasm/bindings.js';

let ModuleFactory = null;
try {
  const mod = await import('../../../wasm/mlir_parser.js');
  ModuleFactory = mod.default ?? mod;
} catch (_) {
  // skip when wasm artifact is not available
}

describe('parseMlirCheck API', () => {
  it('returns ok on valid module', async () => {
    if (!ModuleFactory) return;
    const { parseMlirCheck } = await createParserModule(ModuleFactory);
    const res = parseMlirCheck('module {}');
    expect(res.ok).toBe(true);
  });

  it('returns error with location on malformed input', async () => {
    if (!ModuleFactory) return;
    const { parseMlirCheck } = await createParserModule(ModuleFactory);
    const bad = 'module {';
    const res = parseMlirCheck(bad);
    expect(res.ok).toBe(false);
    expect(typeof res.error).toBe('string');
    expect(res.error.length).toBeGreaterThan(0);
    expect(/:\d+:\d+/.test(res.error)).toBe(true);
  });
});

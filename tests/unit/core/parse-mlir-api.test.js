import { describe, it, expect } from 'vitest';
import { createParserModule } from '../../../wasm/bindings.js';

let ModuleFactory = null;
try {
  const mod = await import('../../../wasm/mlir_parser.js');
  ModuleFactory = mod.default ?? mod;
} catch (e) {
  // skip when wasm artifact is not available
}

describe('parseMlir (text roundtrip)', () => {
  it('returns canonical MLIR text for a simple module', async () => {
    if (!ModuleFactory) return; // skip if wasm artifact not available
    const { parseMlir } = await createParserModule(ModuleFactory);
    const input = 'module { }';
    const res = parseMlir(input);
    expect(res.ok).toBe(true);
    expect(typeof res.module).toBe('string');
    // Canonical printing usually collapses whitespace; ensure it contains 'module' and braces
    expect(res.module.includes('module')).toBe(true);
    expect(res.module.includes('{')).toBe(true);
    expect(res.module.includes('}')).toBe(true);
  });

  it('reports an error for malformed input', async () => {
    if (!ModuleFactory) return; // skip if wasm artifact not available
    const { parseMlir } = await createParserModule(ModuleFactory);
    const bad = 'module {'; // missing closing brace
    const res = parseMlir(bad);
    expect(res.ok).toBe(false);
    expect(typeof res.error).toBe('string');
    expect(res.error.length).toBeGreaterThan(0);
    // Diagnostics should include line and column like ":1:..."
    expect(/:\d+:\d+/.test(res.error)).toBe(true);
  });
});

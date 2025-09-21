import { describe, it, expect } from 'vitest';
import { createParserModule } from '../../../wasm/bindings.js';

let ModuleFactory = null;
try {
  const mod = await import('../../../wasm/mlir_parser.js');
  ModuleFactory = mod.default ?? mod;
} catch (_) {
  // skip when wasm artifact is not available
}

describe('allow-unregistered option', () => {
  it('accepts unknown ops when allowUnregistered=true', async () => {
    if (!ModuleFactory) return; // skip if wasm artifact not available
    const { parseMlirJson } = await createParserModule(ModuleFactory);
    const mlir = `module { %0 = "some.unknown_op"() : () -> i32 }`;
    const strict = parseMlirJson(mlir, { allowUnregistered: false });
    const lax = parseMlirJson(mlir, { allowUnregistered: true });

    expect(strict.ok).toBe(false);
    expect(typeof strict.error).toBe('string');
    expect(lax.ok).toBe(true);
    expect(lax.json).toBeTypeOf('object');
  });
});

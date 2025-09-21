import { describe, it, expect } from 'vitest';
import { createParserModule } from '../../../wasm/bindings.js';
import ModuleFactory from '../../../wasm/mlir_parser.js';

// Contract: parseMlirJson returns { ok: true, json } or { ok: false, error }
// Minimal module parsing and error path

describe('Basic MLIR structure', () => {
  it('parses empty module to a well-formed JSON shape', async () => {
    const { parseMlirJson } = await createParserModule(ModuleFactory);
    const input = 'module {}';
    const res = parseMlirJson(input);

    expect(res).toBeTruthy();
    expect(typeof res.ok).toBe('boolean');
    expect(res.ok).toBe(true);

    const j = res.json;
    // Minimal structural assertions—future snapshots will expand this
    expect(j).toBeTypeOf('object');
    expect(j.name).toBe('builtin.module');
    expect(Array.isArray(j.operands)).toBe(true);
    expect(Array.isArray(j.results)).toBe(true);
    expect(Array.isArray(j.regions)).toBe(true);

    // Regions -> Blocks structure exists
    expect(j.regions.length).toBe(1);
    const region0 = j.regions[0];
    expect(Array.isArray(region0.blocks)).toBe(true);
    expect(region0.blocks.length).toBe(1);
    const block0 = region0.blocks[0];
    expect(Array.isArray(block0.operations)).toBe(true);
    expect(Array.isArray(block0.arguments)).toBe(true);
  });

  it('reports a readable error on malformed input', async () => {
    const { parseMlirJson } = await createParserModule(ModuleFactory);
    const bad = 'module { '; // missing closing brace
    const res = parseMlirJson(bad);

    expect(res).toBeTruthy();
    expect(res.ok).toBe(false);
    expect(typeof res.error).toBe('string');
    // Don’t assert exact message; toolchains vary. Just ensure non-empty.
    expect(res.error.length).toBeGreaterThan(0);
  });
});

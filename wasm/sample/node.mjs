// Node sample to test the WASM parser.
// Adjust the import path to the generated Emscripten module file.

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createParserModule } from '../bindings.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// The generated Emscripten JS loader (e.g., mlir_parser.js) should be placed adjacent or referenced here.
const emscriptenFactory = (await import('../mlir_parser.js')).default;

const { parseMlir } = await createParserModule(emscriptenFactory);

const sample = `module {}`;
const res = parseMlir(sample);
console.log(res);

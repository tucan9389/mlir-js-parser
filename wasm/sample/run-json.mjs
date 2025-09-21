// Parse a sample MLIR string to JSON and print the result
import { createParserModule } from '../bindings.js';

const ModuleFactory = (await import('../mlir_parser.js')).default;
const { parseMlirJson } = await createParserModule(ModuleFactory);

const input = 'module attributes { test.meta = { a = 1 : i32, msg = "hi" } } {}';
const res = parseMlirJson(input);
console.log(JSON.stringify(res, null, 2));

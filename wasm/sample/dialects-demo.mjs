// Demonstrate parsing MLIR using func/arith/scf and print JSON
import { createParserModule } from '../bindings.js';

const ModuleFactory = (await import('../mlir_parser.js')).default;
const { parseMlirJson } = await createParserModule(ModuleFactory);

const src = `module {
  func.func @branch(%a: i32, %b: i32) -> i32 {
    %cond = arith.cmpi sgt, %a, %b : i32
    %res = scf.if %cond -> i32 {
      %sum = arith.addi %a, %b : i32
      scf.yield %sum : i32
    } else {
      %diff = arith.subi %a, %b : i32
      scf.yield %diff : i32
    }
    func.return %res : i32
  }
}`;

const out = parseMlirJson(src);
if (!out.ok) {
  console.error('Parse failed:', out.error);
  process.exit(1);
}
console.log(JSON.stringify(out.json, null, 2));

#!/usr/bin/env node
import fs from 'node:fs';
import zlib from 'node:zlib';

function fmtBytes(n) {
  return `${n} B (~${(n/1024).toFixed(1)} KiB)`;
}

function measure(file) {
  const b = fs.readFileSync(file);
  const gz = zlib.gzipSync(b);
  const br = zlib.brotliCompressSync(b);
  return { raw: b.length, gz: gz.length, br: br.length };
}

// Optional label via CLI arg or env
const label = process.argv[2] || process.env.SIZE_LABEL || 'current';

const files = {
  js: 'wasm/mlir_parser.js',
  wasm: 'wasm/mlir_parser.wasm',
  bindings: 'wasm/bindings.js',
};

const m = {
  js: measure(files.js),
  wasm: measure(files.wasm),
  bindings: measure(files.bindings),
};

const total = {
  raw: m.js.raw + m.wasm.raw + m.bindings.raw,
  gz: m.js.gz + m.wasm.gz + m.bindings.gz,
  br: m.js.br + m.wasm.br + m.bindings.br,
};

console.log('Current bundle sizes');
console.log('---------------------');
console.log(`${files.js}: raw ${fmtBytes(m.js.raw)}, gz ${fmtBytes(m.js.gz)}, br ${fmtBytes(m.js.br)}`);
console.log(`${files.wasm}: raw ${fmtBytes(m.wasm.raw)}, gz ${fmtBytes(m.wasm.gz)}, br ${fmtBytes(m.wasm.br)}`);
console.log(`${files.bindings}: raw ${fmtBytes(m.bindings.raw)}, gz ${fmtBytes(m.bindings.gz)}, br ${fmtBytes(m.bindings.br)}`);
console.log('TOTAL:', `raw ${fmtBytes(total.raw)}, gz ${fmtBytes(total.gz)}, br ${fmtBytes(total.br)}`);

console.log('\nMarkdown row (paste into README table):');
const jsGz = m.js.gz + m.bindings.gz;
const jsBr = m.js.br + m.bindings.br;
console.log(`| ${label} | ${fmtBytes(m.wasm.raw)} | ${fmtBytes(m.wasm.gz)} | ${fmtBytes(m.wasm.br)} | ${fmtBytes(jsGz)} | ${fmtBytes(total.gz)} | ${fmtBytes(total.br)} |`);

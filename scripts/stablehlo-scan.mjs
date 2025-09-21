#!/usr/bin/env node
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function parseArgs(argv) {
  // Default dir candidates: env, ../stablehlo (from repo root), ./stablehlo
  const cwd = process.cwd();
  const candidates = [
    process.env.STABLEHLO_DIR,
    path.resolve(cwd, '../stablehlo'),
    path.resolve(cwd, 'stablehlo'),
  ].filter(Boolean);

  let defaultDir = candidates.find(d => fs.existsSync(d)) || candidates[0];
  if (!defaultDir) defaultDir = path.resolve(cwd, '../stablehlo');

  const args = { dir: defaultDir, outDir: path.resolve(cwd, 'tmp'), listOnly: false, allowUnregistered: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--dir') {
      args.dir = path.resolve(argv[++i]);
    } else if (a === '--out') {
      args.outDir = path.resolve(argv[++i]);
    } else if (a === '--list-only') {
      args.listOnly = true;
    } else if (a === '--allow-unregistered') {
      args.allowUnregistered = true;
    } else if (a === '--help' || a === '-h') {
      console.log('Usage: node scripts/stablehlo-scan.mjs [--dir <path>] [--out <path>] [--list-only] [--allow-unregistered]');
      process.exit(0);
    }
  }
  return args;
}

async function *walk(dir) {
  const entries = await fsp.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      yield *walk(p);
    } else if (e.isFile() && p.endsWith('.mlir')) {
      yield p;
    }
  }
}

function ensureDirSync(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function normalizeError(msg) {
  if (!msg) return 'unknown error';
  const firstLine = String(msg).split(/\r?\n/)[0];
  // Drop leading file:line:col: prefix if present
  const cleaned = firstLine.replace(/^.*?:\d+:\d+:\s*/, '');
  return cleaned.trim().toLowerCase();
}

function extractUnknownDialect(msg) {
  // Known shapes:
  // 1) "'stablehlo.add' op is unknown" => capture stablehlo
  // 2) "dialect `gpu' not found for custom op 'gpu.module'" => capture gpu
  // 3) "...'none' attribute created with unregistered dialect... #\"dlti\"<...>" => capture dlti (best-effort)

  let m = msg.match(/'([a-zA-Z0-9_]+)\.[a-zA-Z0-9_]+'\s+op\s+/);
  if (m) return m[1];

  m = msg.match(/dialect [`'"]([a-zA-Z0-9_]+)[`'"]\s+not\s+found/i);
  if (m) return m[1];

  // Best-effort: look for #"name"< or !"name"< prefixes in printed attrs/types
  m = msg.match(/[#!]["']([a-zA-Z0-9_]+)["']</);
  if (m) return m[1];

  return null;
}

async function main() {
  const { dir, outDir, listOnly, allowUnregistered } = parseArgs(process.argv);
  ensureDirSync(outDir);

  // Collect all .mlir files
  const files = [];
  try {
    for await (const f of walk(dir)) files.push(f);
  } catch (e) {
    console.error('Failed to read directory:', dir);
    console.error(e.message || e);
    process.exit(2);
  }

  const listFile = path.join(outDir, 'stablehlo-files.txt');
  await fsp.writeFile(listFile, files.join('\n') + '\n', 'utf8');
  console.log(`Wrote file list (${files.length}) to ${path.relative(process.cwd(), listFile)}`);

  if (listOnly) {
    console.log('List-only mode: skipping parse.');
    process.exit(0);
  }

  // Load WASM parser (ensure artifacts exist)
  const bindingsPath = path.resolve(__dirname, '../wasm/bindings.js');
  const wasmPath = path.resolve(__dirname, '../wasm/mlir_parser.js');
  if (!fs.existsSync(wasmPath)) {
    console.error('wasm/mlir_parser.js not found. Build wasm first (bash scripts/build-wasm.sh)');
    process.exit(3);
  }

  const { createParserModule } = await import(pathToFileURLCompat(bindingsPath));
  const ModuleFactory = (await import(pathToFileURLCompat(wasmPath))).default;
  const { parseMlirCheck } = await createParserModule(ModuleFactory);

  const results = [];
  const errorCounts = new Map();
  const dialectCounts = new Map();

  let okCount = 0, failCount = 0;
  const start = Date.now();
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (i % 100 === 0) {
      const elapsed = ((Date.now() - start) / 1000).toFixed(1);
      console.log(`[${i}/${files.length}] elapsed ${elapsed}s`);
    }
    try {
      const text = await fsp.readFile(file, 'utf8');
      const res = parseMlirCheck(text, { allowUnregistered });
      if (res.ok) {
        okCount++;
        results.push({ file, ok: true });
      } else {
        failCount++;
        const norm = normalizeError(res.error);
        const prev = errorCounts.get(norm) || 0;
        errorCounts.set(norm, prev + 1);

        const d = extractUnknownDialect(norm);
        if (d) {
          const dv = dialectCounts.get(d) || 0;
          dialectCounts.set(d, dv + 1);
        }
        results.push({ file, ok: false, error: norm });
      }
    } catch (e) {
      failCount++;
      const norm = normalizeError(e.message || String(e));
      const prev = errorCounts.get(norm) || 0;
      errorCounts.set(norm, prev + 1);
      results.push({ file, ok: false, error: norm });
    }
  }

  const errStats = Array.from(errorCounts.entries())
    .map(([message, count]) => ({ message, count }))
    .sort((a, b) => b.count - a.count);

  const diaStats = Array.from(dialectCounts.entries())
    .map(([dialect, count]) => ({ dialect, count }))
    .sort((a, b) => b.count - a.count);

  const report = {
    dir,
    total: files.length,
    ok: okCount,
    failed: failCount,
    topErrors: errStats.slice(0, 50),
    topUnknownDialects: diaStats.slice(0, 50),
  };

  await fsp.writeFile(path.join(outDir, 'stablehlo-parse-report.json'), JSON.stringify(report, null, 2));
  await fsp.writeFile(path.join(outDir, 'stablehlo-parse-results.jsonl'), results.map(r => JSON.stringify(r)).join('\n') + '\n');
  await fsp.writeFile(path.join(outDir, 'stablehlo-error-stats.json'), JSON.stringify(errStats, null, 2));
  await fsp.writeFile(path.join(outDir, 'stablehlo-dialect-stats.json'), JSON.stringify(diaStats, null, 2));

  console.log(`Done. OK: ${okCount}, Failed: ${failCount}`);
  console.log(`- ${path.relative(process.cwd(), path.join(outDir, 'stablehlo-parse-report.json'))}`);
  console.log(`- ${path.relative(process.cwd(), path.join(outDir, 'stablehlo-error-stats.json'))}`);
  console.log(`- ${path.relative(process.cwd(), path.join(outDir, 'stablehlo-dialect-stats.json'))}`);
}

function pathToFileURLCompat(p) {
  // Build a file:// URL string for dynamic import
  const u = new URL('file:');
  u.pathname = path.resolve(p).split(path.sep).map(encodeURIComponent).join('/');
  if (!u.pathname.startsWith('/')) u.pathname = '/' + u.pathname;
  return u.href;
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});

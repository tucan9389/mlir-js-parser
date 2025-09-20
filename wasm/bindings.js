// Minimal JS wrapper to call the C API from the Emscripten-generated module.
// Assumes the Emscripten build exports the function `mlir_parse_to_string`.

export async function createParserModule(moduleFactory) {
  // moduleFactory is the Emscripten Module factory (e.g., import from generated .js)
  const maybeModule = await moduleFactory();
  const Module = (maybeModule && typeof maybeModule.then === 'function')
    ? await maybeModule
    : maybeModule;
  if (Module && Module.ready && typeof Module.ready.then === 'function') {
    await Module.ready;
  }

  function parseMlir(text) {
    const c_mlir = Module.cwrap('mlir_parse_to_string', 'number', ['string', 'number', 'number', 'number', 'number']);

    // Allocate output and error buffers (start with 64KB each; grow on demand)
    let outCap = 64 * 1024;
    let errCap = 8 * 1024;
    let outPtr = Module._malloc(outCap);
    let errPtr = Module._malloc(errCap);

    try {
      let rc = c_mlir(text, outPtr, outCap, errPtr, errCap);
      if (rc === 0) {
        const outStr = Module.UTF8ToString(outPtr);
        return { ok: true, module: outStr };
      }
      if (rc < 0) {
        const need = -rc;
        Module._free(outPtr);
        outPtr = Module._malloc(need);
        outCap = need;
        rc = c_mlir(text, outPtr, outCap, errPtr, errCap);
        if (rc === 0) {
          const outStr = Module.UTF8ToString(outPtr);
          return { ok: true, module: outStr };
        }
      }
      const errStr = Module.UTF8ToString(errPtr) || 'parse failed';
      return { ok: false, error: errStr };
    } finally {
      Module._free(outPtr);
      Module._free(errPtr);
    }
  }

  return { Module, parseMlir };
}

// UTF8ToString is provided by Emscripten EXPORTED_RUNTIME_METHODS

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
        // Grow both buffers to be safe (API doesn't distinguish which one was short)
        Module._free(outPtr);
        Module._free(errPtr);
        outPtr = Module._malloc(need);
        errPtr = Module._malloc(need);
        outCap = need;
        errCap = need;
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

  function parseMlirJson(text) {
    const c_mlir_json = Module.cwrap('mlir_parse_to_json', 'number', ['string', 'number', 'number', 'number', 'number']);

    let outCap = 64 * 1024;
    let errCap = 8 * 1024;
    let outPtr = Module._malloc(outCap);
    let errPtr = Module._malloc(errCap);

    try {
      let rc = c_mlir_json(text, outPtr, outCap, errPtr, errCap);
      if (rc === 0) {
        const outStr = Module.UTF8ToString(outPtr);
        try {
          const json = JSON.parse(outStr);
          return { ok: true, json };
        } catch (e) {
          return { ok: false, error: 'invalid json from parser' };
        }
      }
      if (rc < 0) {
        const need = -rc;
        // Grow both buffers to be safe
        Module._free(outPtr);
        Module._free(errPtr);
        outPtr = Module._malloc(need);
        errPtr = Module._malloc(need);
        outCap = need;
        errCap = need;
        rc = c_mlir_json(text, outPtr, outCap, errPtr, errCap);
        if (rc === 0) {
          const outStr = Module.UTF8ToString(outPtr);
          try {
            const json = JSON.parse(outStr);
            return { ok: true, json };
          } catch (e) {
            return { ok: false, error: 'invalid json from parser' };
          }
        }
      }
      const errStr = Module.UTF8ToString(errPtr) || 'parse failed';
      return { ok: false, error: errStr };
    } finally {
      Module._free(outPtr);
      Module._free(errPtr);
    }
  }

  return { Module, parseMlir, parseMlirJson };
}

// UTF8ToString is provided by Emscripten EXPORTED_RUNTIME_METHODS

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

  function parseMlir(text, opts = {}) {
    const allow = !!opts.allowUnregistered;
    const fnName = allow ? 'mlir_parse_to_string_opts' : 'mlir_parse_to_string';
    const sig = allow ? ['string', 'number', 'number', 'number', 'number', 'number'] : ['string', 'number', 'number', 'number', 'number'];
    const c_mlir = Module.cwrap(fnName, 'number', sig);

    // Allocate output and error buffers (start with 64KB each; grow on demand)
    let outCap = 64 * 1024;
    let errCap = 8 * 1024;
    let outPtr = Module._malloc(outCap);
    let errPtr = Module._malloc(errCap);

    try {
      let rc = allow
        ? c_mlir(text, /*allow_unregistered=*/1, outPtr, outCap, errPtr, errCap)
        : c_mlir(text, outPtr, outCap, errPtr, errCap);
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
        rc = allow
          ? c_mlir(text, /*allow_unregistered=*/1, outPtr, outCap, errPtr, errCap)
          : c_mlir(text, outPtr, outCap, errPtr, errCap);
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

  function parseMlirJson(text, opts = {}) {
    const allow = !!opts.allowUnregistered;
    const fnName = allow ? 'mlir_parse_to_json_opts' : 'mlir_parse_to_json';
    const sig = allow ? ['string', 'number', 'number', 'number', 'number', 'number'] : ['string', 'number', 'number', 'number', 'number'];
    const c_mlir_json = Module.cwrap(fnName, 'number', sig);

    let outCap = 64 * 1024;
    let errCap = 8 * 1024;
    let outPtr = Module._malloc(outCap);
    let errPtr = Module._malloc(errCap);

    try {
      let rc = allow
        ? c_mlir_json(text, /*allow_unregistered=*/1, outPtr, outCap, errPtr, errCap)
        : c_mlir_json(text, outPtr, outCap, errPtr, errCap);
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
        rc = allow
          ? c_mlir_json(text, /*allow_unregistered=*/1, outPtr, outCap, errPtr, errCap)
          : c_mlir_json(text, outPtr, outCap, errPtr, errCap);
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

  function parseMlirCheck(text, opts = {}) {
    const allow = !!opts.allowUnregistered;
    const c_check = Module.cwrap('mlir_parse_check', 'number', ['string', 'number', 'number', 'number']);
    let errCap = 4096;
    let errPtr = Module._malloc(errCap);
    try {
      let rc = c_check(text, allow ? 1 : 0, errPtr, errCap);
      if (rc === 0) return { ok: true };
      if (rc < 0) {
        const need = -rc;
        Module._free(errPtr);
        errPtr = Module._malloc(need);
        errCap = need;
        rc = c_check(text, allow ? 1 : 0, errPtr, errCap);
        if (rc === 0) return { ok: true };
      }
      const errStr = Module.UTF8ToString(errPtr) || 'parse failed';
      return { ok: false, error: errStr };
    } finally {
      Module._free(errPtr);
    }
  }

  return { Module, parseMlir, parseMlirJson, parseMlirCheck };
}

// UTF8ToString is provided by Emscripten EXPORTED_RUNTIME_METHODS

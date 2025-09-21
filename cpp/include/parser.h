#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success; non-zero on error. On success, writes a canonical
// MLIR string into the provided buffer (UTF-8). If the buffer is too small,
// returns a negative value with the required size.
int mlir_parse_to_string(const char* mlir_text,
                         char* out_buffer,
                         int out_capacity,
                         char* err_buffer,
                         int err_capacity);

// Variant with options: set allow_unregistered to non-zero to call
// MLIRContext::allowUnregisteredDialects() before parsing.
int mlir_parse_to_string_opts(const char* mlir_text,
                              int allow_unregistered,
                              char* out_buffer,
                              int out_capacity,
                              char* err_buffer,
                              int err_capacity);

// Parses MLIR text and serializes the IR to a structured JSON string.
// Returns 0 on success; non-zero on error. If the output buffer is too small,
// returns a negative value with the required size.
// JSON schema (simplified, actual fields):
// {
//   "name": "builtin.module",               // operation name (e.g., dialect.op)
//   "loc": { "file": string, "line": number, "column": number } | { "unknown": true },
//   "attributes": { ... },                   // dictionary of attributes
//   "operands": [ { "type": string }, ... ],
//   "results":  [ { "type": string }, ... ],
//   "regions": [
//     { "blocks": [
//         { "arguments": [ { "type": string, "loc": { ... } }, ... ],
//           "operations": [ { /* recursively the same object shape */ } ]
//         }
//       ]
//     }
//   ]
// }
int mlir_parse_to_json(const char* mlir_text,
                       char* out_buffer,
                       int out_capacity,
                       char* err_buffer,
                       int err_capacity);

// Variant with options (see above for semantics of allow_unregistered).
int mlir_parse_to_json_opts(const char* mlir_text,
                            int allow_unregistered,
                            char* out_buffer,
                            int out_capacity,
                            char* err_buffer,
                            int err_capacity);

// Lightweight parse-only check. Returns 0 if parse succeeds, non-zero otherwise.
// On failure, writes diagnostics to err_buffer (same negative size convention).
int mlir_parse_check(const char* mlir_text,
                     int allow_unregistered,
                     char* err_buffer,
                     int err_capacity);

#ifdef __cplusplus
}
#endif

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

// Parses MLIR text and serializes the IR to a structured JSON string.
// Returns 0 on success; non-zero on error. If the output buffer is too small,
// returns a negative value with the required size.
// JSON schema (simplified):
// {
//   "op": "builtin.module",
//   "name": "module",
//   "attributes": { ... },
//   "regions": [
//     { "blocks": [
//         { "arguments": [ {"type": "i32"}, ... ],
//           "operations": [ { "name": "...", "attributes": {...}, "operands": [ {"type": "..."}, ... ], "results": ["..."], "regions": [...] } ]
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

#ifdef __cplusplus
}
#endif

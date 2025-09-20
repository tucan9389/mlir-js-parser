#include <iostream>
#include <string>
#include "parser.h"

int main() {
  const char *input = "module {}";
  char out[64 * 1024];
  char err[4096];
  int rc = mlir_parse_to_string(input, out, sizeof(out), err, sizeof(err));
  if (rc == 0) {
    std::cout << out << "\n";
    return 0;
  }
  if (rc < 0) {
    std::cerr << "Output buffer too small; need " << -rc << " bytes\n";
    return 2;
  }
  std::cerr << "Parse error: " << err << "\n";
  return 1;
}

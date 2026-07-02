// Native host harness for the embedded LSI filter: runs the *exact* MCU code
// (dtfit_lsi.h, float32) over the embedded real-data test vector and prints each
// estimate, so it can be diffed against the float64 golden (tools/embed_lsi.py)
// without the board. Compile:
//   clang++ -O2 -ffp-contract=off -I<firmware/nano_lsi_onboard> test_lsi.cpp -o test_lsi
#include <cstdio>
#include "dtfit_lsi.h"
#include "lsi_testvec.h"

int main() {
  LsiFilter filt;
  float p0[LSI_N];
  p0[0] = LSI_Y[0];
  for (int k = 1; k < LSI_N; k++) p0[k] = 0.0f;
  filt.reset(p0);
  for (int i = 0; i < LSI_NT; i++) {
    if (filt.update(LSI_T[i], LSI_Y[i])) {
      printf("VAL %d %.7f %.9f\n", i, filt.p[0], filt.p[1]);
    }
  }
  printf("footprint %u\n", (unsigned)sizeof(LsiFilter));
  return 0;
}

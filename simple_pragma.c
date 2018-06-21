#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void vadd_16_avx2(const float *x, int32_t x_off, const float *y,
                  int32_t y_off, float *z, int32_t z_off) {
  for (size_t i = 0; i < 16; ++i) {
    z[z_off + i] = y[y_off + i] + x[x_off + i];
  }
}

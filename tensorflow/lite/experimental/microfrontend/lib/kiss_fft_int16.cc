#include <cstdint>

#include "kiss_fft_common.h"

#define FIXED_POINT 16
namespace kissfft_fixed16 {
#include "../../micro/tools/make/downloads/kissfft/kiss_fft.c"
#include "../../micro/tools/make/downloads/kissfft/tools/kiss_fftr.c"
}  // namespace kissfft_fixed16
#undef FIXED_POINT

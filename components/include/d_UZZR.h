#ifndef GPU_UZZR
#define GPU_UZZR

#include <type_traits>

template <typename T>
static __device__ inline bool d_UZZR(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  using U = std::make_unsigned_t<T>;
  constexpr int bits = sizeof(U) * 8;

  U* const in_t = (U*)in;
  U* const out_t = (U*)out;
  const int size = csize / sizeof(U);
  const int tid = threadIdx.x;

  // convert from unsigned integer to zig-zag encoded with radius = 2^{sizeof(T)*8 - 1}
  for (int i = tid; i < size; i+=TPB){
    const U x = in_t[i];
    const U high = x >> (bits - 1);
    const U flip = high - U(1);

    out_t[i] = (x << 1) ^ flip;
  }


  // copy leftover bytes
  if constexpr (sizeof(U) > 1) {
    const int extra = csize % sizeof(U);
    if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  }

  return true;
}

template <typename T>
static __device__ inline void d_iUZZR(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  using U = std::make_unsigned_t<T>;
  constexpr int bits = sizeof(U) * 8;
  constexpr U msb = U(1) << (bits - 1);

  U* const in_t = (U*)in;
  U* const out_t = (U*)out;
  const int size = csize / sizeof(U);
  const int tid = threadIdx.x;

  // convert from radius zig-zag encoding back to the original unsigned bit pattern
  for (int i = tid; i < size; i += TPB) {
    const U y = in_t[i];
    const U mask = U(0) - (y & U(1));
    const U negative = (~y) >> 1;
    const U positive = (y >> 1) | msb;

    out_t[i] = (negative & mask) | (positive & ~mask);
  }

  // copy leftover bytes
  if constexpr (sizeof(U) > 1) {
    const int extra = csize % sizeof(U);
    if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  }
}


#endif

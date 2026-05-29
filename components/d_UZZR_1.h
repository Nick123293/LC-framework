#include "include/d_UZZR.h"


static __device__ inline bool d_UZZR_1(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  return d_UZZR<byte>(csize, in, out, temp);
}


static __device__ inline void d_iUZZR_1(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  d_iUZZR<byte>(csize, in, out, temp);
}

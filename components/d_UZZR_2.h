#include "include/d_UZZR.h"


static __device__ inline bool d_UZZR_2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  return d_UZZR<unsigned short>(csize, in, out, temp);
}


static __device__ inline void d_iUZZR_2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  d_iUZZR<unsigned short>(csize, in, out, temp);
}

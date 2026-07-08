/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#ifndef GPU_RZE
#define GPU_RZE


#include "d_zero_elimination.h"
#include "d_repetition_elimination.h"
#include "bitmap_pyramid.h"


template <typename T>
static __device__ inline bool d_RZE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int extra = csize % sizeof(T);
  const int avail = CS - 2 - extra;
  const int bits = 8 * sizeof(T);
  constexpr int BitmapRange = lc_detail::BitmapPyramid<T>::first_range;
  constexpr int LastOffset = lc_detail::BitmapPyramid<T>::last_offset;
  constexpr int LastRange = lc_detail::BitmapPyramid<T>::last_range;

  // zero out end of bitmap
  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (csize < CS) {
    for (int i = csize / bits + tid; i < BitmapRange; i += TPB) {
      bitmap[i] = 0;
    }
    __syncthreads();
  }

  // copy non-zero values and generate bitmap
  int wpos = 0;
  if (size > 0) d_ZEencode((T*)in, size, (T*)out, wpos, (T*)bitmap, temp_w);
  wpos *= sizeof(T);
  if (wpos >= avail) return false;
  __syncthreads();

  // check if not all zeros
  if (wpos != 0) {
    if (!lc_detail::d_encode_re_bitmap_pyramid<0, BitmapRange>(bitmap, out, wpos, avail, temp_w)) return false;

    // output last level of bitmap
    if (wpos >= avail - LastRange) return false;
    if (tid < LastRange) {
      out[wpos + tid] = bitmap[LastOffset + tid];
    }
    wpos += LastRange;
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[wpos + tid] = in[csize - extra + tid];
  }

  // output old csize and update csize
  const int new_size = wpos + 2 + extra;
  if (tid == 0) {
    out[new_size - 2] = csize;  // bottom byte
    out[new_size - 1] = csize >> 8;  // second byte
  }
  csize = new_size;
  return true;
}


template <typename T>
static __device__ inline void d_iRZE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;  // second byte
  csize |= in[--rpos];  // bottom byte
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  constexpr int BitmapRange = lc_detail::BitmapPyramid<T>::first_range;
  constexpr int LastOffset = lc_detail::BitmapPyramid<T>::last_offset;
  constexpr int LastRange = lc_detail::BitmapPyramid<T>::last_range;
  assert(TPB >= 256);

  // copy leftover byte
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[rpos - extra + tid];
    rpos -= extra;
  }

  if (rpos == 0) {
    // all zeros
    T* const out_t = (T*)out;
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = 0;
    }
  } else {
    int* const temp_w = (int*)temp;
    byte* const bitmap = (byte*)&temp_w[WS];

    // iteratively decompress bitmaps
    rpos -= LastRange;
    if (tid < LastRange) bitmap[LastOffset + tid] = in[rpos + tid];
    __syncthreads();
    lc_detail::d_decode_re_bitmap_pyramid<BitmapRange, LastOffset, LastRange>(bitmap, in, rpos, temp_w);

    // copy non-zero values based on bitmap
    if (size > 0) d_ZEdecode(size, (T*)in, (T*)bitmap, (T*)out, temp_w);
  }
}


#endif

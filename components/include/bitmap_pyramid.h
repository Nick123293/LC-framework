/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.
*/

#ifndef LC_BITMAP_PYRAMID
#define LC_BITMAP_PYRAMID

#ifdef USE_GPU
template <typename T, int maxsize, bool check>
static __device__ inline bool d_REencode(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w);

template <typename T, int maxsize>
static __device__ inline void d_REdecode(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w);
#endif

namespace lc_detail {

template <int Range, bool Continue = (Range >= 8)>
struct BitmapPyramidBytesImpl;

template <int Range>
struct BitmapPyramidBytesImpl<Range, true> {
  static constexpr int value = Range + BitmapPyramidBytesImpl<Range / 8>::value;
};

template <int Range>
struct BitmapPyramidBytesImpl<Range, false> {
  static constexpr int value = Range;
};

template <int Range>
struct BitmapPyramidBytes {
  static_assert(Range > 0, "bitmap range must be positive");
  static constexpr int value = BitmapPyramidBytesImpl<Range>::value;
};

template <int Offset, int Range, bool Continue = (Range >= 8)>
struct BitmapPyramidLastOffsetImpl;

template <int Offset, int Range>
struct BitmapPyramidLastOffsetImpl<Offset, Range, true> {
  static constexpr int value = BitmapPyramidLastOffsetImpl<Offset + Range, Range / 8>::value;
};

template <int Offset, int Range>
struct BitmapPyramidLastOffsetImpl<Offset, Range, false> {
  static constexpr int value = Offset;
};

template <int Offset, int Range>
struct BitmapPyramidLastOffset {
  static constexpr int value = BitmapPyramidLastOffsetImpl<Offset, Range>::value;
};

template <int Range, bool Continue = (Range >= 8)>
struct BitmapPyramidLastRangeImpl;

template <int Range>
struct BitmapPyramidLastRangeImpl<Range, true> {
  static constexpr int value = BitmapPyramidLastRangeImpl<Range / 8>::value;
};

template <int Range>
struct BitmapPyramidLastRangeImpl<Range, false> {
  static constexpr int value = Range;
};

template <int Range>
struct BitmapPyramidLastRange {
  static constexpr int value = BitmapPyramidLastRangeImpl<Range>::value;
};

template <typename T>
struct BitmapPyramid {
  static_assert(CS % (8 * sizeof(T)) == 0, "LC_CHUNK_SIZE must contain a whole bitmap byte for this element size");
  static constexpr int first_range = CS / (8 * sizeof(T));
  static_assert(first_range > 0, "LC_CHUNK_SIZE is too small for this element size");
  static constexpr int bytes = BitmapPyramidBytes<first_range>::value;
  static constexpr int last_offset = BitmapPyramidLastOffset<0, first_range>::value;
  static constexpr int last_range = BitmapPyramidLastRange<first_range>::value;
};

#ifdef USE_GPU
template <int Offset, int Range>
static __device__ inline bool d_encode_re_bitmap_pyramid(byte* bitmap, byte* out, int& wpos, const int avail, int* temp_w)
{
  if constexpr (Range >= 8) {
    int cnt = avail - wpos;
    if (!d_REencode<byte, Range, true>(&bitmap[Offset], Range, &out[wpos], cnt, &bitmap[Offset + Range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();
    return d_encode_re_bitmap_pyramid<Offset + Range, Range / 8>(bitmap, out, wpos, avail, temp_w);
  }
  return true;
}

template <int Range>
static __device__ inline int d_count_bitmap_ones(const byte* bitmap)
{
  int sum = 0;
  for (int base = 0; base < Range * 8; base += TPB) {
    const int idx = base + threadIdx.x;
    sum += __syncthreads_count((idx < Range * 8) && ((bitmap[idx / 8] >> (idx & 7)) & 1));
  }
  return sum;
}

template <int FirstRange, int Offset, int Range>
static __device__ inline void d_decode_re_bitmap_pyramid(byte* bitmap, byte* in, int& rpos, int* temp_w)
{
  if constexpr (Range < FirstRange) {
    constexpr int PrevRange = Range * 8;
    constexpr int PrevOffset = Offset - PrevRange;
    rpos -= d_count_bitmap_ones<Range>(&bitmap[Offset]);
    d_REdecode<byte, PrevRange>(PrevRange, &in[rpos], &bitmap[Offset], &bitmap[PrevOffset], temp_w);
    __syncthreads();
    d_decode_re_bitmap_pyramid<FirstRange, PrevOffset, PrevRange>(bitmap, in, rpos, temp_w);
  }
}
#endif

}  // namespace lc_detail

#endif

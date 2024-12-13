// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/top_p_sampling_kernel.h"

#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 11060
#define CUDA_BFLOAT16_AVAILABLE
#include <cuda_bf16.h>
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"

#include "paddle/common/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/top_k_function_cuda.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

#ifdef PADDLE_WITH_CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

#ifdef PADDLE_WITH_HIP
#define GPU(str) hip##str
#else
#define GPU(str) cu##str
#endif

PHI_DECLARE_bool(use_air_topp);

namespace phi {

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 12000
template <typename T, typename IdxT = int, typename AccT = T>
struct alignas(128) Counter {
  T const* in;
  IdxT const* inIdx;

  IdxT oriLen;

  AccT sum;
  IdxT len;
  float p;

  IdxT previousLen;

  typename cub::Traits<T>::UnsignedBits kthValueBits;

  alignas(128) IdxT filterCnt;

  alignas(128) uint32_t finishedBlockCnt;
};

template <typename IntType>
constexpr __host__ __device__ IntType ceilDiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}

template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b) {
  return ceilDiv(a, b) * b;
}

/**
 * This function calculate the bufLen, which is the size of buffer.
 * When the number of candidates for next pass exceeds the bufLen, we choose not
 * to store the candidates. Otherwise, we will load candidates from the original
 * input data.
 */
template <typename T, typename IdxT>
__host__ __device__ IdxT calcBufLen(IdxT len) {
  IdxT constexpr ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);

  IdxT bufLen = len / (ratio * 8);
  bufLen = alignTo(bufLen, 256);
  return bufLen;
}

template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calcNumPasses() {
  return ceilDiv<int>(sizeof(T) * 8, BitsPerPass);
}

template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddleIn(T key,
                                                           bool selectMin) {
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
  bits = cub::Traits<T>::TwiddleIn(bits);
  if (!selectMin) {
    bits = ~bits;
  }
  return bits;
}

template <typename T>
__device__ T twiddleOut(typename cub::Traits<T>::UnsignedBits bits,
                        bool selectMin) {
  if (!selectMin) {
    bits = ~bits;
  }
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

template <int BitsPerPass>
__host__ __device__ constexpr int calcNumBuckets() {
  return 1 << BitsPerPass;
}

template <typename T, int BitsPerPass, int Pass>
__device__ constexpr int calcStartBit() {
  constexpr int tmpBit = sizeof(T) * 8 - (Pass + 1) * BitsPerPass;

  constexpr int startBit = tmpBit < 0 ? 0 : tmpBit;
  return startBit;
}

template <typename T, int BitsPerPass, int Pass>
__device__ constexpr uint32_t calcMask() {
  static_assert(BitsPerPass <= 31);
  constexpr int numBits = calcStartBit<T, BitsPerPass, Pass - 1>() -
                          calcStartBit<T, BitsPerPass, Pass>();
  return (1 << numBits) - 1;
}

/**
 * Find the bucket based on the radix
 */
template <typename T, int BitsPerPass>
__device__ int calcBucket(T x, int startBit, uint32_t mask, bool selectMin) {
  return (twiddleIn(x, selectMin) >> startBit) & mask;
}

/**
 *  Replace histogram with its own prefix sum (step 2 in `airTopPSampling`
 * description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(IdxT volatile* histogram, IdxT* histogramOut) {
  int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
  if constexpr (numBuckets >= BlockSize) {
    static_assert(numBuckets % BlockSize == 0);
    int constexpr itemsPerThread = numBuckets / BlockSize;
    typedef cub::
        BlockLoad<IdxT, BlockSize, itemsPerThread, cub::BLOCK_LOAD_TRANSPOSE>
            BlockLoad;
    typedef cub::
        BlockStore<IdxT, BlockSize, itemsPerThread, cub::BLOCK_STORE_TRANSPOSE>
            BlockStore;
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

    __shared__ union {
      typename BlockLoad::TempStorage load;
      typename BlockScan::TempStorage scan;
      typename BlockStore::TempStorage store;
    } tempStorage;

    IdxT threadData[itemsPerThread];  // NOLINT

    BlockLoad(tempStorage.load).Load(histogram, threadData);
    __syncthreads();

    BlockScan(tempStorage.scan).InclusiveSum(threadData, threadData);
    __syncthreads();

    BlockStore(tempStorage.store).Store(histogramOut, threadData);
  } else {
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;

    IdxT threadData = 0;
    if (threadIdx.x < numBuckets) {
      threadData = histogram[threadIdx.x];
    }

    BlockScan(tempStorage).InclusiveSum(threadData, threadData);
    __syncthreads();

    if (threadIdx.x < numBuckets) {
      histogramOut[threadIdx.x] = threadData;
    }
  }
}

template <typename T, int BitsPerPass, int NumBuckets, int Pass>
__device__ __forceinline__ void filterAndHistogram(const T* in_buffer,
                                                   const int* in_idx_buffer,
                                                   T* out_buffer,
                                                   int* out_idx_buffer,
                                                   T* out_scores,
                                                   int64_t* out_ids,
                                                   int previous_len,
                                                   Counter<T>* counter,
                                                   T* histogram,
                                                   int* count_histogram,
                                                   T* histogram_shm,
                                                   int* count_histogram_shm,
                                                   const bool early_stop) {
  // scan and filter
  constexpr int start_bit = calcStartBit<T, BitsPerPass, Pass>();
  const uint32_t mask = calcMask<T, BitsPerPass, Pass>();
  constexpr int VecSize = 16 / sizeof(T);
  const int bid = blockIdx.y, tid = threadIdx.x;
  using VecT = uint4;
  union {
    VecT v;
    T array[VecSize];
  } vec;
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x);
       i < ceilDiv(previous_len, VecSize);
       i += blockDim.x * gridDim.x) {
    vec.v = reinterpret_cast<const VecT*>(in_buffer)[i];
    if constexpr (Pass == 0) {
#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        if (i * VecSize + j < previous_len) {
          int bucket =
              calcBucket<T, BitsPerPass>(vec.array[j], start_bit, mask, false);
          atomicAdd(histogram_shm + bucket, vec.array[j]);
          atomicAdd(count_histogram_shm + bucket, 1);
        }
      }
    } else {
      int* filter_cnt = &counter->filterCnt;
      const auto kthValueBits = counter->kthValueBits;
      constexpr int previousStartBit = calcStartBit<T, BitsPerPass, Pass - 1>();
#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        const int idx = i * VecSize + j;
        if (idx < previous_len) {
          const auto previousBits =
              (twiddleIn(vec.array[j], false) >> previousStartBit)
              << previousStartBit;
          if (previousBits == kthValueBits) {
            if (early_stop) {
              const int pos = in_idx_buffer ? in_idx_buffer[idx] : idx;
              out_scores[bid] = vec.array[j];
              out_ids[bid] = pos;
            }
            if (out_buffer) {
              int pos = atomicAdd(filter_cnt, 1);
              out_buffer[pos] = vec.array[j];
              out_idx_buffer[pos] = in_idx_buffer ? in_idx_buffer[idx] : idx;
            }
            int bucket = calcBucket<T, BitsPerPass>(
                vec.array[j], start_bit, mask, false);
            atomicAdd(histogram_shm + bucket, vec.array[j]);
            atomicAdd(count_histogram_shm + bucket, 1);
          }
        }
      }
    }
  }
  __syncthreads();
  if (early_stop) {
    return;
  }
  // 合并多个block的结果
  for (int i = tid; i < NumBuckets; i += blockDim.x) {
    if (count_histogram_shm[i] > 0) {
      atomicAdd(histogram + i, histogram_shm[i]);
      atomicAdd(count_histogram + i, count_histogram_shm[i]);
    }
  }
}

#define BID 106
#define BATCH_ID 1

template <typename T, int BitsPerPass, int BlockSize, int NumBuckets, int Pass>
__global__ void air_topp_sampling(Counter<T>* counters,
                                  T* histograms,
                                  int* count_histograms,
                                  T* out,
                                  int64_t* ids,
                                  T* buf1,
                                  int* idx_buf1,
                                  T* buf2,
                                  int* idx_buf2,
                                  int* count_iter,
                                  int* count_iter_begin,
                                  const int buf_len) {
  /***
   * calc - filter - scan -find
   * TODO: calc - scan - find - filter
   ***/
  const int bid = blockIdx.y;
  if (count_iter_begin[bid] == count_iter[bid + 1]) {
    // topk
    return;
  }

  const int tid = threadIdx.x;
  auto counter = counters + bid;

  T current_sum;
  int previous_len, current_len;
  if constexpr (Pass == 0) {
    current_sum = 0;
    previous_len = counter->len;
    current_len = counter->len;
  } else {
    current_sum = counter->sum;
    previous_len = counter->previousLen;
    current_len = counter->len;
  }
  if (current_len == 0) {
    return;
  }
  const bool early_stop = (current_len == 1);
  const T* in_buf = nullptr;
  const int* in_idx_buf = nullptr;
  T* out_buf = nullptr;
  int* out_idx_buf = nullptr;
  const int buf_offset = bid * buf_len;
  if constexpr (Pass == 0) {
    in_buf = counter->in;
    in_idx_buf = nullptr;
    out_buf = nullptr;
    out_idx_buf = nullptr;
  } else if constexpr (Pass == 1) {
    in_buf = counter->in;
    in_idx_buf = nullptr;
    out_buf = buf1 + buf_offset;
    out_idx_buf = idx_buf1 + buf_offset;
  } else {
    in_buf = buf1 + buf_offset;
    in_idx_buf = idx_buf1 + buf_offset;
    out_buf = buf2 + buf_offset;
    out_idx_buf = idx_buf2 + buf_offset;
  }

  if (Pass == 0 || Pass == 1 || previous_len > buf_len) {
    // 没有写入buffer，滞后一个pass
    // 表示上一轮没有写入buf
    previous_len = counter->oriLen;
    in_buf = counter->in;
    in_idx_buf = nullptr;
  }
  if (Pass == 0 || current_len > buf_len) {
    // 当前pass无需写入buffer
    out_buf = nullptr;
    out_idx_buf = nullptr;
  }

#ifdef DEBUG_TOPP
  if (blockIdx.x == BID && bid == BATCH_ID && tid == 0) {
    printf("previous_len: %d, current_len: %d, buf_len: %d, NumBuckets: %d\n",
           previous_len,
           current_len,
           buf_len,
           NumBuckets);
  }
  __syncthreads();
#endif

  auto histogram = histograms + bid * NumBuckets;
  auto count_histogram = count_histograms + bid * NumBuckets;
  __shared__ T histogram_shm[NumBuckets];
  __shared__ int count_histogram_shm[NumBuckets];
  for (int i = tid; i < NumBuckets; i += blockDim.x) {
    histogram_shm[i] = 0;
    count_histogram_shm[i] = 0;
  }
  __syncthreads();

  filterAndHistogram<T, BitsPerPass, NumBuckets, Pass>(in_buf,
                                                       in_idx_buf,
                                                       out_buf,
                                                       out_idx_buf,
                                                       out,
                                                       ids,
                                                       previous_len,
                                                       counter,
                                                       histogram,
                                                       count_histogram,
                                                       histogram_shm,
                                                       count_histogram_shm,
                                                       early_stop);
  __syncthreads();
  // 保证全局内存操作对所有grid可见
  __threadfence();

  // #ifdef DEBUG_TOPP
  //   if (blockIdx.x == BID && bid == 0 && tid == 0) {
  //     for (int i = 0; i < NumBuckets; ++i) {
  //       printf("histogram[%d]: %f, count_histogram[%d]: %d\n", i,
  //       histogram[i], i, count_histogram[i]);
  //     }
  //   }
  //   __syncthreads();
  // #endif

  // find last block
  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    uint32_t finished = atomicInc(&counter->finishedBlockCnt, gridDim.x - 1);
    isLastBlock = (finished == (gridDim.x - 1));
  }

  if (__syncthreads_or(isLastBlock)) {
    if (early_stop) {
      if (threadIdx.x == 0) {
        counter->previousLen = 0;
        counter->len = 0;
      }
      return;
    }

    // scan/find
    // constexpr int WARP_SIZE = 32;
    constexpr int WARP_COUNT = NumBuckets / WARP_SIZE;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ T warpSum[WARP_COUNT];
    __shared__ cuda::atomic<T, cuda::thread_scope_block> blockSum;
    for (int i = tid; i < WARP_COUNT; i += BlockSize) {
      warpSum[i] = 0;
    }
    if (tid == 0) {
      blockSum = 0;
    }
    __syncthreads();
    // Acquire the summation of each 32 buckets
    for (int i = threadIdx.x; i < NumBuckets; i += BlockSize) {
      reduce_store_async(
          warp, warpSum + i / WARP_SIZE, histogram[i], cg::plus<float>{});
    }
    __syncthreads();
    // Acquire the summation of all the 2048 buckets
    if (threadIdx.x < WARP_SIZE) {
      reduce_store_async(
          warp, blockSum, warpSum[threadIdx.x], cg::plus<float>{});
      reduce_update_async(
          warp, blockSum, warpSum[threadIdx.x + WARP_SIZE], cg::plus<float>{});
    }
    __syncthreads();

    if constexpr (Pass == 0) {
      current_sum = blockSum * counter->p;
    }

    if (tid == 0) {
      T prev = 0;

      // Add 32 elements each step
      int iStep = 0;
      int targetStep = 0;
      for (; iStep < WARP_COUNT; iStep++) {
        if (warpSum[iStep]) {
          targetStep = iStep;
          if ((prev + warpSum[iStep]) >= current_sum) {
            break;
          }
          prev += warpSum[iStep];
        }
      }

      int targetIdx = 0;
      for (int i = targetStep * WARP_SIZE; i < NumBuckets; i++) {
        if (count_histogram[i]) {
          targetIdx = i;
          if ((prev + histogram[i]) >= current_sum) {
            break;
          }
          prev += histogram[i];
        }
      }
      counter->sum =
          current_sum - prev;  // how many values still are there to find
      counter->len = count_histogram[targetIdx];  // cur - prev; // number of
                                                  // values in next pass
      typename cub::Traits<T>::UnsignedBits bucket = targetIdx;
      int startBit = calcStartBit<T, BitsPerPass, Pass>();
      counter->kthValueBits |= bucket << startBit;
#ifdef DEBUG_TOPP
      if (bid == BATCH_ID && tid == 0) {
        printf("targetIdx: %d, count_histogram[%d]: %d, current_sum: %f\n",
               targetIdx,
               targetIdx,
               count_histogram[targetIdx],
               current_sum);
      }
#endif
    }
    __syncthreads();
    constexpr int numPasses = calcNumPasses<T, BitsPerPass>();
    if constexpr (Pass != numPasses - 1) {
      for (int i = tid; i < NumBuckets; i += BlockSize) {
        histogram[i] = 0;
        count_histogram[i] = 0;
      }
    }
    if (tid == 0) {
      // recover
      counter->previousLen = current_len;
      counter->filterCnt = 0;
    }
    if constexpr (Pass == numPasses - 1) {
      const auto kthValueBits = counter->kthValueBits;
      const auto equal_value = twiddleOut<T>(kthValueBits, false);

      const T* last_data =
          out_buf ? out_buf : in_buf;  // 最后一次Pass的输入数据
      const int* last_idx_data = out_idx_buf ? out_idx_buf : in_idx_buf;
      const int last_len =
          out_buf ? current_len : counter->oriLen;  // 最后一次Pass的token长度
#ifdef DEBUG_TOPP
      if (bid == BATCH_ID && tid == 0) {
        printf("equal_value: %f, last_len: %d\n", equal_value, last_len);
      }
      __syncthreads();
#endif
      for (int i = tid; i < last_len; i += BlockSize) {
        if (last_data[i] == equal_value) {
          out[bid] = equal_value;
          ids[bid] = last_idx_data ? last_idx_data[i] : i;
        }
      }
    }
  }
}

template <typename T, int BitsPerPass>
__global__ void air_topp_init(Counter<T>* counters,
                              T* histograms,
                              int* countHistograms,
                              const T* in,
                              const T* ps,
                              curandState_t* curandstate,
                              const int bsz,
                              const int vocab_size,
                              const int buf_len,
                              const int num_buckets) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  Counter<T>* counter_now = counters + bid;
  T* histogram_now = histograms + bid * num_buckets;
  int* count_histogram_now = countHistograms + bid * num_buckets;
  const int offset = bid * vocab_size;
  if (tid == 0) {
    counter_now->in = in + offset;

    counter_now->len = vocab_size;
    counter_now->oriLen = vocab_size;
    counter_now->previousLen = vocab_size;

    const T p = ps[bid];
    const T rand_p = curand_uniform(curandstate + bid) * p;
    counter_now->p = rand_p;

    counter_now->sum = 0;

    counter_now->kthValueBits = 0;
    counter_now->filterCnt = 0;
    counter_now->finishedBlockCnt = 0;
  }
  for (int i = tid; i < num_buckets; i += blockDim.x) {
    histogram_now[i] = 0;
    count_histogram_now[i] = 0;
  }
}
#endif

template <typename T>
struct DataTypeTraits {
  using DataType = T;
};

template <>
struct DataTypeTraits<phi::dtype::float16> {
  using DataType = half;
};

#ifdef CUDA_BFLOAT16_AVAILABLE
template <>
struct DataTypeTraits<phi::dtype::bfloat16> {
  using DataType = __nv_bfloat16;
};
#endif

#define FINAL_MASK 0xFFFFFFFF

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#ifdef PADDLE_WITH_HIP
#define WARP_SIZE 64
#define FIXED_BLOCK_DIM(...)                 \
  FIXED_BLOCK_DIM_BASE(1024, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(512, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);
#else
#define WARP_SIZE 32
#define FIXED_BLOCK_DIM(...)                 \
  FIXED_BLOCK_DIM_BASE(1024, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(512, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);   \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)
#endif

struct SegmentOffsetIter {
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  __host__ __device__ __forceinline__ int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int id) {
    this->v = value;
    this->id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (static_cast<float>(v) < static_cast<float>(value));
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return (static_cast<float>(v) > static_cast<float>(value));
  }
  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (static_cast<float>(v) < static_cast<float>(in.v)) ||
           ((static_cast<float>(v) == static_cast<float>(in.v)) &&
            (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (static_cast<float>(v) > static_cast<float>(in.v)) ||
           ((static_cast<float>(v) == static_cast<float>(in.v)) &&
            (id < in.id));
  }

  T v;
  int id;
};

int GetBlockSize(int vocab_size) {
  if (vocab_size > 512) {
    return 1024;
  } else if (vocab_size > 256) {
    return 512;
  } else if (vocab_size > 128) {
    return 256;
  } else if (vocab_size > 64) {
    return 128;
  } else {
    return 64;
  }
}

inline int div_up(int a, int n) { return (a + n - 1) / n; }

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[],
                                      const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(
    Pair<T> topk[], const T* src, int idx, int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[],
                                        const T* src,
                                        int idx,
                                        int dim,
                                        const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[],
                                              int* beam,
                                              int beam_size,
                                              const T* src,
                                              bool* firstStep,
                                              bool* is_empty,
                                              Pair<T>* max,
                                              int dim,
                                              const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(std::numeric_limits<T>::min(), -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(
            topk + MaxLength - *beam, src, tid, dim, *max, length);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).id == -1) *is_empty = true;
    *beam = 0;
  }
}

template <typename T>
__forceinline__ __device__ Pair<T> WarpReduce(Pair<T> input) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    T tmp_val = phi::backends::gpu::CudaShuffleDownSync(
        FINAL_MASK, input.v, offset, WARP_SIZE);
    int tmp_id = phi::backends::gpu::CudaShuffleDownSync(
        FINAL_MASK, input.id, offset, WARP_SIZE);
    if (static_cast<float>(input.v) < static_cast<float>(tmp_val)) {
      input.v = tmp_val;
      input.id = tmp_id;
    }
  }
  return input;
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T> shared_max[],
                                            Pair<T> topk[],
                                            Pair<T> beam_max[],
                                            int* beam,
                                            int* k,
                                            int* count,
                                            const int tid,
                                            const int wid,
                                            const int lane) {
  while (true) {
    __syncthreads();
    Pair<T> input_now = topk[0];
    input_now = WarpReduce(input_now);

    if (lane == 0) {
      shared_max[wid] = input_now;
    }
    __syncthreads();
    input_now = (tid < BlockSize / WARP_SIZE)
                    ? shared_max[lane]
                    : Pair<T>(std::numeric_limits<T>::min(), -1);
    if (wid == 0) {
      input_now = WarpReduce(input_now);
      if (lane == 0) shared_max[0] = input_now;
    }
    __syncthreads();
    if (tid == 0) {
      beam_max[*count] = shared_max[0];
      (*count)++;
    }
    int tid_max = shared_max[0].id % BlockSize;
    if (tid == tid_max) {
      (*beam)++;
    }
    if (--(*k) == 0) break;
    __syncthreads();

    if (tid == tid_max) {
      if (*beam < MaxLength) {
        topk[0] = topk[*beam];
      }
    }

    if (MaxLength < 5) {
      if (*beam >= MaxLength) break;
    } else {
#ifdef PADDLE_WITH_HIP
      uint64_t mask = 0u;
      mask = __ballot(true);
      if (tid_max / WARP_SIZE == wid) {
        if (__shfl_down(*beam, tid_max % WARP_SIZE, WARP_SIZE) == MaxLength)
          break;
      }
#else
      unsigned mask = 0u;
      mask = __ballot_sync(FINAL_MASK, true);
      if (tid_max / WARP_SIZE == wid) {
        if (__shfl_down_sync(
                FINAL_MASK, *beam, tid_max % WARP_SIZE, WARP_SIZE) == MaxLength)
          break;
      }
#endif
    }
  }
}

template <typename T>
__device__ inline T exponential_transform(T val, T lambda) {
#if defined(__NVCC__) || defined(__HIPCC__)
  T log = -std::numeric_limits<T>::epsilon() / 2;
  if (val < static_cast<T>(1.) - std::numeric_limits<T>::epsilon() / 2) {
    if (std::is_same<T, double>::value) {
      log = logf(val);
    } else {
      log = __logf(val);
    }
  }
  return static_cast<T>(-1.0) / lambda * log;
#else
  return static_cast<T>(-1.0) / lambda * std::log(static_cast<T>(1.0) - val);
#endif
}

template <typename T, int MaxLength, int TopPBeamTopK, int BlockSize>
__global__ void KeMatrixTopPBeamTopK(const T* src,
                                     const T* threshold,
                                     GPU(randState_t) * states,
                                     T* top_ps,
                                     int64_t* out_id,  // topk id
                                     T* out_val,       // topk val
                                     int64_t* topk_ids,
                                     T* topk_scores,
                                     int vocab_size,
                                     int* count_iter,
                                     int* count_iter_begin,
                                     const int k,
                                     const bool need_batch_random) {
  const int tid = threadIdx.x;
  const int wid = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int bid = blockIdx.x;
  const float threshold_now =
      threshold ? static_cast<float>(threshold[bid]) : 0.f;

  int top_num = TopPBeamTopK;
  float top_p_num = static_cast<float>(top_ps[bid]);
  const int offset = bid * vocab_size;
  int64_t* topk_ids_now = nullptr;
  T* topk_scores_now = nullptr;
  if (k > 0) {
    topk_ids_now = topk_ids + bid * k;
    topk_scores_now = topk_scores + bid * k;
  }

  __shared__ Pair<T> shared_max[BlockSize / WARP_SIZE];
  __shared__ Pair<T> beam_max[TopPBeamTopK];

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;
  __shared__ int count;

  if (tid == 0) {
    count = 0;
  }

  for (int j = 0; j < MaxLength; j++) {
    topk[j].set(std::numeric_limits<T>::min(), -1);
  }

  while (top_num) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk,
                                           &beam,
                                           TopPBeamTopK,
                                           src + offset,
                                           &firststep,
                                           &is_empty,
                                           &max,
                                           vocab_size,
                                           tid);
    BlockReduce<T, MaxLength, BlockSize>(
        shared_max, topk, beam_max, &beam, &top_num, &count, tid, wid, lane);
  }
  if (tid == 0) {
    count_iter_begin[bid] = count_iter[bid];
    float top_p = top_ps[bid];
    float sum_prob = 0.0f;
    bool flag = false;
    float max_val = 0.f;
    int max_id = -1;
    for (int i = 0; i < TopPBeamTopK; i++) {
      if (i < k) {
        topk_ids_now[i] = static_cast<int64_t>(beam_max[i].id);
        topk_scores_now[i] = beam_max[i].v;
      }
      if (!flag) {
        float val = static_cast<float>(beam_max[i].v);
        sum_prob += val;
        float random_ratio =
            exponential_transform(GPU(rand_uniform)(states + bid), 1.0f);
        float random_val = (val >= threshold_now ? val : 0.f) / random_ratio;
        if (max_val < random_val) {
          max_val = random_val;
          max_id = i;
        }
        if (sum_prob >= top_p) {
          flag = true;
          count_iter_begin[bid] += 1;
          if (max_id == -1) {
            // don't sample low score token
            out_id[bid] = static_cast<int64_t>(beam_max[0].id);
            out_val[bid] = beam_max[0].v;
          } else {
            out_id[bid] = static_cast<int64_t>(beam_max[max_id].id);
            out_val[bid] = beam_max[max_id].v;
          }
        }
      }
      if (flag && i >= k - 1) {
        break;
      }
    }
  }
}

template <typename T, int MaxLength, int TopPBeamTopK, int BlockSize>
__global__ void KeMatrixTopPBeamTopKFt(const T* src,
                                       const T* threshold,
                                       GPU(randState_t) * states,
                                       T* top_ps,
                                       int64_t* out_id,  // topk id
                                       T* out_val,       // topk val
                                       int64_t* topk_ids,
                                       T* topk_scores,
                                       int vocab_size,
                                       int* count_iter,
                                       int* count_iter_begin,
                                       const int k,
                                       const bool need_batch_random) {
  const int tid = threadIdx.x;
  const int wid = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int bid = blockIdx.x;
  const float threshold_now =
      threshold ? static_cast<float>(threshold[bid]) : 0.f;

  int top_num = TopPBeamTopK;
  float top_p_num = static_cast<float>(top_ps[bid]);
  int64_t* topk_ids_now = nullptr;
  T* topk_scores_now = nullptr;
  if (k > 0) {
    topk_ids_now = topk_ids + bid * k;
    topk_scores_now = topk_scores + bid * k;
  }

  __shared__ Pair<T> shared_max[BlockSize / WARP_SIZE];
  __shared__ Pair<T> beam_max[TopPBeamTopK];

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;
  __shared__ int count;

  if (tid == 0) {
    count = 0;
  }

  for (int j = 0; j < MaxLength; j++) {
    topk[j].set(std::numeric_limits<T>::min(), -1);
  }

  while (top_num) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk,
                                           &beam,
                                           TopPBeamTopK,
                                           src + bid * vocab_size,
                                           &firststep,
                                           &is_empty,
                                           &max,
                                           vocab_size,
                                           tid);
    BlockReduce<T, MaxLength, BlockSize>(
        shared_max, topk, beam_max, &beam, &top_num, &count, tid, wid, lane);
  }
  if (tid == 0) {
    count_iter_begin[bid] = count_iter[bid];
    float rand_top_p = GPU(rand_uniform)(states + bid) * top_p_num;
    top_ps[bid] = (T)rand_top_p;
    float sum_prob = 0.0f;
    bool flag = false;
    for (int i = 0; i < TopPBeamTopK; i++) {
      if (i < k) {
        topk_ids_now[i] = static_cast<int64_t>(beam_max[i].id);
        topk_scores_now[i] = beam_max[i].v;
      }
      if (!flag) {
        float val = static_cast<float>(beam_max[i].v);
        sum_prob += val;
#ifdef DEBUG_TOPP
        printf("bi: %d, top_p: %f, rand_top_p: %f, sum_prob: %f\n",
               bid,
               top_p_num rand_top_p,
               sum_prob);
#endif
        if (sum_prob >= rand_top_p) {
          flag = true;
          count_iter_begin[bid] += 1;
          if (val < threshold_now) {
            // don't sample low score token
            int start_id = i == 0 ? 0 : i - 1;
            for (int j = start_id; j >= 0; j--) {
              float val_now = static_cast<float>(beam_max[j].v);
              if (val_now >= threshold_now || j == 0) {
                out_id[bid] = static_cast<int64_t>(beam_max[j].id);
                out_val[bid] = beam_max[j].v;
                break;
              }
            }
          } else {
            out_id[bid] = static_cast<int64_t>(beam_max[i].id);
            out_val[bid] = beam_max[i].v;
          }
        }
      }
      if (flag && i >= k - 1) {
        break;
      }
    }
  }
}

__global__ void SetCountIter(int* count_iter, int num) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  for (int i = idx; i < num; i += gridDim.x * blockDim.x) {
    count_iter[i] = i;
  }
}

template <typename T>
__global__ void FillIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (T j = row_id; j < num_rows; j += gridDim.x) {
    for (T i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

template <typename T, typename Context, int TopKMaxLength, int TopPBeamTopK>
void DispatchKeMatrixTopPBeamTopK(const Context& dev_ctx,
                                  const T* src,
                                  const T* threshold,
                                  GPU(randState_t) * states,
                                  T* top_ps,
                                  int64_t* out_id,  // topk id
                                  T* out_val,       // topk val
                                  int64_t* topk_ids,
                                  T* topk_scores,
                                  int vocab_size,
                                  int* count_iter,
                                  int* count_iter_begin,
                                  const int k,
                                  const int bs,
                                  const bool need_batch_random,
                                  const std::string& mode) {
  int BlockSize = GetBlockSize(vocab_size);
  if (mode == "truncated") {
    switch (BlockSize) {
      FIXED_BLOCK_DIM(
          KeMatrixTopPBeamTopKFt<T, TopKMaxLength, TopPBeamTopK, kBlockDim>
          <<<bs, kBlockDim, 0, dev_ctx.stream()>>>(src,
                                                   threshold,
                                                   states,
                                                   top_ps,
                                                   out_id,
                                                   out_val,
                                                   topk_ids,
                                                   topk_scores,
                                                   vocab_size,
                                                   count_iter,
                                                   count_iter_begin,
                                                   k,
                                                   need_batch_random));
      default:
        PD_THROW(
            "the input data shape has error in the topp_beam_topk kernel.");
    }
  } else {
    switch (BlockSize) {
      FIXED_BLOCK_DIM(
          KeMatrixTopPBeamTopK<T, TopKMaxLength, TopPBeamTopK, kBlockDim>
          <<<bs, kBlockDim, 0, dev_ctx.stream()>>>(src,
                                                   threshold,
                                                   states,
                                                   top_ps,
                                                   out_id,
                                                   out_val,
                                                   topk_ids,
                                                   topk_scores,
                                                   vocab_size,
                                                   count_iter,
                                                   count_iter_begin,
                                                   k,
                                                   need_batch_random));
      default:
        PD_THROW(
            "the input data shape has error in the topp_beam_topk kernel.");
    }
  }
}

struct BlockPrefixCallbackOp {
  // Running prefix
  float running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(float running_total)
      : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ float operator()(float block_aggregate) {
    float old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename T>
__device__ T max_func(const T a, const T b) {
  return a > b ? a : b;
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max_func(a, b);
  }
};

template <typename T, int BLOCK_SIZE>
__global__ void topp_sampling(T* sorted_probs,
                              int64_t* sorted_id,
                              T* out_val,
                              int64_t* out_id,
                              const T* top_ps,
                              const T* threshold,
                              GPU(randState_t) * states,
                              const int p_num,
                              const int vocab_size,
                              const bool need_batch_random,
                              int* count_iter,
                              int* count_iter_begin) {
  __shared__ int stop_shared;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;
  const float p_t = static_cast<float>(top_ps[bid]);
  const float threshold_now =
      threshold ? static_cast<float>(threshold[bid]) : 0.f;
  if (tid == 0) {
    stop_shared = 0;
  }
  if (count_iter_begin[bid] == count_iter[bid + 1]) {
    // topk
    return;
  }

  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
  typedef cub::BlockReduce<Pair<T>, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  int offset = bid * vocab_size;
#ifdef DEBUG_TOPP
  if (tid == 0) {
    printf(
        "first_elem1_1: %f, first_elem1_2: %f, first_id1_1: %d, first_id1_2: "
        "%d\n",
        static_cast<float>(sorted_probs[offset]),
        static_cast<float>(sorted_probs[offset + 1]),
        static_cast<int>(sorted_id[offset]),
        static_cast<int>(sorted_id[offset + 1]));
  }
#endif
  int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int i_activate = 0;
  float thread_offset = 0;
  Pair<T> max_thread_pair(static_cast<T>(0.), -1);
  for (int i = tid; i < end; i += BLOCK_SIZE) {
    float thread_count =
        (i < vocab_size) ? static_cast<float>(sorted_probs[offset + i]) : 0.f;
    BlockScan(temp_storage)
        .InclusiveSum(thread_count, thread_offset, prefix_op);

    if (thread_offset < p_t ||
        (thread_offset >= p_t && thread_offset - thread_count < p_t)) {
      float random_ratio =
          exponential_transform(GPU(rand_uniform)(states + bid), 1.0f);
      float tmp_val =
          (thread_count >= threshold_now ? thread_count : 0.f) / random_ratio;
      if (static_cast<float>(max_thread_pair.v) < tmp_val) {
        max_thread_pair.set(static_cast<T>(tmp_val), i);
      }
#ifdef DEBUG_TOPP
      if (i < 10) {
        printf(
            "tid: %d, i: %d, random_ratio: %f, thread_count: %f, tmp_val: %f, "
            "max_thread_pair.v: %f, max_thread_pair.id: %d\n",
            tid,
            i,
            random_ratio,
            thread_count,
            tmp_val,
            max_thread_pair.v,
            static_cast<int>(max_thread_pair.id));
      }
#endif
    }
#ifdef DEBUG_TOPP
    printf("tid: %d, thread_count: %f, thread_offset: %f\n",
           tid,
           thread_count,
           thread_offset);
#endif
#ifdef PADDLE_WITH_HIP
    uint64_t activate_mask = __ballot(p_t <= thread_offset);
#else
    uint32_t activate_mask = __ballot_sync(FINAL_MASK, p_t <= thread_offset);
#endif

    i_activate = i;
    if (activate_mask != 0) {
      if (lane_id == 0) {
        atomicAdd(&stop_shared, 1);
      }
    }
    __syncthreads();
    if (stop_shared > 0) {
      break;
    }
  }
  __syncthreads();
  if (stop_shared == 0) {
    if (tid == 0) {
      out_id[bid] = sorted_id[offset];
      out_val[bid] = sorted_probs[offset];
    }
    return;
  }

  Pair<T> max_pair = BlockReduce(temp_storage_reduce)
                         .Reduce(max_thread_pair, MaxOp<Pair<T>>());
  if (tid == 0) {
    if (max_pair.id == -1) {
      max_pair.id = 0;
    }
#ifdef DEBUG_TOPP
    printf("max_id: %d, max_val: %f\n",
           static_cast<int>(max_pair.id),
           static_cast<float>(max_pair.v));
#endif
    out_id[bid] = sorted_id[offset + max_pair.id];
    out_val[bid] = sorted_probs[offset + max_pair.id];
  }
}

template <typename T, int BLOCK_SIZE>
__global__ void topp_sampling_ft(T* sorted_probs,
                                 int64_t* sorted_id,
                                 T* out_val,
                                 int64_t* out_id,
                                 const T* top_ps,
                                 const T* threshold,
                                 GPU(randState_t) * states,
                                 const int p_num,
                                 const int vocab_size,
                                 const bool need_batch_random,
                                 int* count_iter,
                                 int* count_iter_begin) {
  __shared__ int stop_shared;
  __shared__ float rand_p;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;
  const float p_t = static_cast<float>(top_ps[bid]);
  const float threshold_now =
      threshold ? static_cast<float>(threshold[bid]) : 0.f;
  if (tid == 0) {
    stop_shared = 0;
    rand_p = p_t;
#ifdef DEBUG_TOPP
    printf("bi: %d, p: %f\n", bid, rand_p);
#endif
  }
  if (count_iter_begin[bid] == count_iter[bid + 1]) {
    // topk
    return;
  }

  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ typename BlockReduce::TempStorage temp_storage_reduce;
#ifdef PADDLE_WITH_HIP
  __shared__ uint64_t selected_shared[NUM_WARPS];
#else
  __shared__ uint32_t selected_shared[NUM_WARPS];
#endif
  int threshold_id = 0;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  if (lane_id == 0) {
    selected_shared[warp_id] = 0;
  }
  __syncthreads();

  int offset = bid * vocab_size;
#ifdef DEBUG_TOPP
  if (tid == 0) {
    printf(
        "first_elem1_1: %f, first_elem1_2: %f, first_id1_1: %d, first_id1_2: "
        "%d\n",
        static_cast<float>(sorted_probs[offset]),
        static_cast<float>(sorted_probs[offset + 1]),
        static_cast<int>(sorted_id[offset]),
        static_cast<int>(sorted_id[offset + 1]));
  }
#endif
  int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int i_activate = 0;
  float thread_offset = 0;
  for (int i = tid; i < end; i += BLOCK_SIZE) {
    float thread_count =
        (i < vocab_size) ? static_cast<float>(sorted_probs[offset + i]) : 0.f;
    if (i < vocab_size && thread_count >= threshold_now) {
      threshold_id = i;
    }
    BlockScan(temp_storage)
        .InclusiveSum(thread_count, thread_offset, prefix_op);

#ifdef PADDLE_WITH_HIP
    uint64_t activate_mask = __ballot(rand_p <= thread_offset);
#else
    uint32_t activate_mask = __ballot_sync(FINAL_MASK, rand_p <= thread_offset);
#endif

    i_activate = i;
    if (activate_mask != 0) {
      if (lane_id == 0) {
        atomicAdd(&stop_shared, 1);
        selected_shared[warp_id] = activate_mask;
      }
    }
    __syncthreads();
    if (stop_shared > 0) {
      break;
    }
  }
  __syncthreads();
  if (stop_shared == 0) {
    if (tid == 0) {
      out_id[bid] = sorted_id[offset];
      out_val[bid] = sorted_probs[offset];
#ifdef DEBUG_TOPP
      printf("stop_shared: %d, out_id: %d, out_val: %f\n",
             static_cast<int>(stop_shared),
             static_cast<int>(out_id[bid]),
             static_cast<float>(out_val[bid]));
#endif
    }
    return;
  }
#ifdef DEBUG_TOPP
  if (tid == 0) {
    printf(
        "first_elem2_1: %f, first_elem2_2: %f, first_id2_1: %d, first_id2_2: "
        "%d\n",
        static_cast<float>(sorted_probs[offset]),
        static_cast<float>(sorted_probs[offset + 1]),
        static_cast<int>(sorted_id[offset]),
        static_cast<int>(sorted_id[offset + 1]));
  }
#endif
  bool skip = (selected_shared[warp_id] > 0) ? false : true;
  for (int i = 0; i < warp_id; i++) {
    if (selected_shared[i] != 0) {
      // If the previous has stopped, skip the current warp
      skip = true;
    }
  }
  if (!skip) {
#ifdef PADDLE_WITH_HIP
    int active_lane_id =
        WARP_SIZE - __popcll(selected_shared[warp_id]);  // first not 0
#else
    int active_lane_id =
        WARP_SIZE - __popc(selected_shared[warp_id]);  // first not 0
#endif
    if (lane_id == active_lane_id) {
      float val = static_cast<float>(sorted_probs[offset + i_activate]);
#ifdef DEBUG_TOPP
      printf(
          "active_lane_id: %d, i_activate: %d.\n", active_lane_id, i_activate);
      for (int i = 0; i < active_lane_id; i++) {
        printf("p %d, value: %f\n",
               i,
               static_cast<float>(sorted_probs[offset + i]));
      }
#endif
      if (val < threshold_now) {
        // don't sample low score token
        int max_id =
            BlockReduce(temp_storage_reduce).Reduce(threshold_id, MaxOp<int>());
#ifdef PADDLE_WITH_HIP
        hiprandStatePhilox4_32_10_t rng;
        hiprand_init(bid * blockDim.x + tid, tid, 0, &rng);
        int random_id = hiprand(&rng) % (max_id + 1);
#else
        curandStatePhilox4_32_10_t rng;
        curand_init(bid * blockDim.x + tid, tid, 0, &rng);
        int random_id = curand(&rng) % (max_id + 1);
#endif
        out_id[bid] = sorted_id[offset + random_id];
        out_val[bid] = sorted_probs[offset + random_id];
      } else {
        out_id[bid] = sorted_id[offset + i_activate];
        out_val[bid] = sorted_probs[offset + i_activate];
      }
    }
  }
}

template <typename T, typename Context>
void DispatchTopPSampling(const Context& dev_ctx,
                          T* sorted_probs,
                          int64_t* sorted_id,
                          T* out_val,
                          int64_t* out_id,
                          const T* top_ps,
                          const T* threshold,
                          GPU(randState_t) * states,
                          const int p_num,
                          const int vocab_size,
                          const int bs,
                          const bool need_batch_random,
                          int* count_iter,
                          int* count_iter_begin,
                          const std::string& mode) {
  int BlockSize = GetBlockSize(vocab_size);
  if (mode == "truncated") {
    switch (BlockSize) {
      FIXED_BLOCK_DIM(
          topp_sampling_ft<T, kBlockDim>
          <<<bs, kBlockDim, 0, dev_ctx.stream()>>>(sorted_probs,
                                                   sorted_id,
                                                   out_val,
                                                   out_id,
                                                   top_ps,
                                                   threshold,
                                                   states,
                                                   p_num,
                                                   vocab_size,
                                                   need_batch_random,
                                                   count_iter,
                                                   count_iter_begin));
      default:
        PD_THROW("the input data shape has error in the topp_sampling kernel.");
    }
  } else {
    switch (BlockSize) {
      FIXED_BLOCK_DIM(
          topp_sampling<T, kBlockDim>
          <<<bs, kBlockDim, 0, dev_ctx.stream()>>>(sorted_probs,
                                                   sorted_id,
                                                   out_val,
                                                   out_id,
                                                   top_ps,
                                                   threshold,
                                                   states,
                                                   p_num,
                                                   vocab_size,
                                                   need_batch_random,
                                                   count_iter,
                                                   count_iter_begin));
      default:
        PD_THROW("the input data shape has error in the topp_sampling kernel.");
    }
  }
}

__global__ void setup_kernel(GPU(randState_t) * state,
                             int64_t* seed,
                             const int bs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < bs; i += gridDim.x * blockDim.x) {
    GPU(rand_init)(static_cast<uint64_t>(seed[i]), 0, 0, &state[i]);
  }
}

__global__ void setup_kernel(GPU(randState_t) * state,
                             const uint64_t seed,
                             const uint64_t offset,
                             const int bs,
                             const bool need_batch_random) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < bs; i += gridDim.x * blockDim.x) {
    if (need_batch_random) {
      GPU(rand_init)(seed, i, offset, &state[i]);
    } else {
      GPU(rand_init)(seed, 0, offset, &state[i]);
    }
  }
}

template <typename T>
T* SafeGetTensorPtr(const DenseTensor& t) {
  return const_cast<T*>(t.data<T>());
}

template <typename T>
T* SafeGetTensorPtr(const DenseTensor* t) {
  return t ? SafeGetTensorPtr<T>(*t) : nullptr;
}

template <typename T>
T* SafeGetTensorPtr(const paddle::optional<DenseTensor>& t) {
  return t ? SafeGetTensorPtr<T>(t.get()) : nullptr;
}

template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& ps,
                        const paddle::optional<DenseTensor>& threshold,
                        const paddle::optional<DenseTensor>& topp_seed,
                        int seed,
                        int k,
                        const std::string& mode,
                        DenseTensor* out,
                        DenseTensor* ids,
                        DenseTensor* topk_scores,
                        DenseTensor* topk_ids) {
  typedef DataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  auto cu_stream = dev_ctx.stream();
  const auto* input = &x;
  // get the input dims
  const auto& in_dims = input->dims();
  int p_num = ps.numel();
  int bs = in_dims[0];
  int vocab_size = in_dims[1];
  T* out_ptr = dev_ctx.template Alloc<T>(out);
  int64_t* ids_ptr = dev_ctx.template Alloc<int64_t>(ids);
  T* topk_scores_data = nullptr;
  int64_t* topk_ids_data = nullptr;
  if (k > 0) {
    topk_scores_data = dev_ctx.template Alloc<T>(topk_scores);
    topk_ids_data = dev_ctx.template Alloc<int64_t>(topk_ids);
  }

  DenseTensor ps_now;
  ps_now.Resize(phi::make_ddim({bs, 1}));
  dev_ctx.template Alloc<T>(&ps_now);
  phi::Copy(dev_ctx, ps, dev_ctx.GetPlace(), false, &ps_now);

  DenseTensor inds_input;
  inds_input.Resize(phi::make_ddim({bs, vocab_size}));
  dev_ctx.template Alloc<int64_t>(&inds_input);

  DenseTensor sorted_out;
  sorted_out.Resize(phi::make_ddim({bs, vocab_size}));
  dev_ctx.template Alloc<T>(&sorted_out);

  DenseTensor sorted_id;
  sorted_id.Resize(phi::make_ddim({bs, vocab_size}));
  dev_ctx.template Alloc<int64_t>(&sorted_id);

  int BlockSize = GetBlockSize(vocab_size);

  switch (BlockSize) {
    FIXED_BLOCK_DIM(FillIndex<int64_t><<<bs, kBlockDim, 0, cu_stream>>>(
        inds_input.data<int64_t>(), bs, vocab_size));
    default:
      PD_THROW("the input data shape has error in the FillIndex kernel.");
  }
  int64_t* infer_seed = SafeGetTensorPtr<int64_t>(topp_seed);

  GPU(randState_t) * states{nullptr};
  phi::Allocator::AllocationPtr rand_states_buf{nullptr};
  rand_states_buf = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      bs * sizeof(GPU(randState_t)),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  states = reinterpret_cast<GPU(randState_t)*>(rand_states_buf->ptr());

  uint64_t seed_now = seed;
  uint64_t offset = 0;
  bool need_batch_random = false;

  if (infer_seed) {
    setup_kernel<<<1, 256, 0, cu_stream>>>(states, infer_seed, bs);
  } else {
    if (seed_now == -1) {
      need_batch_random = true;
      auto gen_cuda = dev_ctx.GetGenerator();
      uint64_t increment = ps.numel() * 4;
      auto seed_offset = gen_cuda->IncrementOffset(increment);
      seed_now = seed_offset.first;
      offset = seed_offset.second;
      setup_kernel<<<1, 256, 0, cu_stream>>>(
          states, seed_now, offset, bs, need_batch_random);
    } else {
      setup_kernel<<<1, 256, 0, cu_stream>>>(
          states, seed_now, offset, bs, need_batch_random);
    }
  }

  DenseTensor count_iter;
  count_iter.Resize(phi::make_ddim({bs + 1}));
  dev_ctx.template Alloc<int>(&count_iter);
  DenseTensor count_iter_begin;
  count_iter_begin.Resize(phi::make_ddim({bs}));
  dev_ctx.template Alloc<int>(&count_iter_begin);
  SetCountIter<<<1, 256, 0, cu_stream>>>(count_iter.data<int>(), bs + 1);

  T* threshold_data = SafeGetTensorPtr<T>(threshold);

  constexpr int TopKMaxLength = 2;
  constexpr int TopPBeamTopK = 20;

  DispatchKeMatrixTopPBeamTopK<T, Context, TopKMaxLength, TopPBeamTopK>(
      dev_ctx,
      x.data<T>(),
      threshold_data,
      states,
      ps_now.data<T>(),
      ids_ptr,
      out_ptr,
      topk_ids_data,
      topk_scores_data,
      vocab_size,
      count_iter.data<int>(),
      count_iter_begin.data<int>(),
      k,
      bs,
      need_batch_random,
      mode);
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 12000
  if (FLAGS_use_air_topp) {
    static_assert(std::is_same<T, float>::value,
                  "air_topp only supports float now!");
    constexpr int BitsPerPass = 11;
    constexpr int SAMPLING_BLOCK_SIZE = 512;
    constexpr int INIT_BLOCK_SIZE = 1024;
    phi::Allocator::AllocationPtr counter_ptr{nullptr};
    counter_ptr = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        bs * sizeof(Counter<T>),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    Counter<T>* counters = reinterpret_cast<Counter<T>*>(counter_ptr->ptr());
    constexpr int numBuckets = calcNumBuckets<BitsPerPass>();
    const int buf_len = calcBufLen<T>(vocab_size);
    DenseTensor histograms, countHistograms, buf1, id_buf1, buf2, id_buf2;
    histograms.Resize(phi::make_ddim({bs, numBuckets}));
    countHistograms.Resize(phi::make_ddim({bs, numBuckets}));
    buf1.Resize(phi::make_ddim({bs, buf_len}));
    id_buf1.Resize(phi::make_ddim({bs, buf_len}));
    buf2.Resize(phi::make_ddim({bs, buf_len}));
    id_buf2.Resize(phi::make_ddim({bs, buf_len}));
    dev_ctx.template Alloc<T>(&histograms);
    dev_ctx.template Alloc<int>(&countHistograms);
    dev_ctx.template Alloc<T>(&buf1);
    dev_ctx.template Alloc<int>(&id_buf1);
    dev_ctx.template Alloc<T>(&buf2);
    dev_ctx.template Alloc<int>(&id_buf2);

    air_topp_init<T, BitsPerPass><<<bs, INIT_BLOCK_SIZE, 0, dev_ctx.stream()>>>(
        counters,
        histograms.data<T>(),
        countHistograms.data<int>(),
        x.data<T>(),
        ps.data<T>(),
        states,
        bs,
        vocab_size,
        buf_len,
        numBuckets);

    constexpr int VecSize = 16 / sizeof(T);

    const int max_block_num_vocab =
        ceilDiv(vocab_size, SAMPLING_BLOCK_SIZE * VecSize);
    auto kernel =
        air_topp_sampling<T, BitsPerPass, SAMPLING_BLOCK_SIZE, numBuckets, 0>;
    const int dev_id = 0;
    int sm_count;
    int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, kernel, SAMPLING_BLOCK_SIZE, 0);
    assert(act_blocks_per_sm > 1);
    const int block_per_wave = sm_count * act_blocks_per_sm;
    const int block_num_vocab =
        std::min(max_block_num_vocab, block_per_wave * 4 / bs);  // !!!
    VLOG(1) << "block_per_wave: " << block_per_wave
            << ", block_num_vocab: " << block_num_vocab;
    dim3 grid(block_num_vocab, bs);
    constexpr int numPasses = calcNumPasses<T, BitsPerPass>();
    for (int pass = 0; pass < numPasses; ++pass) {
      VLOG(1) << "pass: " << pass;
      if (pass == 0) {
        air_topp_sampling<T, BitsPerPass, SAMPLING_BLOCK_SIZE, numBuckets, 0>
            <<<grid, SAMPLING_BLOCK_SIZE, 0, dev_ctx.stream()>>>(
                counters,
                histograms.data<T>(),
                countHistograms.data<int>(),
                out_ptr,
                ids_ptr,
                buf1.data<T>(),
                id_buf1.data<int>(),
                buf2.data<T>(),
                id_buf2.data<int>(),
                count_iter.data<int>(),
                count_iter_begin.data<int>(),
                buf_len);
      } else if (pass == 1) {
        air_topp_sampling<T, BitsPerPass, SAMPLING_BLOCK_SIZE, numBuckets, 1>
            <<<grid, SAMPLING_BLOCK_SIZE, 0, dev_ctx.stream()>>>(
                counters,
                histograms.data<T>(),
                countHistograms.data<int>(),
                out_ptr,
                ids_ptr,
                buf1.data<T>(),
                id_buf1.data<int>(),
                buf2.data<T>(),
                id_buf2.data<int>(),
                count_iter.data<int>(),
                count_iter_begin.data<int>(),
                buf_len);
      } else if (pass == 2) {
        air_topp_sampling<T, BitsPerPass, SAMPLING_BLOCK_SIZE, numBuckets, 2>
            <<<grid, SAMPLING_BLOCK_SIZE, 0, dev_ctx.stream()>>>(
                counters,
                histograms.data<T>(),
                countHistograms.data<int>(),
                out_ptr,
                ids_ptr,
                buf1.data<T>(),
                id_buf1.data<int>(),
                buf2.data<T>(),
                id_buf2.data<int>(),
                count_iter.data<int>(),
                count_iter_begin.data<int>(),
                buf_len);
      } else {
        PD_THROW("pass must be 0,1 or 2!");
      }
    }
  } else {
#endif
    size_t temp_storage_bytes = 0;

    cub::TransformInputIterator<int, SegmentOffsetIter, int*>
        segment_offsets_t_begin(count_iter_begin.data<int>(),
                                SegmentOffsetIter(vocab_size));

    cub::TransformInputIterator<int, SegmentOffsetIter, int*>
        segment_offsets_t_end(count_iter.data<int>(),
                              SegmentOffsetIter(vocab_size));

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        temp_storage_bytes,
        reinterpret_cast<DataType_*>(const_cast<T*>(x.data<T>())),
        reinterpret_cast<DataType_*>(const_cast<T*>(sorted_out.data<T>())),
        inds_input.data<int64_t>(),
        sorted_id.data<int64_t>(),
        vocab_size * bs,
        bs,
        segment_offsets_t_begin,
        segment_offsets_t_end + 1,
        0,
        sizeof(T) * 8,
        cu_stream);

    temp_storage_bytes = div_up(temp_storage_bytes, 256) * 256;
    int64_t temp_size = temp_storage_bytes;
    DenseTensor temp_storage;
    temp_storage.Resize(phi::make_ddim({temp_size}));
    dev_ctx.template Alloc<uint8_t>(&temp_storage);

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(),
        temp_storage_bytes,
        reinterpret_cast<DataType_*>(const_cast<T*>(x.data<T>())),
        reinterpret_cast<DataType_*>(const_cast<T*>(sorted_out.data<T>())),
        inds_input.data<int64_t>(),
        sorted_id.data<int64_t>(),
        vocab_size * bs,
        bs,
        segment_offsets_t_begin,
        segment_offsets_t_end + 1,
        0,
        sizeof(T) * 8,
        cu_stream);

    DispatchTopPSampling<T>(dev_ctx,
                            sorted_out.data<T>(),
                            sorted_id.data<int64_t>(),
                            out_ptr,
                            ids_ptr,
                            ps_now.data<T>(),
                            threshold_data,
                            states,
                            p_num,
                            vocab_size,
                            bs,
                            need_batch_random,
                            count_iter.data<int>(),
                            count_iter_begin.data<int>(),
                            mode);
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 12000
  }
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(
    top_p_sampling, GPU, ALL_LAYOUT, phi::TopPSamplingKernel, float) {}

// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "paddle/phi/backends/gpu/gpu_info.h"

#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_helper.h"

#pragma GCC diagnostic pop

namespace phi {

namespace fusion {

template <typename T, typename Context>
void MoeDispatchKernel(const Context& ctx,
                       const DenseTensor& X,
                       const DenseTensor& gating_output,
                       const int moe_topk,
                       const bool group_moe,
                       DenseTensor* permute_input,
                       DenseTensor* token_nums_per_expert,
                       DenseTensor* permute_indices_per_token,
                       DenseTensor* expert_scales_float,
                       DenseTensor* top_k_indices,
                       DenseTensor* group_max_prob) {
  PADDLE_ENFORCE_EQ(
      X.dims().size(),
      2,
      common::errors::InvalidArgument("the input X should be a 2D tensor"));
  const int num_rows = X.dims()[0];
  const int hidden_size = X.dims()[1];
  const int expert_num = gating_output.dims()[1];
  if (group_moe == true) {
    // Check if expert_num is divisible by moe_topk, else throw an error
    PADDLE_ENFORCE_EQ(expert_num % moe_topk,
                      0,
                      common::errors::InvalidArgument(
                          "The number of experts (expert_num) "
                          "must be divisible by moe_topk. "
                          "Got expert_num = %d and moe_topk = %d.",
                          expert_num,
                          moe_topk));
  }

  // correspond to the weighted coefficients of the results from each expert.
  expert_scales_float->Resize({num_rows, moe_topk});

  DenseTensor finished_tensor = Empty<bool>(ctx, {num_rows});
  bool* finished = finished_tensor.data<bool>();
  // set false
  funcs::SetConstant<GPUContext, bool> zero;
  zero(ctx, &finished_tensor, false);

  const int num_moe_inputs = AlignTo16(num_rows * moe_topk);
  const int bytes = num_moe_inputs * sizeof(int);
  DenseTensor ws_ptr_tensor = Empty<int8_t>(ctx, {bytes});
  int8_t* ws_ptr = ws_ptr_tensor.data<int8_t>();
  int* source_rows_ = reinterpret_cast<int*>(ws_ptr);

  top_k_indices->Resize({num_rows, moe_topk});
  int* expert_for_source_row = ctx.template Alloc<int>(top_k_indices);

  group_max_prob->Resize({num_rows, moe_topk});
  float* group_max_out = ctx.template Alloc<float>(group_max_prob);

  DenseTensor softmax_buffer = Empty<float>(ctx, {num_rows * expert_num});
  float* softmax_out_ = softmax_buffer.data<float>();

  VLOG(4) << "num_rows: " << num_rows << ", expert_num: " << expert_num
          << ", moe_topk: " << moe_topk << ", group_moe: " << group_moe;

  topk_gating_softmax_kernelLauncher<float>(
      gating_output.data<float>(),
      finished,
      ctx.template Alloc<float>(expert_scales_float),
      softmax_out_,
      group_max_out,
      expert_for_source_row,
      source_rows_,
      num_rows,
      expert_num,
      moe_topk,
      group_moe,
      ctx.stream());

  CubKeyValueSorter sorter_;

  const int sorter_ws_size_bytes =
      AlignTo16(sorter_.getWorkspaceSize(moe_topk * num_rows));
  DenseTensor sorter_ws = Empty<int8_t>(ctx, {sorter_ws_size_bytes});
  int8_t* sorter_ws_ptr = sorter_ws.data<int8_t>();

  DenseTensor permutation_buffer = Empty<int32_t>(ctx, {num_moe_inputs * 2});
  int* permuted_experts_ = permutation_buffer.data<int32_t>();
  int* permuted_rows_ = permuted_experts_ + num_moe_inputs;

  sorter_.run(reinterpret_cast<void*>(sorter_ws_ptr),
              sorter_ws_size_bytes,
              expert_for_source_row,
              permuted_experts_,
              source_rows_,
              permuted_rows_,
              moe_topk * num_rows,
              false,
              ctx.stream());

  permute_input->Resize({moe_topk * num_rows, hidden_size});
  permute_indices_per_token->Resize({moe_topk, num_rows});

  initialize_moe_routing_kernelLauncher(
      X.data<T>(),
      ctx.template Alloc<T>(permute_input),
      permuted_rows_,
      ctx.template Alloc<int32_t>(permute_indices_per_token),
      num_rows,
      num_rows,
      hidden_size,
      moe_topk,
      ctx.stream());

  token_nums_per_expert->Resize({expert_num});

  compute_total_rows_before_expert<T>(
      permuted_experts_,
      X.data<T>(),
      moe_topk * num_rows,
      expert_num,
      ctx.template Alloc<int64_t>(token_nums_per_expert),
      ctx.stream());
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_CUDA_BF16
PD_REGISTER_KERNEL(moe_dispatch,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeDispatchKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(moe_dispatch,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeDispatchKernel,
                   phi::dtype::float16) {}
#endif

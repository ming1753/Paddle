# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode


def fused_moe(
    x,
    gate_weight,
    ffn1_weight,
    ffn2_weight,
    ffn1_bias=None,
    ffn1_scale=None,
    ffn2_bias=None,
    ffn2_scale=None,
    quant_method="None",
    moe_topk=2,
    norm_topk_prob=True,
    group_moe=False,
):
    """
    Applies fused moe kernel.
    This method requires SM_ARCH in sm75, sm80, sm86.

    Args:
        x (Tensor): the input Tensor. Its shape is [bsz, seq_len, d_model].
        gate_weight (Tensor): the gate Tensor to choose expert. Its shape is [bsz, seq_len, num_experts].
        ffn1_weight (Tensor): the first batch matrix matmul weight. Its shape is [num_experts, d_model, d_feed_forward*2].
        ffn2_weight (Tensor): the second batch matrix matmul weight. Its shape is [num_experts, d_feed_forward, d_model].
        ffn1_bias (Tensor, optional): the first batch matrix matmul bias. Its shape is [num_experts, 1, d_feed_forward*2].
        ffn1_scale (Tensor, optional): the input scale Tensor Provided to weight for dequantization. Its shape is [num_experts, d_feed_forward*2].
        ffn2_bias (Tensor, optional): the second batch matrix matmul bias. Its shape is [num_experts, 1, d_model].
        ffn2_scale (Tensor, optional): the input scale Tensor Provided to weight for dequantization. Its shape is [num_experts, d_model].
        quant_method (string): Currently not supported.
        moe_topk (int): Select the top k experts for each token.
        norm_topk_prob (bool): Whether to normalize the topk probabilities.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_moe

            >>> paddle.set_device('gpu')
            >>> paddle.set_default_dtype("float16")
            >>> x = paddle.randn([10, 128, 1024])
            >>> gate_weight = paddle.randn([10, 128, 8], dtype=paddle.float32)
            >>> ffn1_weight = paddle.randn([8, 1024, 4096])
            >>> ffn1_bias = paddle.randn([8, 1, 4096])
            >>> ffn2_weight = paddle.randn([8, 2048, 1024])
            >>> ffn2_bias = paddle.randn([8, 1, 1024])
            >>> moe_topk = 2
            >>> out = fused_moe(x, gate_weight, ffn1_weight, ffn2_weight, ffn1_bias, None, ffn2_bias, None, "None", moe_topk, True)
            >>> print(out.shape)
            [10, 128, 1024]

    """

    if in_dynamic_or_pir_mode():
        final_out = _C_ops.fused_moe(
            x,
            gate_weight,
            ffn1_weight,
            ffn1_scale,
            ffn1_bias,
            ffn2_weight,
            ffn2_scale,
            ffn2_bias,
            quant_method,
            moe_topk,
            group_moe,
            norm_topk_prob,
        )
        return final_out
    else:
        helper = LayerHelper('fused_moe', **locals())
        final_out = helper.create_variable_for_type_inference(dtype=x.dtype)

        inputs = {
            'x': x,
            'gate_weight': gate_weight,
            'ffn1_weight': ffn1_weight,
            'ffn2_weight': ffn2_weight,
        }
        if ffn1_bias is not None:
            inputs['ffn1_bias'] = ffn1_bias
        if ffn1_scale is not None:
            inputs['ffn1_scale'] = ffn1_scale
        if ffn2_bias is not None:
            inputs['ffn2_bias'] = ffn2_bias
        if ffn2_scale is not None:
            inputs['ffn2_scale'] = ffn2_scale

        helper.append_op(
            type='fused_moe',
            inputs=inputs,
            outputs={'out': final_out},
            attrs={
                'quant_method': quant_method,
                'moe_topk': moe_topk,
                'group_moe': group_moe,
                'norm_topk_prob': norm_topk_prob,
            },
        )
        return final_out


def moe_dispatch(x, gate_out, topk, group_moe=True):
    """
    Dispatches tokens to experts based on gating probabilities.

    This function routes each token to its top-k selected experts according to the gating
    output. It prepares the inputs for expert processing by reordering and scaling.

    Args:
        x (Tensor): The input tensor with shape `[batch_size * seq_len, d_model]`.
        gate_out (Tensor): The gating output probabilities with shape `[batch_size * seq_len, num_experts]`.
        topk (int): The number of top experts to select for each token.
        group_moe (bool, optional): Whether to use group MoE. Default is `True`.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            - permute_input (Tensor): The permuted input tensor ready for expert processing.
            - token_nums_per_expert (Tensor): The number of tokens assigned to each expert.
            - scatter_index (Tensor): The index mapping for scattering outputs back to the original order.
            - expert_scales_float (Tensor): The scaling factors for each expert's outputs.
            - top_k_indices (Tensor): The indices of the selected experts for each token.
            - group_max_prob (Tensor): The maximum probability in each group (used in group MoE).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.incubate.nn.functional import moe_dispatch

            >>> x = paddle.randn([1280, 768]) # 1280 = bs * 128
            >>> gate_out = paddle.rand([1280, 48])
            >>> topk = 6
            >>> (
            ...     permute_input,
            ...     token_nums_per_expert,
            ...     scatter_index,
            ...     expert_scales_float,
            ...     top_k_indices,
            ...     group_max_prob,
            ... ) = moe_dispatch(x, gate_out, topk, True)
            >>> print(permute_input.shape)
            [7680, 768]
            >>> print(token_nums_per_expert.shape)
            [48]
            >>> print(group_max_prob.shape)
            [7680]

    """

    if in_dynamic_or_pir_mode():
        (
            permute_input,
            token_nums_per_expert,
            scatter_index,
            expert_scales_float,
            top_k_indices,
            group_max_prob,
        ) = _C_ops.moe_dispatch(x, gate_out, topk, group_moe)
        return (
            permute_input,
            token_nums_per_expert,
            scatter_index,
            expert_scales_float,
            top_k_indices,
            group_max_prob,
        )

    helper = LayerHelper('moe_dispatch', **locals())

    outputs_dict = {}

    permute_input = helper.create_variable_for_type_inference(dtype=x.dtype)
    token_nums_per_expert = helper.create_variable_for_type_inference(
        dtype="int64"
    )
    scatter_index = helper.create_variable_for_type_inference(dtype="int32")
    expert_scales_float = helper.create_variable_for_type_inference(
        dtype="float32"
    )
    top_k_indices = helper.create_variable_for_type_inference(dtype="int32")
    group_max_prob = helper.create_variable_for_type_inference(dtype="float32")

    outputs_dict["out"] = permute_input
    outputs_dict["token_nums_per_expert"] = token_nums_per_expert
    outputs_dict["scatter_index"] = scatter_index
    outputs_dict["expert_scales_float"] = expert_scales_float
    outputs_dict["expert_for_source_row_tensor"] = top_k_indices
    outputs_dict["group_max_prob"] = group_max_prob

    inputs = {"X": x}
    inputs["gating_output"] = gate_out

    helper.append_op(
        type='moe_dispatch',
        inputs=inputs,
        attrs={"moe_topk": topk, "group_moe": group_moe},
        outputs=outputs_dict,
    )

    return (
        permute_input,
        token_nums_per_expert,
        scatter_index,
        expert_scales_float,
        top_k_indices,
        group_max_prob,
    )


def moe_ffn(
    X,
    rows_per_expert,
    ffn1_weight,
    ffn1_scale,
    ffn1_bias,
    ffn2_weight,
    ffn2_scale,
    quant_method,
):
    """
    Applies the feed-forward network (FFN) to the dispatched tokens for each expert.

    This function performs the FFN computation for the tokens assigned to each expert.
    It supports optional quantization methods for the weights.

    Args:
        X (Tensor): The input tensor after dispatching, with shape `[total_tokens, d_model]`.
        rows_per_expert (Tensor): The number of tokens assigned to each expert.
        ffn1_weight (Tensor): The weight for the first linear layer, with shape `[num_experts, d_model, d_ffn * 2]`.
        ffn1_scale (Tensor, optional): Scale tensor for dequantization of `ffn1_weight`, with shape `[num_experts, d_ffn * 2]`.
        ffn1_bias (Tensor, optional): Bias for the first linear layer, with shape `[num_experts, 1, d_ffn * 2]`.
        ffn2_weight (Tensor, optional): The weight for the second linear layer, with shape `[num_experts, d_ffn, d_model]`.
        ffn2_scale (Tensor, optional): Scale tensor for dequantization of `ffn2_weight`, with shape `[num_experts, d_model]`.
        quant_method (str, optional): Quantization method to be used. Currently not supported. Default is `"None"`.

    Returns:
        Tensor: The output tensor after FFN computation, with shape `[total_tokens, d_model]`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.incubate.nn.functional import moe_ffn

            >>> permute_input = paddle.randn([7680, 768])  # 7680 = bs * 128 *6
            >>> rows_per_expert = paddle.to_tensor([48], dtype='int64')
            >>> ffn1_weight = paddle.randn([48, 768, 6144])
            >>> ffn1_bias = paddle.randn([48, 1, 6144])
            >>> ffn2_weight = paddle.randn([48, 3072, 768])
            >>> out = moe_ffn(permute_input, rows_per_expert, ffn1_weight, None, ffn1_bias, ffn2_weight)
            >>> print(out.shape)
            [7680, 768]

    """

    if in_dynamic_or_pir_mode():
        return _C_ops.moe_ffn(
            X,
            rows_per_expert,
            ffn1_weight,
            ffn1_scale,
            ffn1_bias,
            ffn2_weight,
            ffn2_scale,
            quant_method,
        )

    helper = LayerHelper('moe_ffn', **locals())

    outputs_dict = {}

    out = helper.create_variable_for_type_inference(dtype=X.dtype)
    outputs_dict["ffn_out"] = out

    inputs = {"X": X}
    inputs["rows_per_expert"] = rows_per_expert
    inputs["ffn1_weight"] = ffn1_weight
    inputs["ffn2_weight"] = ffn2_weight

    if ffn1_scale is not None:
        inputs["ffn1_scale"] = ffn1_scale
    if ffn1_bias is not None:
        inputs["ffn1_bias"] = ffn1_bias
    if ffn2_scale is not None:
        inputs["ffn2_scale"] = ffn2_scale
    helper.append_op(
        type='moe_ffn',
        inputs=inputs,
        attrs={
            "quant_method": quant_method,
        },
        outputs=outputs_dict,
    )

    return out


def moe_reduce(
    fc2_result,
    fc2_expert_biases,
    expert_scales_float,
    expanded_source_row_to_expanded_dest_row,
    topk_indices,
    norm_topk_prob,
):
    """
    Reduces the outputs from experts back to the original token order.

    This function gathers the outputs from different experts and combines them according to
    the original token positions. It also applies scaling factors to the outputs.

    Args:
        fc2_result (Tensor): The output tensor from experts' FFN computation, with shape `[total_tokens, d_model]`.
        fc2_expert_biases (Tensor, optional): The biases for the second FFN layer, with shape `[num_experts, 1, d_model]`.
        expert_scales_float (Tensor): The scaling factors for each expert's outputs, with shape `[batch_size * seq_len, moe_topk]`.
        expanded_source_row_to_expanded_dest_row (Tensor): The index mapping from expert outputs to original token positions.
        topk_indices (Tensor): The indices of the selected experts for each token.
        norm_topk_prob (bool): Whether to normalize the top-k probabilities.

    Returns:
        Tensor: The final output tensor with shape `[batch_size * seq_len, d_model]`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.incubate.nn.functional import moe_reduce

            >>> fc2_result = paddle.randn([7680, 768])  7680 = bs*128*6
            >>> fc2_expert_biases = paddle.randn([48, 1, 768])
            >>> expert_scales_float = paddle.rand([1280, 6])
            >>> expanded_source_row_to_expanded_dest_row = paddle.to_tensor([6, 1280], dtype='int32')
            >>> topk_indices = paddle.to_tensor([1280, 6], dtype='int32')
            >>> norm_topk_prob = False
            >>> output = moe_reduce(
            ...     fc2_result,
            ...     fc2_expert_biases,
            ...     expert_scales_float,
            ...     expanded_source_row_to_expanded_dest_row,
            ...     topk_indices,
            ...     norm_topk_prob,
            ... )
            >>> print(output.shape)
            [1280, 768]

    """

    if in_dynamic_or_pir_mode():
        return _C_ops.moe_reduce(
            fc2_result,
            fc2_expert_biases,
            expert_scales_float,
            expanded_source_row_to_expanded_dest_row,
            topk_indices,
            norm_topk_prob,
        )

    helper = LayerHelper('moe_reduce', **locals())

    outputs_dict = {}

    output = helper.create_variable_for_type_inference(dtype=fc2_result.dtype)
    outputs_dict["output"] = output

    inputs = {"fc2_result": fc2_result}
    if fc2_expert_biases is not None:
        inputs["fc2_expert_biases"] = fc2_expert_biases
    inputs["expert_scales_float"] = expert_scales_float
    inputs["expanded_source_row_to_expanded_dest_row"] = (
        expanded_source_row_to_expanded_dest_row
    )
    inputs["topk_indices"] = topk_indices

    helper.append_op(
        type='moe_reduce',
        inputs=inputs,
        attrs={
            "norm_topk_prob": norm_topk_prob,
        },
        outputs=outputs_dict,
    )

    return output

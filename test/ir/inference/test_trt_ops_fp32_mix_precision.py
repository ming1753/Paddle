# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig


class TensorRTConv2dFp32MixPrecisionTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32"
            )
            conv_out = paddle.static.nn.conv2d(
                input=data,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                use_cudnn=self.use_cudnn,
                act=None,
            )
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }

        output_name = ["save_infer_model/scale_0.tmp_0"]
        self.enable_trt = True
        self.trt_parameters = InferencePassTest.TensorRTParam(
            1 << 30,
            1,
            0,
            AnalysisConfig.Precision.Half,
            False,
            False,
            False,
            False,
            set(output_name),
        )
        self.fetch_list = [conv_out]

    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = [1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)


if __name__ == "__main__":
    unittest.main()

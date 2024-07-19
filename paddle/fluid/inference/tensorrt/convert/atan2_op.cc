/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

/*
 * Atan2 Op
 */
class Atan2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a atan2 op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string x_name = op_desc.Input("X1").front();
    std::string y_name = op_desc.Input("X2").front();
    std::string output_name = op_desc.Output("Out").front();

    auto* x = engine_->GetITensor(x_name);
    auto* y = engine_->GetITensor(y_name);
    auto* shape_tensor = Shape(x);
    auto rank = x->getDimensions().nbDims;
    auto* zero = FillConstantLayer(shape_tensor, rank, 0.f);
    // auto* one = FillConstantLayer(shape_tensor, rank, 1.f);
    auto* two = FillConstantLayer(shape_tensor, rank, 2.f);
    auto* PI =
        FillConstantLayer(shape_tensor, rank, static_cast<float>(3.1415926535));
    // Calculate x_zero, y_zero (whether inputs are zero)
    auto* x_zero = TRT_ENGINE_ADD_LAYER(engine_,
                                        ElementWise,
                                        *x,
                                        *zero,
                                        nvinfer1::ElementWiseOperation::kEQUAL)
                       ->getOutput(0);
    auto* y_zero = TRT_ENGINE_ADD_LAYER(engine_,
                                        ElementWise,
                                        *y,
                                        *zero,
                                        nvinfer1::ElementWiseOperation::kEQUAL)
                       ->getOutput(0);

    // Get sign of inputs
    auto* x_positive =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x,
                             *zero,
                             nvinfer1::ElementWiseOperation::kGREATER)
            ->getOutput(0);

    auto* x_zero_positive =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x_zero,
                             *x_positive,
                             nvinfer1::ElementWiseOperation::kOR)
            ->getOutput(0);
    auto* x_negative =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x,
                             *zero,
                             nvinfer1::ElementWiseOperation::kLESS)
            ->getOutput(0);
    auto* y_positive =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *y,
                             *zero,
                             nvinfer1::ElementWiseOperation::kGREATER)
            ->getOutput(0);

    auto* y_negative =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *y,
                             *zero,
                             nvinfer1::ElementWiseOperation::kLESS)
            ->getOutput(0);
    // Calculate atan(x/y)
    auto* intermediate_div = Div(x, y);
    auto* atan2_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Unary, *intermediate_div, nvinfer1::UnaryOperation::kATAN);
    auto* atan_val = atan2_layer->getOutput(0);

    // atan(x/y)+π if x≥0 and y<0,
    auto* atan_add_pi = Sum(atan_val, PI);
    // atan(x/y)-π if x<0 and y<0,
    auto* atan_sub_pi = Sub(atan_val, PI);

    // atan(x/y)+π if x≥0 and y<0,
    auto* atan_corrected_indices =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x_zero_positive,
                             *y_negative,
                             nvinfer1::ElementWiseOperation::kAND)
            ->getOutput(0);
    auto* atan_corrected =
        TRT_ENGINE_ADD_LAYER(
            engine_, Select, *atan_corrected_indices, *atan_add_pi, *atan_val)
            ->getOutput(0);

    // atan(x/y)-π if x<0 and y<0,
    auto* atan_corrected_indices_2 =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x_negative,
                             *y_negative,
                             nvinfer1::ElementWiseOperation::kAND)
            ->getOutput(0);
    auto* atan_corrected_2 = TRT_ENGINE_ADD_LAYER(engine_,
                                                  Select,
                                                  *atan_corrected_indices_2,
                                                  *atan_sub_pi,
                                                  *atan_corrected)
                                 ->getOutput(0);

    // atan(x/y) if y>0
    auto* atan_output =
        TRT_ENGINE_ADD_LAYER(
            engine_, Select, *y_positive, *atan_val, *atan_corrected_2)
            ->getOutput(0);

    // pi_over_2_tensor
    auto* pi_over_2_tensor = Div(PI, two);
    auto* minus_pi_over_2_tensor = Div(Sub(zero, PI), two);

    // π/2 if x>0 and y=0,
    auto* pi_over_2_output_indices =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x_positive,
                             *y_zero,
                             nvinfer1::ElementWiseOperation::kAND)
            ->getOutput(0);
    auto* pi_over_2_output = TRT_ENGINE_ADD_LAYER(engine_,
                                                  Select,
                                                  *pi_over_2_output_indices,
                                                  *pi_over_2_tensor,
                                                  *atan_output)
                                 ->getOutput(0);

    // -π/2 if x<0 and y=0,
    auto* minus_pi_over_2_output_indices =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x_negative,
                             *y_zero,
                             nvinfer1::ElementWiseOperation::kAND)
            ->getOutput(0);
    auto* minus_pi_over_2_output =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Select,
                             *minus_pi_over_2_output_indices,
                             *minus_pi_over_2_tensor,
                             *pi_over_2_output)
            ->getOutput(0);

    // 0 if x=0 and y=0,
    auto* zero_output_indices =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *x_zero,
                             *y_zero,
                             nvinfer1::ElementWiseOperation::kAND)
            ->getOutput(0);
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Select, *zero_output_indices, *zero, *minus_pi_over_2_output);

    ReplenishLayerAndOutput(layer, "atan2", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(atan2, Atan2OpConverter);

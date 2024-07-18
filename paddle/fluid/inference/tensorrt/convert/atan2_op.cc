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

    auto* intermediate_div = Div(x, y);
    auto* atan2_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Unary, *intermediate_div, nvinfer1::UnaryOperation::kATAN);
    auto* atan2_intermediate = atan2_layer->getOutput(0);
    auto* shape_tensor = Shape(x);
    auto rank = x->getDimensions().nbDims;
    auto* zero = FillConstantLayer(shape_tensor, rank, 0.f);
    auto* one = FillConstantLayer(shape_tensor, rank, 1.f);
    auto* two = FillConstantLayer(shape_tensor, rank, 2.f);
    auto* PI =
        FillConstantLayer(shape_tensor, rank, static_cast<float>(3.1415926535));

    auto* x_mask =
        Cast(TRT_ENGINE_ADD_LAYER(engine_,
                                  ElementWise,
                                  *x,
                                  *zero,
                                  nvinfer1::ElementWiseOperation::kLESS)
                 ->getOutput(0),
             nvinfer1::DataType::kFLOAT);

    auto* y_mask =
        Cast(TRT_ENGINE_ADD_LAYER(engine_,
                                  ElementWise,
                                  *y,
                                  *zero,
                                  nvinfer1::ElementWiseOperation::kLESS)
                 ->getOutput(0),
             nvinfer1::DataType::kFLOAT);

    x_mask = TRT_ENGINE_ADD_LAYER(engine_,
                                  ElementWise,
                                  *x_mask,
                                  *two,
                                  nvinfer1::ElementWiseOperation::kPROD)
                 ->getOutput(0);
    x_mask = TRT_ENGINE_ADD_LAYER(engine_,
                                  ElementWise,
                                  *x_mask,
                                  *one,
                                  nvinfer1::ElementWiseOperation::kSUB)
                 ->getOutput(0);
    x_mask = TRT_ENGINE_ADD_LAYER(engine_,
                                  ElementWise,
                                  *x_mask,
                                  *PI,
                                  nvinfer1::ElementWiseOperation::kPROD)
                 ->getOutput(0);

    auto* correction_term =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *y_mask,
                             *x,
                             nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *atan2_intermediate,
                                       *correction_term,
                                       nvinfer1::ElementWiseOperation::kSUB);

    ReplenishLayerAndOutput(layer, "atan2", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(atan2, Atan2OpConverter);

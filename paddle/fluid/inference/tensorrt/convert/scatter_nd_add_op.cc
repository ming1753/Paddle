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
 * scatter_nd_add Op
 */
class ScatterNdAddOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a scatter_nd_add op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string index_name = op_desc.Input("Index").front();
    std::string update_name = op_desc.Input("Updates").front();
    std::string output_name = op_desc.Output("Out").front();

    auto* input_tensor = engine_->GetITensor(input_name);
    auto* index_tensor = engine_->GetITensor(index_name);
    auto* update_tensor = engine_->GetITensor(update_name);
    auto* input_shape_tensor = Shape(input_tensor);
    nvinfer1::Dims input_dims = input_tensor->getDimensions();
    auto rank = input_dims.nbDims;
    auto* zero_tensor = FillConstantLayer(input_shape_tensor, rank, 0.f);
    auto* value_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                             Scatter,
                                             *zero_tensor,
                                             *index_tensor,
                                             *update_tensor,
                                             nvinfer1::ScatterMode::kND);
    value_layer->setAxis(0);
    auto* value_tensor = value_layer->getOutput(0);
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *input_tensor,
                                       *value_tensor,
                                       nvinfer1::ElementWiseOperation::kSUM);
    ReplenishLayerAndOutput(layer, "scatter_nd_add", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(scatter_nd_add, ScatterNdAddOpConverter);

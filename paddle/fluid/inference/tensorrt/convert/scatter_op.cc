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
 * scatter Op
 */
class ScatterOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a scatter op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string index_name = op_desc.Input("Ids").front();
    std::string update_name = op_desc.Input("Updates").front();
    std::string output_name = op_desc.Output("Out").front();
    auto* input_tensor = engine_->GetITensor(input_name);
    auto* index_tensor = engine_->GetITensor(index_name);
    auto* update_tensor = engine_->GetITensor(update_name);
    auto input_dims = input_tensor->getDimensions();
    auto index_dims = index_tensor->getDimensions();
    if (input_dims.nbDims == index_dims.nbDims) {
      index_tensor = Reshape(
          index_tensor, Concat({Shape(index_tensor), Add1DConstantLayer(1)}));
    }
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       Scatter,
                                       *input_tensor,
                                       *index_tensor,
                                       *update_tensor,
                                       nvinfer1::ScatterMode::kND);
    layer->setAxis(0);
    ReplenishLayerAndOutput(layer, "scatter", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(scatter, ScatterOpConverter);

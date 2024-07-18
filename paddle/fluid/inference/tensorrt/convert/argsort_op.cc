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
 * Argsort Op
 */
class ArgsortOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a argsort op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();
    std::string indices_name = op_desc.Output("Indices").front();
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    bool descending = PADDLE_GET_CONST(bool, op_desc.GetAttr("descending"));
    auto* input_tensor = engine_->GetITensor(input_name);
    nvinfer1::Dims input_tensor_dims = input_tensor->getDimensions();
    nvinfer1::TopKOperation operation = nvinfer1::TopKOperation::kMIN;
    if (descending) {
      operation = nvinfer1::TopKOperation::kMAX;
    }
    if (axis < 0) {
      axis += input_tensor_dims.nbDims;
    }
    int k = 5;
    auto* size_tensor = Add1DConstantLayer(k, "", true);
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, TopK, *input_tensor, operation, 0, 1 << axis);
    layer->setInput(1, *size_tensor);
    ReplenishLayerAndOutput(
        layer, "argsort", {output_name, indices_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(argsort, ArgsortOpConverter);

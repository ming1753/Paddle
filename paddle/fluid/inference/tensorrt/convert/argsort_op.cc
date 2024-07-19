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
    int k = PADDLE_GET_CONST(int, op_desc.GetAttr("trick_k"));
    bool need_cast = PADDLE_GET_CONST(bool, op_desc.GetAttr("need_cast"));
    if (need_cast) {
      auto* cast_layer1 = TRT_ENGINE_ADD_LAYER(engine_, Identity, *input_tensor);
      cast_layer1->setOutputType(0, nvinfer1::DataType::kFLOAT);
      cast_layer1->getOutput(0)->setType(nvinfer1::DataType::kFLOAT);
      auto* input = cast_layer1->getOutput(0);

      auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, TopK, *input, operation, k, 1 << axis);
      auto* tmp_output = layer->getOutput(0);

      auto* cast_layer2 = TRT_ENGINE_ADD_LAYER(engine_, Identity, *tmp_output);
      cast_layer2->setOutputType(0, nvinfer1::DataType::kINT32);
      cast_layer2->getOutput(0)->setType(nvinfer1::DataType::kINT32);

      std::string layer_name = "argsort (Output: ";
      cast_layer2->getOutput(0)->setName(output_name.c_str());
      engine_->SetITensor(output_name, cast_layer2->getOutput(0));
      layer_name += output_name + ", ";
      
      layer->getOutput(1)->setName(indices_name.c_str());
      engine_->SetITensor(indices_name, layer->getOutput(1));
      layer_name += indices_name;
      layer->setName((layer_name + ")").c_str());
    } else {
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, TopK, *input_tensor, operation, k, 1 << axis);
    ReplenishLayerAndOutput(
        layer, "argsort", {output_name, indices_name}, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(argsort, ArgsortOpConverter);

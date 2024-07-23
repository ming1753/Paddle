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
 * where_index Op
 */
class WhereIndexOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a where_index op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("Condition").front();
    auto output_name = op_desc.Output("Out")[0];
    auto* input = engine_->GetITensor(input_name);

    auto* nonzero_layer = TRT_ENGINE_ADD_LAYER(
        engine_, NonZero, *input);

    auto* nonzero_layer_output = nonzero_layer->getOutput(0);
    
    auto* transpose_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *nonzero_layer_output);
              transpose_layer->setName("reshape_before_matrix(Output: just_for_test)");
    nvinfer1::Permutation perm = {1, 0};
    transpose_layer->setFirstTranspose(perm);

    ReplenishLayerAndOutput(transpose_layer, "where_index", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(where_index, WhereIndexOpConverter);

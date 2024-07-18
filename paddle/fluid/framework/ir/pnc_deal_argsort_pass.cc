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

#include "paddle/fluid/framework/ir/pnc_deal_argsort_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES             \
  GET_IR_NODE(argsort_X);     \
  GET_IR_NODE(argsort_Op);    \
  GET_IR_NODE(argsort_Out);

PncDealArgsortPass::PncDealArgsortPass() {
  AddOpCompat(OpCompat("argsort"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<int>()
      .End()
      .AddAttr("descending")
      .IsType<bool>()
      .End();
}

void PncDealArgsortPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "pnc_deal_argsort_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument(
          "Scope in PncDealArgsortPass should not be null."));
  // Create pattern
  patterns::PncDealArgsortPartern pattern(gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    /*
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "pnc_deal_argsort_pass compat check failed.";
      return;
    }
    */

    if (!argsort_Op->Op()->HasAttr("stable")) {
        argsort_Op->Op()->SetAttr("stable", false);
    }
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(pnc_deal_argsort_pass,
              paddle::framework::ir::PncDealArgsortPass);

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

#include "paddle/fluid/framework/ir/pnc_where_index_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES               \
  GET_IR_NODE(where_index_x);   \
  GET_IR_NODE(where_index_op);  \
  GET_IR_NODE(where_index_out); \
  GET_IR_NODE(gather_nd_x);     \
  GET_IR_NODE(gather_nd_op);    \
  GET_IR_NODE(gather_nd_out);

PncWhereIndexPass::PncWhereIndexPass() {
  AddOpCompat(OpCompat("gather_nd"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}


void PncWhereIndexPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "pnc_where_index_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument(
          "Scope in PncWhereIndexPass should not be null."));
  // Create pattern
  patterns::PncWhereIndexPartern pattern(gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    gather_nd_op->Op()->SetInput("Index", {where_index_x->Name()});

    IR_NODE_LINK_TO(where_index_x, gather_nd_op); 

    std::unordered_set<const Node*> nodes2rm = {};

    nodes2rm.insert(where_index_op);
    nodes2rm.insert(where_index_out);

    GraphSafeRemoveNodes(graph, nodes2rm);

    gather_nd_op->Op()->SetAttr("where_index", true);
    
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(pnc_where_index_pass,
              paddle::framework::ir::PncWhereIndexPass);

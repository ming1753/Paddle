#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import warnings

from .parallel_base import ParallelOptimizer, parallelize_model_and_optimizer
from .pipeline_parallel import pipeline_parallel
from .sharded_data_parallel import sharded_data_parallel
from .tensor_parallel import tensor_parallel


def parallelize(
    model, optimizer, mesh=None, dp_config=None, mp_config=None, pp_config=None
):
    # TODO(yaliu): global mesh and split axis support
    if pp_config is not None:
        assert isinstance(pp_config, dict)
        model, optimizer = pipeline_parallel(
            model,
            optimizer,
            pp_config,
        )
    if mp_config is not None:
        assert isinstance(mp_config, dict)
        model, optimizer = tensor_parallel(model, optimizer, mp_config)
    if dp_config is not None:
        assert isinstance(dp_config, dict)
        if 'sharding_level' not in dp_config.keys():
            warnings.warn(
                "The dp_config doesn't contain sharding_level, will run under dp."
            )
        model, optimizer = sharded_data_parallel(
            model,
            optimizer,
            config=dp_config,
        )
    model, optimizer = parallelize_model_and_optimizer(model, optimizer)
    return model, optimizer


has_parallelized_model = False


def parallelize_model(
    model, mesh=None, dp_config=None, mp_config=None, pp_config=None
):
    global has_parallelized_model
    has_parallelized_model = True
    model, _ = parallelize(model, None, mesh, dp_config, mp_config, pp_config)
    return model


def parallelize_optimizer(
    optimizer, mesh=None, dp_config=None, mp_config=None, pp_config=None
):
    global has_parallelized_model
    assert (
        has_parallelized_model
    ), "Please parallelize the model before parallelize optimizer."
    param_list = optimizer._parameter_list
    if isinstance(param_list[0], dict):
        for param_group in param_list:
            for param in param_group['params']:
                assert (
                    param.is_dist()
                ), "Please use model after parallelize to create optimizer."
    else:
        for param in param_list:
            assert (
                param.is_dist()
            ), "Please use model after parallelize to create optimizer."

    level = None
    sharding_mesh_dim = None
    if dp_config is not None:
        if 'sharding_level' not in dp_config.keys():
            warnings.warn(
                "The dp_config doesn't contain sharding_level, will run under dp."
            )
        level = dp_config.get('sharding_level')
        sharding_mesh_dim = dp_config.get('sharding_mesh_dim', "dp")
    optimizer = ParallelOptimizer(optimizer, level, sharding_mesh_dim)
    optimizer = optimizer.parallelize()
    return optimizer

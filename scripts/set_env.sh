#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export ACLTRANSFORMER_HOME_PATH=`pwd`
export ASDOPS_HOME_PATH=`pwd`
export ASDOPS_OPS_PATH=$ACLTRANSFORMER_HOME_PATH/ops
export LD_LIBRARY_PATH=$ACLTRANSFORMER_HOME_PATH/lib:$ACLTRANSFORMER_HOME_PATH/examples:$LD_LIBRARY_PATH
export PATH=$ACLTRANSFORMER_HOME_PATH/bin:$PATH
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
export LD_LIBRARY_PATH=$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export TASK_QUEUE_ENABLE=0 #Torch npu
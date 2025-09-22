#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

handle_error(){
    rm -f linear_parallel_generation
    rm -f *.bin
}

trap handle_error ERR

set -e

cxx_abi=$(python3 -c '
try:
    import torch
    print("1" if torch.compiled_with_cxx11_abi() else "0")
except ImportError:
    print("1")
')

echo "Using cxx_abi=$cxx_abi"

g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" \
linear_parallel_generation.cpp ../demo_util.h -l atb -l ascendcl -l hccl -l nnopbase -l opapi -o linear_parallel_generation
./linear_parallel_generation

python linear_parallel_mc2_linear_reduce_scatter.py

rm -f linear_parallel_generation
rm -f *.bin
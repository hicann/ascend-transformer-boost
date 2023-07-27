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
import sys
import os
import torch
sys.path.append(os.path.dirname(__file__))
from tensor_file import read_tensor  # NOQA: E402


def main():
    tensor1 = read_tensor(sys.argv[1])
    tensor2 = read_tensor(sys.argv[2])

    print("tensor1:" + str(tensor1))
    print("tensor2:" + str(tensor2))
    print("tensor1.shape", tensor1.shape, ", dtype:", tensor1.dtype)
    print("tensor2.shape", tensor2.shape, ", dtype:", tensor2.dtype)

    sub_tensor = tensor1 - tensor2
    abs_tensor = sub_tensor.abs()

    absolute_err = 0
    relative_err = 0
    
    if abs_tensor.numel() != 0:
        absolute_err = abs_tensor.type(torch.float64).sum() / abs_tensor.numel()
        relative_err = torch.div(abs_tensor, tensor2.abs()).type(torch.float64).sum() / abs_tensor.numel()

    print("Absolute error: ")
    print(absolute_err)
    print("Relative error:")
    print(relative_err)



if __name__ == "__main__":
    main()

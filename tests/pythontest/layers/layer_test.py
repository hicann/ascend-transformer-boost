# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
import os
import unittest
import json
import numpy
import torch
import torch_npu

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "examples/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


class LayerTest(unittest.TestCase):
    def execute(self, layer_name, layer_param, in_tensors):
        layer = torch.classes.LayerTorch.LayerTorch(
            layer_name)
        if isinstance(layer_param, dict):
            layer.set_param(json.dumps(layer_param))
        elif isinstance(layer_param, str):
            layer.set_param(layer_param)
        out_tensors = layer.execute(in_tensors)
        golden_out_tensors = self.golden_calc(in_tensors)
        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.02, atol=0.02)

    def __golden_compare_all(self, out_tensors, golden_out_tensors):
        self.assertEqual(len(out_tensors), len(golden_out_tensors))
        tensor_count = len(out_tensors)
        for i in range(tensor_count):
            self.assertTrue(self.golden_compare(
                out_tensors[i], golden_out_tensors[i]))

    def __get_npu_device(self):
        npu_device = os.environ.get("ASDOPS_NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device

    def __create_tensor(self, dtype, format, shape, minValue, maxValue, device=None):
        if device is None:
            device = self.__get_npu_device()
        input = numpy.random.uniform(minValue, maxValue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input)
        npu_input = torch.from_numpy(input).to(device)
        if format != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, format)
        return cpu_input, npu_input

#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2018. All rights reserved
"""
common launcher
"""
import abc
import os
import time

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu


class BaseLauncher:
    """
    BaseLauncher
    """

    def __init__(self, device_ids: str = "0", using_acl_transformers=True):
        self.set_torch_npu_env(device_ids, using_acl_transformers)
        self.model, self.tokenizer = self.init_model()
        self.fit_npu(self.model)

    @staticmethod
    def set_torch_npu_env(device_ids, using_acl_transformers=True):
        """

        :param device_ids:
        :param using_acl_transformers:
        :return:
        """
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_ids
        if using_acl_transformers:
            # torch.npu.config.allow_internal_format = False
            print("torch.npu.set_compile_mode(jit_compile=False)")
            torch.npu.set_compile_mode(jit_compile=False)  # 使用加速库替换DecodeLayer后可以开启

        option = {"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceProd"}
        # option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,SoftmaxV2,LayerNormGrad,ReduceProd"
        torch.npu.set_option(option)

    @staticmethod
    def fit_npu(model):
        """
        芯片适配
        :param model:
        :return:
        """
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in [104, 220, 221, 222, 223]:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.data = module.weight.data.npu_format_cast(2)
            print("soc version: ", soc_version, " is 910B, support ND")
        else:
            # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if name == 'lm_head':
                        # eliminate TransData op before lm_head calculation
                        module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                    module.weight.data = module.weight.data.npu_format_cast(29)
            print("soc version: ", soc_version, " is not 910B, support NZ")

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.weight.data = module.weight.data.npu_format_cast(2)

    @abc.abstractmethod
    def init_model(self):
        """
        模型初始化
        :return:
        """
        ...

    @abc.abstractmethod
    def infer(self, query):
        """
        推理代码
        :param query:
        :return:
        """
        inputs = self.tokenizer([query], return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.npu()
        gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": False,
                      "repetition_penalty": 1.1}
        with torch.no_grad():
            start_time = time.time()
            output = self.model.generate(**inputs, **gen_kwargs)
            end_time = time.time()
            time_cost = end_time - start_time
        final_answer = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        print(final_answer)
        print(f"cost {time_cost}s")
        new_tokens = len(output[0]) - len(inputs.input_ids[0])
        print(f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s")
        return output

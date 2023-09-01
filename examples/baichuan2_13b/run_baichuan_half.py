import inspect
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from launcher import BaseLauncher


class BaichuanLM(BaseLauncher):
    def init_model(self):
        """
        模型初始化
        :return:
        """
        pwd = os.path.realpath(os.path.dirname(__file__))
        model_path = os.path.join(pwd, "..", "model")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().npu()
        model.eval()
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id  # set as the token
        if tokenizer.pad_token_id == 64000:
            tokenizer.pad_token_id = 0  # for baichuan model (need fix)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))
        print("##########")
        print(f"load model from {os.path.basename(inspect.getmodule(model).__file__)}")
        print("##########")

        return model, tokenizer

    def infer(self, query):
        """
        推理代码
        :param query:
        :return:
        """
        inputs = self.tokenizer(query, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v.npu()
        with torch.no_grad():
            start_time = time.time()
            # pred = self.model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1) # CANN7.0算子缺失，不能开启
            pred = self.model.generate(**inputs, max_new_tokens=64)
            end_time = time.time()
            time_cost = end_time - start_time
        output = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        print(output)
        print(f"cost {time_cost}s")
        new_tokens = len(pred[0]) - len(inputs.input_ids[0])
        print(f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s")
        return output


if __name__ == '__main__':
    baichuan = BaichuanLM(device_ids="2", using_acl_transformers=True)
    print("---------------warm-up---------------")
    baichuan.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->')

    print("---------------inference---------------")
    baichuan.infer('登鹳雀楼->王之涣\n夜雨寄北->')

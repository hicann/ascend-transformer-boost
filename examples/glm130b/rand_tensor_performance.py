import time
import sys
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

def warm_up(model):
    seq_len = 16
    input_ids = torch.randint(150000, (1, seq_len), device=torch.npu.current_device(), dtype=torch.int64)
    input_ids[:, -2] = 150001
    input_ids[:, -1] = 150004
    position_ids = torch.arange(seq_len, device=torch.npu.current_device(), dtype=torch.int64).view(1, -1)
    attention_mask = torch.randn(1, 1, seq_len, seq_len, device=torch.npu.current_device()) < 0.5
    with torch.no_grad():
        logits, *_ = model(
            input_ids,
            position_ids,
            attention_mask,
        )

    for i in range(5):
        input_ids = torch.randint(150000, (1, 1), device=torch.npu.current_device(), dtype=torch.int64)
        position_ids = torch.arange(1, device=torch.npu.current_device(), dtype=torch.int64).view(1, -1)
        attention_mask = torch.randn(1, 1, 1, seq_len + i + 1, device=torch.npu.current_device()) < 0.5
        logits, *_ = model(
            input_ids,
            position_ids,
            attention_mask,
        )

# 全量+增量
def full_and_incremental_test(seq_len, batch, test_cycle, model):
    warm_up(model)
    past_key_values = None
    input_ids = torch.randint(150000, (batch, seq_len), device=torch.npu.current_device())
    input_ids[:, -2] = 150001
    input_ids[:, -1] = 150004
    position_ids = torch.arange(seq_len, device=torch.npu.current_device(), dtype=torch.int64).view(1, -1)
    attention_mask = torch.randn(1, 1, seq_len, seq_len, device=torch.npu.current_device()) < 0.5
    model_time = []
    for i in range(test_cycle):
        torch.npu.synchronize()
        model_start = time.time()
        logits, *_ = model(
            input_ids,
            position_ids,
            attention_mask
        )
        # synchronize to make sure the model time is correct.
        torch.npu.synchronize()
        model_time.append(time.time() - model_start)
        input_ids = torch.randint(150000, (batch, 1), device=torch.npu.current_device())
        position_ids = torch.arange(1, device=torch.npu.current_device(), dtype=torch.int64).view(1, -1)
        attention_mask = torch.randn(1, 1, 1, seq_len + i + 1, device=torch.npu.current_device()) < 0.5

    if torch.distributed.get_rank() == 0:
        print('Batch size = ', batch)
        print('Input seqlen = ', seq_len)
        print('Output seqlen = {}, takes {} ms.'.format(
            test_cycle, round(sum(model_time) * 1000, 4)))
        print('E2E performance is {} token/second.'.format(
            round((test_cycle) / (sum(model_time)), 4)))
        print('First token\'s model latency is {} ms.'.format(
            round(model_time[0] * 1000, 2)))
        print('Model latency is {} ms.'.format(
            round((sum(model_time) - model_time[0]) * 1000 / (test_cycle - 1), 2)))
        print('Model latency list is:{}'.format(
            [round(item * 1000, 2) for item in model_time]))
    return round(model_time[0] * 1000, 2), round((sum(model_time) - model_time[0]) * 1000 / (test_cycle - 1), 2)

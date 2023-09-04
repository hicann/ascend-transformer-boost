import time
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from initialize import initialize, initialize_model_and_tokenizer
from rand_tensor_performance import full_and_incremental_test

soc_version_map = {-1: "unknown soc version",
                   100: "910PremiumA", 101: "910ProA", 102: "910A", 103: "910ProB", 104: "910B",
                   200: "310P1", 201: "310P2", 202:  "310P3", 203: "310P4",
                   220: "910B1", 221: "910B2", 222: "910B3", 223: "910B4",
                   240: "310B1", 241: "310B2", 242: "310B3",
                   250: "910C1", 251: "910C2", 252: "910C3", 253: "910C4"
                   }

if __name__ == "__main__":
    args = initialize(extra_args_provider=lambda parser: None)
    model, tokenizer = initialize_model_and_tokenizer(args)

    rank = torch.distributed.get_rank()
    device_version = soc_version_map[torch_npu._C._npu_get_soc_version()]
    file = open(f"zhiputest_{device_version}_rank{rank}.csv", 'w')
    file.write(f"Batch,MaxSeqLen,InputSeqLen(Encoding),OutputSeqLen(Decoding),TokensPerSecond(ms),ResponseTime(ms),FirstTokenTime(ms),TimePerTokens(ms)\n")
    for batch_level in [1]:
        for seq_len_level in range(5, 11):
            for test_cycle_level in range(5, 11):
                seq_len = 2 ** seq_len_level
                test_cycle = 2 ** test_cycle_level
                input_param = {"seq_len": seq_len,
                               "batch": batch_level,
                               "test_cycle": test_cycle,
                               "model": model}
                first_time, avg_token = full_and_incremental_test(
                    **input_param)
                file.write(
                    f"{batch_level},2048,{seq_len},{test_cycle},{round(1000/avg_token,2)},{round(first_time+avg_token*test_cycle, 2)},{round(first_time, 2)},{round(avg_token, 2)}\n")

    file.close()

    # for seq_len in [512, 1024, 2048]:
    #     torch.distributed.barrier()
    #     start = time.time()
    #     with torch.no_grad():
    #         _, *_ = model(
    #             torch.ones(1, seq_len, device=torch.cuda.current_device(), dtype=torch.int64),
    #             torch.arange(seq_len, device=torch.cuda.current_device(), dtype=torch.int64).view(1, -1),
    #             torch.randn(1, 1, seq_len, seq_len, device=torch.cuda.current_device()) < 0.5,
    #         )
    #     torch.distributed.barrier()
    #     if torch.distributed.get_rank() == 0:
    #         print(f"Encode {seq_len}: {(time.time() - start) * 1000:.2f} ms")

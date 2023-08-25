from run_bloom_npu import full_and_incremental_test, load_model
import torch_npu
import torch
import os

soc_version_map = {-1: "unknown soc version",
    100: "910PremiumA", 101: "910ProA", 102: "910A", 103: "910ProB", 104: "910B",
    200: "310P1", 201: "310P2", 202: "310P3", 203: "310P4",
    220: "910B1", 221: "910B2", 222: "910B3", 223: "910B4",
    240: "310B1", 241: "310B2", 242: "310B3",
    250: "910C1", 251: "910C2", 252: "910C3", 253: "910C4"
}

if __name__ == "__main__":
    # change running NPU, please use "export SET_NPU_DEVICE=3"
    DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
    device_id = 0
    if DEVICE_ID is not None:
        device_id = int(DEVICE_ID)
    print(f"user npu:{device_id}")
    torch.npu.set_device(torch.device(f"npu:{device_id}"))
    
    device_version = soc_version_map[torch_npu._C._npu_get_soc_version()]

    file = open(f"zhiputest_{device_version}.csv", 'w')
    file.write(f"Batch,MaxSeqLen,InputSeqLen(Encoding),OutputSeqLen(Decoding),TokensPerSecond(ms),ResponseTime(ms),FirstTokenTime(ms),TimePerTokens(ms)\n")
    model, tokenizer = load_model()
    for batch_level in [1]:
        for seq_len_level in range(5, 11):
            for test_cycle_level in range(5, 11):
                seq_len = 2 ** seq_len_level
                test_cycle = 2 ** test_cycle_level
                input_param = {"seq_len": seq_len,
                            "batch": batch_level,
                            "test_cycle": test_cycle,
                            "model": model,
                            "tokenizer": tokenizer}
                print(f"batch: {batch_level}, seq_len: {seq_len}, test_cycle: {test_cycle}")
                first_time, avg_token = full_and_incremental_test(**input_param)
                file.write(f"{batch_level},2048,{seq_len},{test_cycle},{round(1000/avg_token,2)},{round(first_time+avg_token*test_cycle, 2)},{round(first_time, 2)},{round(avg_token, 2)}\n")

    file.close()
from rand_tensor_performance import full_and_incremental_test, load_model
import torch_npu
import torch
import os

if __name__ == "__main__":
    # change running NPU, please use "export SET_NPU_DEVICE=3"
    DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
    device_id = 0
    if DEVICE_ID is not None:
        device_id = int(DEVICE_ID)
    print(f"user npu:{device_id}")
    torch.npu.set_device(torch.device(f"npu:{device_id}"))
    
    file = open("multi_performance_test.csv", 'w')
    file.write(f"Batch,max_seq_token,input_seq_len(Encoding),output_seq_len(Decoding),TokensPerSecond(ms),ResponseTime(ms),FirstTokenTime(ms),TimePerTokens(ms)\n")
    model = load_model()
    for batch_level in [1]:
        for seq_len_level in range(5,11):
            for test_cycle_level in range(5, 11):
                seq_len = 2 ** seq_len_level
                test_cycle = 2 ** test_cycle_level
                input_param = {"seq_len": seq_len,
                            "batch": batch_level,
                            "test_cycle": test_cycle,
                            "model": model}
                print(f"batch: {batch_level}, seq_len: {seq_len}, test_cycle: {test_cycle}")
                first_time, avg_token = full_and_incremental_test(**input_param)
                file.write(f"{batch_level},2048,{seq_len},{test_cycle},{round(1000/avg_token,2)},{round(first_time+avg_token*test_cycle, 2)},{round(first_time, 2)},{round(avg_token, 2)}\n")

    file.close()

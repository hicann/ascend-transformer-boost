from rand_tensor_performance import full_and_incremental_test_with_input, load_model
import torch_npu
import torch
import os

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
CHATGLM6B_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH, "examples/chatglm6b")
RUN_SHELL_PATH = os.path.join(CHATGLM6B_PATH, "run.sh")

soc_version_map = {-1: "unknown soc version",
    100: "910PremiumA", 101: "910ProA", 102: "910A", 103: "910ProB", 104: "910B",
    200: "310P1", 201: "310P2", 202: "310P3", 203: "310P4",
    220: "910B1", 221: "910B2", 222: "910B3", 223: "910B4",
    240: "310B1", 241: "310B2", 242: "310B3",
    250: "910C1", 251: "910C2", 252: "910C3", 253: "910C4"
}

def generate_data(batch, seq_len):
    past_key_values = None
    input_ids = torch.randint(150000, (batch, seq_len)).npu()
    input_ids[:, -2] = 150001
    input_ids[:, -1] = 150004
    position_ids = torch.randint(2048, (1, 2, seq_len)).npu()
    position_ids[0][0][0] = 2047
    attention_mask = (torch.randint(4, (1, 1, seq_len, seq_len)) == torch.randint(1, (1, 1, seq_len, seq_len))).npu()
    past_key_values = ()
    for i in range(28):
        k_cache = torch.rand(seq_len, batch, 32, 128)
        v_cache = torch.rand(seq_len, batch, 32, 128)
        past_key_values = past_key_values + ((k_cache, v_cache),)
    input = {
        "input_ids":input_ids,
        "past_key_values":past_key_values,
        "position_ids":position_ids,
        "attention_mask":attention_mask
    }
    return input

if __name__ == "__main__":
    # change running NPU, please use "export SET_NPU_DEVICE=3"
    DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
    device_id = 0
    if DEVICE_ID is not None:
        device_id = int(DEVICE_ID)
    print(f"user npu:{device_id}")
    torch.npu.set_device(torch.device(f"npu:{device_id}"))
    
    device_version = soc_version_map[torch_npu._C._npu_get_soc_version()]

    input_param = {"seq_len": 20,
                    "batch": 1,
                    "test_cycle": 2}
    input_ids_path = os.path.join(CHATGLM6B_PATH, "random_input_ids.pth")
    past_key_path = []
    past_values_path = []
    for i in range(28):
        past_key_path.append(os.path.join(CHATGLM6B_PATH, f"random_past_key{i}.pth"))
    for i in range(28):
        past_values_path.append(os.path.join(CHATGLM6B_PATH, f"random_past_value{i}.pth"))
    position_ids_path = os.path.join(CHATGLM6B_PATH, "random_position_ids.pth")
    attention_mask_path = os.path.join(CHATGLM6B_PATH, "random_attention_mask.pth")
    if os.path.exists(input_ids_path):
        input_ids = torch.load(input_ids_path).npu()
        past_key_values = ()
        for i in range(28):
            k_cache = torch.load(past_key_path[i]).npu()
            v_cache = torch.load(past_values_path[i]).npu()
            past_key_values = past_key_values + ((k_cache, v_cache),)
        position_ids = torch.load(position_ids_path).npu()
        attention_mask = torch.load(attention_mask_path).npu()
        input = {
            "input_ids":input_ids,
            "past_key_values":past_key_values,
            "position_ids":position_ids,
            "attention_mask":attention_mask
        }
    else:
        input = generate_data(input_param["batch"], input_param["seq_len"])
        torch.save(input["input_ids"].cpu(), input_ids_path)
        for i in range(28):
            torch.save(input["past_key_values"][i][0].cpu(), past_key_path[i])
        for i in range(28):
            torch.save(input["past_key_values"][i][1].cpu(), past_values_path[i])
        torch.save(input["position_ids"].cpu(), position_ids_path)
        torch.save(input["attention_mask"].cpu(), attention_mask_path)
    
    model = load_model()
    first_time, avg_token = full_and_incremental_test_with_input(input, input_param["test_cycle"], model)

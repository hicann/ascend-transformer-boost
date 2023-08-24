import time
import sys
import torch
import transformers
from transformers import AutoTokenizer, AutoModel

# 适配昇腾NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
    model = AutoModel.from_pretrained("./", trust_remote_code=True).half().npu()
    
    # 量化模型适配
    for name, mod in model.named_modules():
        if type(mod).__name__ == "GLMBlock":
            if hasattr(mod, "qkv_deq_scale"):
                mod.qkv_deq_scale = torch.nn.Parameter(
                    mod.qkv_deq_scale.data.to(torch.float32))
                mod.dense_deq_scale = torch.nn.Parameter(
                    mod.dense_deq_scale.data.to(torch.float32))
                mod.hto4h_deq_scale = torch.nn.Parameter(
                    mod.hto4h_deq_scale.data.to(torch.float32))
                mod.fhtoh_deq_scale = torch.nn.Parameter(
                    mod.fhtoh_deq_scale.data.to(torch.float32))

    # 优化ND NZ排布，消除transdata
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223]:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.npu_format_cast(2)
        print("soc_version:", soc_version, " is 910B, support ND")
    else:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == "lm_head":
                    module.weight = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = module.weight.data.npu_format_cast(29)
        print("soc_version:", soc_version, " is not 910B, support NZ")

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = module.weight.data.npu_format_cast(2)
    return model


# 修改transformers的TopPLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    cumulative_probs = sorted_logits.softmax(
        dim=-1).cpu().float().cumsum(dim=-1).to(sorted_logits.device).to(sorted_logits.dtype)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
    if self.min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

transformers.generation.TopPLogitsWarper.__call__ = __call__


def warm_up(model):
    past_key_values = None
    input_ids = torch.randint(150000, (1, 4)).npu()
    input_ids[:, -2] = 150001
    input_ids[:, -1] = 150004
    position_ids = torch.randint(2048, (1, 2, 4)).npu()
    position_ids[0][0][0] = 2047
    attention_mask = (torch.randint(4, (1, 1, 4, 4)) == torch.randint(1, (1, 1, 4, 4))).npu()
    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    for _ in range(5):
        past_key_values = outputs.past_key_values
        input_ids = torch.randint(150000, (1, 1)).npu()
        position_ids = torch.randint(2048, (1, 2, 1)).npu()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids
        }
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )


# 全量
def full_test(seq_len, batch, test_cycle, model):
    print("start run.")
    warm_up(model)
    past_key_values = None
    sum_time = 0
    for i in range(test_cycle):
        input_ids = torch.randint(150000, (batch, seq_len)).npu()
        input_ids[:, -2] = 150001
        input_ids[:, -1] = 150004
        position_ids = torch.randint(2048, (1, 2, seq_len)).npu()
        position_ids[0][0][0] = 2047
        attention_mask = (torch.randint(4, (1, 1, seq_len, seq_len)) == torch.randint(1, (1, 1, seq_len, seq_len))).npu()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask
        }
        torch.npu.synchronize()
        start = time.time()
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        torch.npu.synchronize()
        end = time.time()
        cur_time = (end - start) * 1000
        sum_time += cur_time
        print(f"{cur_time}ms")
    sum_time = sum_time / test_cycle
    print(f"average = {sum_time}ms")


#全量+增量
def full_and_incremental_test(seq_len, batch, test_cycle, model):
    print("start run.")
    warm_up(model)
    past_key_values = None
    input_ids = torch.randint(150000, (batch, seq_len)).npu()
    input_ids[:, -2] = 150001
    input_ids[:, -1] = 150004
    position_ids = torch.randint(2048, (1, 2, seq_len)).npu()
    position_ids[0][0][0] = 2047
    attention_mask = (torch.randint(4, (1, 1, seq_len, seq_len)) == torch.randint(1, (1, 1, seq_len, seq_len))).npu()
    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }
    torch.npu.synchronize()
    start = time.time()
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    torch.npu.synchronize()
    end = time.time()
    first_time = (end - start) * 1000
    print(f"first token: {first_time}ms")
    sum_time = 0
    test_cycle -= 1
    for i in range(test_cycle):
        past_key_values = outputs.past_key_values
        input_ids = torch.randint(150000, (batch, 1)).npu()
        position_ids = torch.randint(2048, (1, 2, 1)).npu()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids
        }
        torch.npu.synchronize()
        start = time.time()
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        torch.npu.synchronize()
        end = time.time()
        cur_time = (end - start) * 1000
        sum_time += cur_time
        # print(f"token_{i + 1}: {cur_time}ms")
    avg_time = sum_time / test_cycle
    print(f"average token: {sum_time / test_cycle}ms")
    print(f"response time: {first_time + sum_time}ms")
    return first_time, avg_time

def full_and_incremental_test_with_input(input, test_cycle, model):
    print("start run.")
    warm_up(model)
    
    model_inputs = {
        "input_ids": input["input_ids"],
        "past_key_values": None,
        "position_ids": input["position_ids"],
        "attention_mask": input["attention_mask"]
    }
    torch.npu.synchronize()
    start = time.time()
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    torch.npu.synchronize()
    end = time.time()
    first_time = (end - start) * 1000
    print(f"first token: {first_time}ms")
    sum_time = 0
    test_cycle -= 1
    avg_time = 0
    model.count = 0
    for i in range(test_cycle):
        model_inputs = {
            "input_ids": input["input_ids"],
            "past_key_values": input["past_key_values"],
            "position_ids": input["position_ids"]
        }
        model.count = model.count + 1
        torch.npu.synchronize()
        start = time.time()
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        torch.npu.synchronize()
        end = time.time()
        cur_time = (end - start) * 1000
        sum_time += cur_time
    if test_cycle != 0:
        avg_time = sum_time / test_cycle
    print(f"average token: {avg_time}ms")
    print(f"response time: {first_time + sum_time}ms")
    return first_time, avg_time

test_funcs = {"full": full_test, "full_and_incremental": full_and_incremental_test}
arg_map = {"seq_len": 512, "batch": 8, "test_cycle": 100, "device_id": 0}

if __name__ == "__main__":
    argv_len = len(sys.argv)
    argv_index = 1
    func_name = "full_and_incremental"
    if argv_index < argv_len and sys.argv[argv_index] in test_funcs:
        func_name = sys.argv[argv_index]
        argv_index += 1
    while argv_index < argv_len:
        (arg_name, arg_value) = sys.argv[argv_index].split('=')
        argv_index += 1
        if arg_name not in arg_map:
            print(f"arg_name: {arg_name} not in arg_map.")
            continue
        arg_value = int(arg_value)
        arg_map[arg_name] = arg_value
    print("arg_map: " + str(arg_map))
    if arg_map["test_cycle"] == 0 or arg_map["batch"] == 0 or arg_map["seq_len"] == 0:
        print("test_cycle & batch & seq_len can't equal 0!")
        exit()
    
    device_id = arg_map["device_id"]
    torch.npu.set_device(torch.device(f"npu:{device_id}"))
    # 使用二进制优化，消除动态shape的编译问题
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
    torch.npu.set_option(option)
    
    model = load_model()
    input_param = {"seq_len": arg_map["seq_len"],
                   "batch": arg_map["batch"],
                   "test_cycle": arg_map["test_cycle"],
                   "model": model}
    test_funcs[func_name](**input_param)

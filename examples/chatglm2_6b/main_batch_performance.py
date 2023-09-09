import argparse
import torch
import os
import torch_npu
from transformers import AutoTokenizer, AutoModel


soc_version_map = {-1: "unknown soc version",
    100: "910PremiumA", 101: "910ProA", 102: "910A", 103: "910ProB", 104: "910B",
    200: "310P1", 201: "310P2", 202: "310P3", 203: "310P4",
    220: "910B1", 221: "910B2", 222: "910B3", 223: "910B4",
    240: "310B1", 241: "310B2", 242: "310B3",
    250: "910C1", 251: "910C2", 252: "910C3", 253: "910C4"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="multi-batch performance")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Whether test model in parallel",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Whether test model in parallel",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./",
        help="the path to model weights",
    )
    args = parser.parse_args()
    return args


def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if local_rank==0:
        torch_npu.npu.set_device(0)
    elif local_rank==1:
        torch_npu.npu.set_device(1)
    torch.manual_seed(1)
    return local_rank, world_size


def load_model(model_path, parallel=False):
    # 加载权重
    if parallel:
        local_rank, world_size = setup_model_parallel()
        tokenizer_path = os.path.join(model_path, "tokenizer")
        part_model_path = os.path.join(model_path, "part_model", str(local_rank))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(part_model_path, trust_remote_code=True).half().npu()
    else:
        # change running NPU, please use "export SET_NPU_DEVICE=3"
        DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
        device_id = 0
        if DEVICE_ID is not None:
            device_id = int(DEVICE_ID)
        print(f"user npu:{device_id}")
        torch.npu.set_device(torch.device(f"npu:{device_id}"))
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().npu()
    
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

    # 优化ND/NZ排布，消除transdata
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

    return model, tokenizer


if __name__ == "__main__":

    opts = parse_args()
    model, tokenizer = load_model(opts.model_path, parallel=opts.parallel)

    prompts = [
        "中国的首都在哪里",
        "请做一首诗歌",
        "如何学习python？",
        "五条直线相交有几个交点？",
        "中国的首都在哪里",
        "请做一首诗歌",
        "如何学习python？",
        "五条直线相交有几个交点？",
        "中国的首都在哪里",
        "请做一首诗歌",
        "如何学习python？",
        "五条直线相交有几个交点？",
        "中国的首都在哪里",
        "请做一首诗歌",
        "如何学习python？",
        "五条直线相交有几个交点？",
    ]

    for batch in [1]:
        for seq_len_in_level in range(5,11):
            for seq_len_out_level in range(5, 11):
                seq_len_in = 2 ** seq_len_in_level
                seq_len_out = 2 ** seq_len_out_level

                print(f"batch: {batch}, seq_len: {seq_len_in}, test_cycle: {seq_len_out}")

                # warm up
                print("=========== warm up start ==========")
                test_prompts = prompts[:1]
                inputs = tokenizer(test_prompts, return_tensors="pt", padding="max_length", max_length=seq_len_in)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids.npu(),
                        attention_mask=inputs.attention_mask.npu(),
                        max_new_tokens=seq_len_out,
                    )
                print("========== warm up end ==========")
                
                # test
                print("=========== performance test start ==========")
                test_prompts = prompts[:batch]
                inputs = tokenizer(test_prompts, return_tensors="pt", padding="max_length", max_length=seq_len_in)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids.npu(),
                        attention_mask=inputs.attention_mask.npu(),
                        max_new_tokens=seq_len_out,
                    )
                print("=========== performance test end ==========")


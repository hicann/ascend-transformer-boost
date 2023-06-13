# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# from fastchat.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
#
# replace_llama_attn_with_flash_attn()

import os
import torch
import torch_npu
import deepspeed_npu
import deepspeed
from torch_npu.contrib import transfer_to_npu

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "TRUE"

from fastchat.train.train import train

if __name__ == "__main__":

    torch.npu.set_compile_mode(jit_compile=False)
    deepspeed.init_distributed('hccl')

    train()

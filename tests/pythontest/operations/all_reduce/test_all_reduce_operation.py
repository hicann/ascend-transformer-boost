import os
import json
import torch
import torch_npu
import torch.distributed as dist

# usage:
# export HCCL_WHITELIST_DISABLE=1
# torchrun --standalone --nnodes=1 --nproc_per_node=8 test_all_reduce_operation.py
# Attention: when you use lccl backend, unset HCCL_MTE_ENABLE and copy lcal.o to current directory

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "lib/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "22345"
dist.init_process_group(backend="hccl", rank=local_rank, world_size=world_size)
torch_npu.npu.set_device(local_rank)
print('dist inited ok...')

acl_allreduce_operation = torch.classes.OperationTorch.OperationTorch(
    "AllReduceOperation")
acl_param = json.dumps({"rank": local_rank, "rankSize": world_size,
                       "rankRoot": 0, "allReduceType": "sum", "backend": "lccl"})
acl_allreduce_operation.set_param(acl_param)

oneTensor = torch.zeros(
    [3, 4, 5], device=torch.npu.current_device(), dtype=torch.half)
oneTensor.add_(0.001)
outTensors = acl_allreduce_operation.execute([oneTensor])
print('acl all reduce output is ' + str(outTensors[0]))

import os
import json
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest

# usage:
# build with option: --use_hccl_runner / --use_lccl_runner
# export HCCL_WHITELIST_DISABLE=1
# python3 -m unittest test_all_gather_operation.py
# Attention: when you use lccl backend, unset HCCL_MTE_ENABLE and copy lcal.o to current directory

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "lib/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


def main_worker(rank, world_size):
    # init process group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12344"
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    # init all gather operation
    acl_allgather_operation = torch.classes.OperationTorch.OperationTorch(
        "AllGatherOperation")
    acl_param = json.dumps({"rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "backend": "hccl"})
    acl_allgather_operation.set_param(acl_param)

    # exec all gather
    inTensor = torch.ones(
        [3, 4, 5], device=torch.npu.current_device(), dtype=torch.half)
    tensorList = []
    for i in range(world_size):
        tensorList.append(inTensor)
    golden_out_tensor = torch.stack(tensorList, dim=0)
    acl_out_tensor = acl_allgather_operation.execute([inTensor])[0]

    # assert result
    assert golden_compare(acl_out_tensor, golden_out_tensor)


def golden_compare(out_tensor, golden_out_tensor):
    print("out_tensor.shape", out_tensor.shape,
          "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
    print("out_tensor:", out_tensor,
          ", \ngolden_oute_tensor:", golden_out_tensor)
    return torch.allclose(out_tensor, golden_out_tensor, rtol=0.02, atol=0.02)


class AllGatherOperationTest(unittest.TestCase):
    def test_all_gather(self):
        world_size = 2
        mp.spawn(main_worker, nprocs=world_size, args=(world_size,))


if __name__ == '__main__':
    unittest.main()

import os
import json
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

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


class AllGatherOperationTest(operation_test.OperationTest):
    def test_all_gather(self):
        command = f"nm -D {ACLTRANSFORMER_HOME_PATH}/lib/libacltransformer.so | grep HcclAllGather > /dev/null"
        res = os.system(command)
        if res == 0:
            world_size = 2
            mp.spawn(self.main_worker, nprocs=world_size, args=(world_size,))
        else:
            print("hccl_runner is not compiled, skip AllGatherOperationTest")
    
    def main_worker(self, rank, world_size):
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
        assert self.golden_compare(acl_out_tensor, golden_out_tensor)


if __name__ == '__main__':
    unittest.main()

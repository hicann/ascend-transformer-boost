import os
import sys
import json
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest

# usage:
# build with option: --use_hccl_runner / --use_lccl_runner
# export HCCL_WHITELIST_DISABLE=1
# python3 -m unittest test_linear_parallel_operation.py
# Attention: when you use lccl backend, unset HCCL_MTE_ENABLE and copy lcal.o to current directory

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "lib/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)

TOOLS_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH, "tools", "python_tools")

sys.path.append(TOOLS_PATH)

import tensor_file  # NOQA:E402

IN_TENSOR1_DIR = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                              "/data/acltransformer_testdata/tensors/operations/linear_parallel/",
                              "inTensor0.bin")
IN_TENSOR2_DIR = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                              "/data/acltransformer_testdata/tensors/operations/linear_parallel/",
                              "inTensor1.bin")
IN_TENSOR3_DIR = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                              "/data/acltransformer_testdata/tensors/operations/linear_parallel/",
                              "inTensor2.bin")
OUT_TENSOR1_DIR = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                               "/data/acltransformer_testdata/tensors/operations/linear_parallel/",
                               "outTensor0.bin")


def main_worker(rank, world_size):
    # init process group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12344"
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    # init linear parallel operation
    linear_parallel_operation = torch.classes.OperationTorch.OperationTorch(
        "LinearParallelOperation")
    acl_param = json.dumps({"transWeight": False, "rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "bias": "", "parallelType": "RowParallel", "backend": "hccl"})
    linear_parallel_operation.set_param(acl_param)

    # exec linear parallel operation
    inTensor1 = tensor_file.read_tensor(IN_TENSOR1_DIR).npu()
    inTensor2 = tensor_file.read_tensor(IN_TENSOR2_DIR).npu()
    inTensor3 = tensor_file.read_tensor(IN_TENSOR3_DIR).npu()
    golden_out_tensor = tensor_file.read_tensor(OUT_TENSOR1_DIR).npu()
    acl_out_tensor = linear_parallel_operation.execute(
        [inTensor1, inTensor2, inTensor3])[0]

    # assert result
    assert golden_compare(acl_out_tensor, golden_out_tensor)


def golden_compare(out_tensor, golden_out_tensor):
    return torch.allclose(out_tensor, golden_out_tensor, rtol=0.02, atol=0.02)


class LinearParallelOperationTest(unittest.TestCase):
    def test_linear_parallel_operation(self):
        command = f"nm -D {ACLTRANSFORMER_HOME_PATH}/lib/libacltransformer.so | grep HcclAllReduce > /dev/null"
        res = os.system(command)
        if res == 0:
            world_size = 2
            mp.spawn(main_worker, nprocs=world_size, args=(world_size,))
        else:
            print("hccl_runner is not compiled, skip LinearParallelOperationTest")


if __name__ == '__main__':
    unittest.main()

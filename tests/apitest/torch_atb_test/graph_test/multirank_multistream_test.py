import torch_atb
import torch
import acl
import torch_npu
import unittest
import sys
import os
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import ret_check

rank_size = 2

def create_streams():
    streams = []
    for i in range(2):
        stream, ret = acl.rt.create_stream()
        ret_check(ret)
        # print(stream)
        streams.append(stream)
    return streams

def graph_build():
    print("----------- graph build begin ------------")
    builder = torch_atb.Builder("Graph")
    builder.set_execute_streams(create_streams())
    x = builder.add_input("x")
    y = builder.add_input("y")

    elewise_add = torch_atb.ElewiseParam()
    elewise_add.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD

    add1 = builder.add_node([x, y], elewise_add)
    add1_out = add1.get_output(0)

    add2 = builder.add_node([x, y], elewise_add)
    add2.set_stream_id(1)

    add2_out = add2.get_output(0)
    z = builder.add_input("z")

    builder.reshape(add2_out, lambda shape: [1, shape[0] * shape[1]], "add_out_")

    elewise_mul = torch_atb.ElewiseParam()
    elewise_mul.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_MUL

    mul = builder.add_node(["add_out_", z], elewise_mul)
    mul.set_stream_id(1)

    builder.mark_output(mul.get_output(0))
    Graph = builder.build()
    print("----------- graph build success ------------")

    print(Graph)

    print("----------- graph forward begin ------------")
    x = torch.ones(2, 3, dtype=torch.float16)
    y = torch.ones(2, 3, dtype=torch.float16)
    z = torch.ones(1, 6, dtype=torch.float16)
    
    tensors_npu = [tensor.npu() for tensor in [x, y, z]]

    def graph_run():
        res = Graph.forward(tensors_npu)
        # print("add1 streamid: ", add1.get_stream_id())
        # print("add2 streamid: ", add2.get_stream_id())
        # print("mul streamid: ", mul.get_stream_id())
        assert add1.get_stream_id() == 0
        assert add2.get_stream_id() == 1
        assert mul.get_stream_id() == 1
        return res

    npu_outputs = graph_run()
    print("npu_outputs: ", npu_outputs)
    print("----------- graph forward success ------------")

def worker(rank):
    torch_npu.npu.set_device(rank)
    graph_build()

class TestMultiRankMultiStream(unittest.TestCase):
    def test(self):
        mp.spawn(worker, nprocs=rank_size, join=True)
        print("------------all sub processes finish------------")

if __name__ == "__main__":
    unittest.main()

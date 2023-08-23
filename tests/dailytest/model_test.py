import os
import sys
sys.path.append('../..')
import torch
import torch_npu
from tools.python_tools.tensor_file import read_tensor

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

soc_version_map = {-1: "unknown soc version",
                   100: "910PremiumA", 101: "910ProA", 102: "910A", 103: "910ProB", 104: "910B",
                   200: "310P1", 201: "310P2", 202: "310P3", 203: "310P4",
                   220: "910B1", 221: "910B2", 222: "910B3", 223: "910B4",
                   240: "310B1", 241: "310B2", 242: "310B3",
                   250: "910C1", 251: "910C2", 252: "910C3", 253: "910C4"
                   }
time_head = ["ModelName", "Batch", "MaxSeqLen", "InputSeqLen(Encoding)", "OutputSeqLen(Decoding)",
             "TokensPerSecond(ms)", "ResponseTime(ms)", "FirstTokenTime(ms)", "TimePerTokens(ms)"]
precision_head = ["ModelName", "AbsoluteError", "AverageCosineSimilarity", "MaxRelativeError"]


class Statistics:
    def __init__(self):
        self.batch = ""
        self.max_seq_len = ""
        self.input_seq_len = ""
        self.output_seq_len = ""
        self.tokens_per_second = ""
        self.response_time = ""
        self.first_token_time = ""
        self.time_per_tokens = ""


class ModelTest:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.time_result_path = self.script_path + "/results/times"
        self.precision_result_path = self.script_path + "/results/precisions"
        self.device_type = self.get_device_type()
        if not os.path.exists(self.time_result_path):
            os.makedirs(self.time_result_path)
        if not os.path.exists(self.precision_result_path):
            os.makedirs(self.precision_result_path)
        self.time_file_name = self.device_type + "_performance.csv"
        self.precision_file_name = self.device_type + "_precision.csv"
        self.precision_golden = []
        self.precision_result = []

    def get_device_type(self):
        DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
        device_id = 0
        if DEVICE_ID is not None:
            device_id = int(DEVICE_ID)
        print(f"user npu:{device_id}")
        torch.npu.set_device(torch.device(f"npu:{device_id}"))
        print("Device Set Success!")

        soc_version = torch_npu._C._npu_get_soc_version()
        return soc_version_map[soc_version]

    def create_time(self):
        os.chdir(self.time_result_path)
        file_handle = open(self.time_file_name, 'a')
        for i in range(len(time_head) - 1):
            file_handle.write(time_head[i] + ",")
        file_handle.write(
            time_head[len(time_head) - 1] + "\n")
        file_handle.close()

    def append_time(self, statistic):
        os.chdir(self.time_result_path)
        file_handle = open(self.time_file_name, 'a')
        file_handle.write(f"{self.model_name},{statistic.batch},{statistic.max_seq_len},{statistic.input_seq_len}," +
                          f"{statistic.output_seq_len},{statistic.tokens_per_second},{statistic.response_time}," +
                          f"{statistic.first_token_time},{statistic.time_per_tokens}\n")
        file_handle.close()
    
    def create_precision(self):
        os.chdir(self.precision_result_path)
        file_handle = open(self.precision_file_name, 'a')
        for i in range(len(precision_head) - 1):
            file_handle.write(precision_head[i] + ",")
        file_handle.write(precision_head[len(precision_head) - 1] + "\n")
        file_handle.close()

    def append_precision_golden(self, tensor_path):
        self.precision_golden.append(tensor_path)

    def append_precision_result(self, tensor_path):
        self.precision_result.append(tensor_path)
    
    def precision_compare(self):
        if len(self.precision_result) != len(self.precision_golden):
            print("result size not equal golden size!!!")
            return
        for i in range(len(self.precision_result)):
            compare_result = self.tensor_compare(self.precision_result[i], self.precision_golden[i])
            os.chdir(self.precision_result_path)
            file_handle = open(self.precision_file_name, 'a')
            file_handle.write(f"{self.model_name},{compare_result[0]},{compare_result[1]},{compare_result[2]}\n")
            file_handle.close()

    def tensor_compare(self, tensor1_path, tensor2_path):
        tensor1 = read_tensor(tensor1_path)
        tensor2 = read_tensor(tensor2_path)

        tensor1 = tensor1.to(torch.float64)
        tensor2 = tensor2.to(torch.float64)

        sub_tensor = tensor1 - tensor2
        abs_tensor = sub_tensor.abs()

        absolute_err = 0
        avg_cosine_similarity = 0
        max_relative_err = 0
        
        if abs_tensor.numel() != 0:
            absolute_err = abs_tensor.type(torch.float64).sum() / abs_tensor.numel()
            cosine_similarity_tensor = torch.cosine_similarity(tensor1, tensor2, dim=0)
            avg_cosine_similarity = cosine_similarity_tensor.abs().sum()/cosine_similarity_tensor.numel()
            div_tensor = tensor2.abs()
            div_tensor.clamp_(1e-6)
            relative_err_tensor = torch.div(abs_tensor, div_tensor)
            max_relative_err = torch.max(relative_err_tensor)
        return [absolute_err, avg_cosine_similarity, max_relative_err]

if __name__ == "__main__":
    modelTest = ModelTest("test")
    modelTest.create_time()
    modelTest.create_precision()
    
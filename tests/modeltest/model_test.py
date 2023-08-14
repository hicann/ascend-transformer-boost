import os
import torch
import torch_npu

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

class Statistics:
    def __init__(self):
        self.time_head = ["Batch", "MaxSeqLen", "InputSeqLen(Encoding)", "OutputSeqLen(Decoding)",
                          "TokensPerSecond(ms)", "ResponseTime(ms)", "FirstTokenTime(ms)", "TimePerTokens(ms)"]
        self.model_name = ""
        self.batch = ""
        self.max_seq_len = ""
        self.input_seq_len = ""
        self.output_seq_len = ""
        self.tokens_per_second = ""
        self.response_time = ""
        self.first_token_time = ""
        self.time_per_tokens = ""


class ModelTest:
    def __init__(self) -> None:
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.time_result_path = self.script_path + "/results/times"
        self.device_type = self.get_device_type()
        if not os.path.exists(self.time_result_path):
            os.makedirs(self.time_result_path)
    
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

    def create_time(self, statistic):
        os.chdir(self.time_result_path)
        file_name = statistic.model_name + "_" + self.device_type + "_performance.csv"
        file_handle = open(file_name, 'a')
        for i in range(len(statistic.time_head) - 1):
            file_handle.write(statistic.time_head[i] + ",")
        file_handle.write(statistic.time_head[len(statistic.time_head) - 1] + "\n")

    def append_time(self, statistic):
        os.chdir(self.time_result_path)
        file_name = statistic.model_name + "_" + self.device_type + "_performance.csv"
        file_handle = open(file_name, 'a')
        file_handle.write(f"{statistic.batch},{statistic.max_seq_len},{statistic.input_seq_len}," + \
                          f"{statistic.output_seq_len},{statistic.tokens_per_second},{statistic.response_time}," + \
                          f"{statistic.first_token_time},{statistic.time_per_tokens}\n")


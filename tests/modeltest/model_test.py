import os
import torch
import torch_npu

class Statistics:
    def __init__(self):
        self.model_name = ""
        self.time_dic = {"Batch": 1, "max_seq_token": 1,
                         "input_seq_len(Encoding)": 1, "output_seq_len(Decoding)": 1,
                         "TokensPerSecond(ms)": 1, "ResponseTime(ms)": 1,
                         "FirstTokenTime(ms)": 1, "TimePerTokens(ms)": 1}
        self.time_head = ["Batch", "max_seq_token", "input_seq_len(Encoding)", "output_seq_len(Decoding)",
                          "TokensPerSecond(ms)", "ResponseTime(ms)", "FirstTokenTime(ms)", "TimePerTokens(ms)"]
        self.device_type = "910b"

class ModelTest:
    def __init__(self) -> None:
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.time_result_path = self.script_path + "/results/times"
        self.get_device_type()
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
        if soc_version in [104, 220, 221, 222, 223]:
            self.device_type = "910b"
        else:
            self.device_type = "310p"

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
        for i in range(len(statistic.time_head) - 1):
            file_handle.write(str(statistic.time_dic[statistic.time_head[i]]) + ",")
        file_handle.write(str(statistic.time_dic[statistic.time_head[-1]]) + "\n")

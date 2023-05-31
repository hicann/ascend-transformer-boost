import torch
import os


class TensorTestCase:
    def __init__(self, test_name, in_tensor_num=1, out_tensor_num=1) -> None:
        self.test_name = test_name
        self.in_tensor_num = in_tensor_num
        self.out_tensor_num = out_tensor_num
        self.in_tensor_list = []
        self.out_tensor_list = []
        self.acl_path = os.getenv("ACLTRANSFORMER_TEST_DATA")
        if not self.acl_path:
            print("env ACLTRANSFORMER_TEST_DATA not exist!")
        # self.acl_path = "/home/raid_data/w00821845/acltransformer"

    def set_in_tensors(self, tensor_list):
        self.in_tensor_list = tensor_list

    def set_out_tensors(self, tensor_list):
        self.out_tensor_list = tensor_list

    def get_in_tensors(self):
        return self.in_tensor_list

    def get_out_tensors(self):
        return self.out_tensor_list

    def read(self, test_idx):
        self.in_tensor_list = []
        self.out_tensor_list = []
        dir_path = os.path.join(
            self.acl_path, 'testcases', self.test_name, str(test_idx))
        for i in range(1, self.in_tensor_num + 1):
            self.in_tensor_list.append(torch.load(
                os.path.join(dir_path, f"inTensor{i}.pth")))
        for i in range(1, self.out_tensor_num + 1):
            self.out_tensor_list.append(torch.load(
                os.path.join(dir_path, f"outTensor{i}.pth")))

    def write(self, test_idx):
        dir_path = os.path.join('testcases', self.test_name, str(test_idx))
        print(f"write {self.test_name} testcase{test_idx}...")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        i = 1
        for in_tensor in self.in_tensor_list:
            torch.save(in_tensor, os.path.join(dir_path, f"inTensor{i}.pth"))
            i += 1
        i = 1
        for out_tensor in self.out_tensor_list:
            torch.save(out_tensor, os.path.join(dir_path, f"outTensor{i}.pth"))
            i += 1

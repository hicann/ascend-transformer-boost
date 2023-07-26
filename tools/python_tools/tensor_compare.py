import sys
import os
import torch
sys.path.append(os.path.dirname(__file__))
from tensor_file import read_tensor  # NOQA: E402


def main():
    tensor1 = read_tensor(sys.argv[1])
    tensor2 = read_tensor(sys.argv[2])

    print("tensor1:" + str(tensor1))
    print("tensor2:" + str(tensor2))
    print("tensor1.shape", tensor1.shape, ", dtype:", tensor1.dtype)
    print("tensor2.shape", tensor2.shape, ", dtype:", tensor2.dtype)

    sub_tensor = tensor1 - tensor2
    abs_tensor = sub_tensor.abs()

    absolute_err = abs_tensor.type(torch.float64).sum() / abs_tensor.numel()
    relative_err = torch.div(abs_tensor, tensor2.abs()).type(torch.float64).sum() / abs_tensor.numel()
    
    print("Absolute error: ")
    print(absolute_err)
    print("Relative error:")
    print(relative_err)



if __name__ == "__main__":
    main()

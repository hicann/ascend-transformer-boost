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

    if torch.allclose(tensor1, tensor2, rtol=0.02, atol=0.02):
        print("equal")
    else:
        print("not equal")


if __name__ == "__main__":
    main()

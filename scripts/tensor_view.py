import sys
import os
import torch
sys.path.append(os.path.dirname(__file__))
from tensor_file import read_tensor  # NOQA: E402


def main():
    tensor = read_tensor(sys.argv[1])
    print("tensor:" + str(tensor))
    print("tensor.shape", tensor.shape, ", dtype:", tensor.dtype)


if __name__ == "__main__":
    main()

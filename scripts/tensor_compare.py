import sys
import torch


def main():
    tensor1 = list(torch.load(sys.argv[1]).state_dict().values())[0]
    tensor2 = list(torch.load(sys.argv[2]).state_dict().values())[0]
    print(tensor1.shape)
    print(tensor2.shape)
    print("tensor1:" + str(tensor1))
    print("tensor2:" + str(tensor2))
    assert torch.allclose(tensor1, tensor2, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    main()

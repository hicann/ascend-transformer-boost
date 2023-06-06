import sys
import torch


def main():
    try:
        tensor1 = list(torch.load(sys.argv[1]).state_dict().values())[0]
    except:
        tensor1 = torch.load(sys.argv[1])

    try:
        tensor2 = list(torch.load(sys.argv[2]).state_dict().values())[0]
    except:
        tensor2 = torch.load(sys.argv[2])

    print("tensor1:" + str(tensor1))
    print("tensor2:" + str(tensor2))
    print(tensor1.shape)
    print(tensor2.shape)
    assert torch.allclose(tensor1, tensor2, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    main()

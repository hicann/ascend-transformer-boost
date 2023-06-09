import sys
import torch


def main():
    try:
        tensor1 = list(torch.load(sys.argv[1]).state_dict().values())[0]
    except:
        tensor1 = torch.load(sys.argv[1])

    print("tensor1:" + str(tensor1))


if __name__ == "__main__":
    main()

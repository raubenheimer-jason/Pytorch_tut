# https://www.youtube.com/watch?v=BzcBsTou0C0&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=1&ab_channel=sentdex

import torch
# print(torch.cuda.is_available())


def main():
    x = torch.Tensor([5, 3])
    y = torch.Tensor([2, 1])
    # print(x*y)

    x = torch.zeros([2, 5])
    # print(x)

    # print(x.shape)

    y = torch.rand([2, 5])
    # print(y)

    print(y.view([1, 10]))
    print(y)


if __name__ == "__main__":
    main()

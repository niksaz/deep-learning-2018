import destiny_nets as nets
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import random
import os.path


def set_random_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    torch.backends.cudnn.deterministic = True


def create_root_for_data():
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    return root


def compute_train_mean_std(root):
    # TODO: Remove!
    mean = 0.2860410809516907
    std = 0.3530243933200836
    return mean, std

    train_set = datasets.FashionMNIST(root=root, train=True,
                                      transform=transforms.ToTensor(),
                                      download=True)

    pixel_sum = torch.zeros(1)
    pixel_num = 0
    for image, _ in train_set:
        pixel_sum += image.sum()
        pixel_num += image.numel()
    mean = pixel_sum / pixel_num

    dev_from_mean = torch.zeros(1)
    for image, _ in train_set:
        dev_from_mean += ((image - mean) ** 2).sum()
    std = torch.sqrt(dev_from_mean / pixel_num)

    return mean.item(), std.item()


def load_datasets(root, transform):
    train_set = datasets.FashionMNIST(root=root, train=True,
                                      transform=transform, download=True)
    test_set = datasets.FashionMNIST(root=root, train=False,
                                     transform=transform, download=True)
    return train_set, test_set


def main():
    set_random_seed()

    root = create_root_for_data()
    print('Data root is', root)
    print()

    mean, std = compute_train_mean_std(root)
    print('Mean is =', mean)
    print('Std is =', std)
    print()

    train_set, test_set = load_datasets(
        root,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    )
    print('train_set len is', len(train_set))
    print('test_set len is', len(test_set))
    print()


if __name__ == "__main__":
    main()

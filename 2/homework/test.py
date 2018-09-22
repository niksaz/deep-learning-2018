import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import random
import os.path

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from destinynets.resnext import resnext18


def set_random_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def create_root_for_data():
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    return root


def load_datasets(root, transform):
    train_set = datasets.STL10(root=root, split='train', transform=transform,
                               download=True)
    test_set = datasets.STL10(root=root, split='test', transform=transform,
                              download=True)

    return train_set, test_set


class SequentialPrefixSampler(Sampler):

    def __init__(self, data_source, prefix):
        super(SequentialPrefixSampler, self).__init__(data_source)
        self.prefix = min(prefix, len(data_source))

    def __iter__(self):
        return iter(range(self.prefix))

    def __len__(self):
        return self.prefix


def dataset_to_loader(dataset, prefix, batch_size=10):
    prefix_sampler = SequentialPrefixSampler(dataset, prefix=prefix)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=prefix_sampler)
    return dataloader


def train_and_evaluate(train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    model = resnext18()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()

        for images, labels in train_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        correct = 0
        for images, labels in test_dataloader:
            outputs = model(images)

            _, predicted_labels = torch.max(outputs.data, dim=1)
            correct += (predicted_labels == labels).sum()

        accuracy = 100 * correct / len(test_dataloader.sampler)

        print('After epoch', epoch)
        print('Loss =', loss.data.item())
        print('Accuracy =', accuracy.item())
        print()


def main():
    set_random_seed()

    root = create_root_for_data()
    print('Data root is', root)
    print()

    train_set, test_set = load_datasets(
        root,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    train_dataloader = dataset_to_loader(train_set, prefix=1000)
    test_dataloader = dataset_to_loader(test_set, prefix=500)

    print('len(train_dataloader) is', len(train_dataloader.sampler))
    print('len(test_dataloader) is', len(test_dataloader.sampler))
    print()

    train_and_evaluate(train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()

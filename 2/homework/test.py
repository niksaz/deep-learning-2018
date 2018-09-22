""" To run the test:
        $ python -m visdom.server -port 8097 &
        $ python test.py
"""

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import numpy as np
import torch
import random
import os.path

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from destinynets.resnext import resnext50


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


class RandomPrefixSampler(Sampler):

    def __init__(self, data_source, prefix):
        super(RandomPrefixSampler, self).__init__(data_source)
        self.prefix = min(prefix, len(data_source))

    def __iter__(self):
        return iter(random.sample(range(self.prefix), self.prefix))

    def __len__(self):
        return self.prefix


def dataset_to_loader(dataset, prefix, batch_size=10):
    prefix_sampler = RandomPrefixSampler(dataset, prefix=prefix)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=prefix_sampler)
    return dataloader


def train_and_track(train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()

    model = resnext50()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    engine = Engine()

    port = 8097
    train_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train CrossEntropyLoss'})
    train_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Class Accuracy'})
    test_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test CrossEntropyLoss'})
    test_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Class Accuracy'})

    meter_loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter(accuracy=True)

    def run_model(sample):
        images, labels = sample
        outputs = model(images)
        loss = criterion(outputs, labels)
        return loss, outputs

    def reset_meters():
        classerr.reset()
        meter_loss.reset()

    def on_forward(state):
        classerr.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data.item())

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_err_logger.log(state['epoch'], classerr.value()[0])

        # Check accuracy on test after each epoch.
        reset_meters()
        engine.test(run_model, test_dataloader)
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_err_logger.log(state['epoch'], classerr.value()[0])

    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(run_model, train_dataloader, maxepoch=10, optimizer=optimizer)


def main():
    set_random_seed()

    root = create_root_for_data()

    train_set, test_set = load_datasets(
        root,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    train_dataloader = dataset_to_loader(train_set, prefix=500)
    test_dataloader = dataset_to_loader(test_set, prefix=500)

    train_and_track(train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()

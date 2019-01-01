import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from homework.vae import VAE, VAETrainer, loss_function


def build_loader(data_root, transform, batch_size, train):
    dataset = datasets.CIFAR10(root=data_root, download=True,
                               transform=transform, train=train)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True)
    return dataloader


def get_config():
    parser = argparse.ArgumentParser(description='Training VAE on CIFAR10')

    parser.add_argument('--log-root', type=str, default='logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)

    return parser.parse_args()


def main():
    device = 'cpu'
    config = get_config()

    os.makedirs(config.log_root, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root, config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose(
        [transforms.Resize(28), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = build_loader(
        config.data_root, transform, config.batch_size, train=True)
    test_loader = build_loader(
        config.data_root, transform, config.batch_size, train=False)

    model = VAE().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    trainer = VAETrainer(model=model,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         optimizer=optimizer,
                         loss_function=loss_function,
                         device=device)

    for epoch in range(0, config.epochs):
        trainer.train(epoch, log_interval=config.log_metrics_every)
        trainer.test(
            epoch,
            log_interval=config.log_metrics_every,
            show_img_every=config.show_img_every)


if __name__ == '__main__':
    main()

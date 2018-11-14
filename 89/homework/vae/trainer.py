import logging
import os

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class VAETrainer:

    def __init__(self, model, train_loader, test_loader, optimizer,
                 loss_function, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.writer = SummaryWriter()

    def train(self, epoch, log_interval):
        self.model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            self.model.zero_grad()

            data = data.to(self.device)
            recon_data, mu, logvar = self.model(data)

            train_loss = self.loss_function(recon_data, data, mu, logvar)
            train_loss.backward()

            self.optimizer.step()

            epoch_loss += train_loss
            norm_train_loss = train_loss / len(data)

            if batch_idx % log_interval == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    norm_train_loss)
                logging.info(msg)

                batches_per_epoch_train = len(self.train_loader.dataset) // self.train_loader.batch_size
                self.writer.add_scalar(tag='data/train_loss',
                                       scalar_value=norm_train_loss,
                                       global_step=batches_per_epoch_train * epoch + batch_idx)

        epoch_loss /= len(self.train_loader.dataset)
        logging.info(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar(tag='data/train_epoch_loss',
                               scalar_value=epoch_loss,
                               global_step=epoch)

    def test(self, epoch, log_interval, show_img_every):
        self.model.eval()
        test_epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):
            data = data.to(self.device)
            recon_data, mu, logvar = self.model(data)
            test_loss = self.loss_function(recon_data, data, mu, logvar)

            test_epoch_loss += test_loss

            batches_per_epoch_test = len(self.test_loader.dataset) // self.test_loader.batch_size
            global_batch_idx = batches_per_epoch_test * epoch + batch_idx
            if batch_idx % log_interval == 0:
                msg = 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.test_loader.dataset),
                    100. * batch_idx / len(self.test_loader),
                    test_loss / len(data))
                logging.info(msg)

                self.writer.add_scalar(tag='data/test_loss',
                                       scalar_value=test_loss / len(data),
                                       global_step=global_batch_idx)

            if batch_idx % show_img_every == 0:
                self._plot_images(data[:10], recon_data[:10], global_batch_idx)

        test_epoch_loss /= len(self.test_loader.dataset)
        logging.info('====> Test set loss: {:.4f}'.format(test_epoch_loss))
        self.writer.add_scalar(tag='data/test_epoch_loss',
                               scalar_value=test_epoch_loss,
                               global_step=epoch)

    def _plot_images(self, actual, recon, global_step):
        imgs_act = vutils.make_grid(actual, normalize=True, scale_each=True)
        self.writer.add_image('img/actual', imgs_act, global_step)

        imgs_recon = vutils.make_grid(recon, normalize=True, scale_each=True)
        self.writer.add_image('img/recon', imgs_recon, global_step)

    def save(self, checkpoint_path):
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""
import logging
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Ours(pl.LightningModule):

    def __init__(self, hparams, train_loader: DataLoader, val_loader: DataLoader, img_shape: int = 28):
        super(Ours, self).__init__()
        self.hparams = hparams
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Define encoder-decoder
        self.encoder = Encoder(latent_dim=hparams.z_dim, img_shape=img_shape)
        self.decoder = Decoder(latent_dim=hparams.z_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        # Store rest of the hyper-params
        self.lr = hparams.lr
        self.lr_disc = hparams.lr_disc
        self.worm_up_epochs = hparams.worm_up_epochs

        # Intermediate results
        self.generated_imgs = None
        self.d_loss = 0
        self.g_loss = 0
        self.l2_loss = 0

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    def forward(self, x):
        # x is the image
        z_structure, z_rot = self.encoder(x)
        x_hat = self.decoder(z_structure, z_rot)
        return x_hat

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)

        scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, self.hparams.step_size)
        scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, self.hparams.step_size_disc)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, _ = batch
        epoch_curr = self.trainer.current_epoch

        # train generator
        if optimizer_idx == 0:
            # Generate images
            self.generated_imgs = self.forward(imgs)

            # Ground truth result (i.e.: all fake)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.cuda(imgs.device.index) if self.on_gpu else valid

            # L2 loss between input and output
            self.l2_loss = F.mse_loss(self.generated_imgs, imgs)

            # adversarial loss is binary cross-entropy
            disc_output = self.discriminator(self.generated_imgs)
            self.g_loss = self.adversarial_loss(disc_output, valid)

            # Overall loss
            loss = 0.5 * (self.l2_loss + self.g_loss) if epoch_curr >= self.worm_up_epochs else self.l2_loss


        # Train discriminator: Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.cuda(imgs.device.index) if self.on_gpu else valid
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.cuda(imgs.device.index) if self.on_gpu else fake
            fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            self.d_loss = 0.5 * (real_loss + fake_loss)

            loss = self.d_loss

            # If we are still in worm up phase- don't use the discriminator
            if epoch_curr < self.worm_up_epochs:
                loss *= 0


        else:
            raise ValueError(f'unexpected optimizer_idx={optimizer_idx}')
        output = OrderedDict({'loss': loss,
                              'd_loss': self.d_loss,
                              'g_loss': self.g_loss,
                              'l2_loss': self.l2_loss})
        # print(optimizer_idx, temp)
        return output

    def training_epoch_end(self, outputs):

        # Get statistic on generator and discriminator losses
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        logs = {'d_loss': torch.stack([out['d_loss'] for out in outputs]).mean(),
                'g_loss': torch.stack([out['g_loss'] for out in outputs]).mean(),
                'l2_loss': torch.stack([out['l2_loss'] for out in outputs]).mean()}

        # Print
        epoch_curr = self.trainer.current_epoch
        logger.info('[{:03d}/{}] Train [d_loss g_loss l2_loss]=[{:.3f} {:.3f} {:.3f}]'.format(
            epoch_curr, self.trainer.max_nb_epochs - 1, logs['d_loss'], logs['g_loss'], logs['l2_loss']))
        for k, v in logs.items():
            self.logger.experiment.add_scalar('train_' + k, v, epoch_curr)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch

        # Generator
        self.generated_imgs = self.forward(imgs)

        # Discriminator prediction
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.cuda(imgs.device.index) if self.on_gpu else fake
        fake_output = self.discriminator(self.generated_imgs)
        fake_loss = self.adversarial_loss(fake_output, fake)

        real = torch.ones(imgs.size(0), 1)
        real = real.cuda(imgs.device.index) if self.on_gpu else real
        real_output = self.discriminator(imgs)
        real_loss = self.adversarial_loss(real_output, fake)

        d_loss = 0.5 * (real_loss + fake_loss)

        # Generator loss: the generator wants that the fake output would be considered as real
        g_loss = self.adversarial_loss(fake_output, real)
        l2_loss = F.mse_loss(self.generated_imgs, imgs)

        # todo: add discriminator accuracy

        # losses
        disc_fake_loss = self.adversarial_loss(fake_output, fake)
        disc_real_loss = self.adversarial_loss(real_output, real)
        loss = disc_real_loss + disc_fake_loss + l2_loss

        output = OrderedDict({'loss': loss,
                              'd_loss': d_loss,
                              'g_loss': g_loss,
                              'l2_loss': l2_loss})

        return output

    def validation_epoch_end(self, outputs):
        logs = {'d_loss': torch.stack([out['d_loss'] for out in outputs]).mean(),
                'g_loss': torch.stack([out['g_loss'] for out in outputs]).mean(),
                'l2_loss': torch.stack([out['l2_loss'] for out in outputs]).mean()}
        val_loss = logs['g_loss'] + logs['l2_loss']

        # Print
        epoch_curr = self.trainer.current_epoch
        logger.info('[{:03d}/{}] Val   [d_loss g_loss l2_loss]=[{:.3f} {:.3f} {:.3f}]'.format(
            epoch_curr, self.trainer.max_nb_epochs - 1, logs['d_loss'], logs['g_loss'], logs['l2_loss']))

        # log sampled images
        if epoch_curr % 5 == 0:
            sample_imgs = self.generated_imgs[:12]
            grid = torchvision.utils.make_grid(sample_imgs, 2)
            self.logger.experiment.add_image(f'generated_images_epoch_{epoch_curr}', grid, global_step=epoch_curr)

        for k, v in logs.items():
            self.logger.experiment.add_scalar('val_' + k, v, epoch_curr)
        return {'val_loss': val_loss, 'logs': logs}


class Encoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Encoder, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(int(np.prod(img_shape)), 512, normalize=False),
            *block(512, 256),
            *block(256, 128),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        z = self.model(x_flat)
        z_rot = z[..., -1].unsqueeze(1)  # latent variable that encodes the rotation
        z_content = z[..., :-1]  # latent vector that encodes the structure
        return z_content, z_rot


class Decoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Decoder, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z_content, z_rot):
        z = torch.cat((z_content, z_rot), 1)
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""
import logging
from collections import OrderedDict

import matplotlib.pyplot as plt
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

        # Number of batch in which the generator is not updated
        self.generator_idle_freq = hparams.generator_idle_freq

        # Intermediate results
        self.gen_imgs = None
        self.d_loss = -1
        self.g_loss = -1
        self.img_loss = -1
        self.angle_loss = -1

        # Visualization
        self.batches_to_viz = hparams.batches_to_viz
        self.viz_num_per_batch = hparams.viz_num_per_batch
        self.viz_epoch_rate = hparams.viz_epoch_rate
        self.batch_idx_to_viz = np.linspace(0, len(val_loader) - 1, self.batches_to_viz).astype(int)
        logger.info('batch_idx_to_viz={}'.format(self.batch_idx_to_viz))

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    def forward(self, x, z_rot_input=None):
        """
        Inference through the encoder-decoder.
        :param x: Input image to encode
        :param z_rot_input: latent variable the is used as input to the decoder
        :return: generated image and the rotation latent variable
        """
        # x is the image
        z_structure, z_rot_output = self.encoder(x)
        z_rot = z_rot_output if z_rot_input is None else z_rot_input
        x_hat = self.decoder(z_structure, z_rot)
        return x_hat, z_rot_output.squeeze()

    @staticmethod
    def calc_adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    @staticmethod
    def calc_angle_loss(angle_hat, angle):
        return F.l1_loss(angle_hat, angle)

    @staticmethod
    def calc_img_loss(img_hat, img):
        return F.l1_loss(img_hat, img) + F.mse_loss(img_hat, img)

    def configure_optimizers(self):

        # Encoder-Decoder (generator)
        opt_g = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                 lr=self.lr, betas=(0.5, 0.999))
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, self.hparams.step_size)

        # Discriminator
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc, betas=(0.5, 0.999))
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, self.hparams.step_size_disc)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, rot, imgs_rot_0 = batch
        epoch_curr = self.trainer.current_epoch

        # Generate images
        z_rot_input = torch.zeros(len(imgs), 1)  # force the decoder to generate images with rotation zero
        z_rot_input = z_rot_input.cuda() if torch.cuda.is_available() else z_rot_input
        gen_imgs, z_rot = self.forward(imgs, z_rot_input=z_rot_input)

        # train generator
        if optimizer_idx == 0:

            # The loss between input and output image
            self.img_loss = self.calc_img_loss(gen_imgs, imgs_rot_0)

            # angle classier
            self.angle_loss = self.calc_angle_loss(z_rot, rot)

            # adversarial loss is binary cross-entropy
            disc_output = self.discriminator(gen_imgs)
            real = torch.ones(imgs.size(0), 1)
            real = real.cuda(imgs.device.index) if self.on_gpu else real
            self.g_loss = self.calc_adversarial_loss(disc_output, real)

            # Overall loss
            loss = (self.img_loss + self.angle_loss + self.g_loss) / 3
            is_correct = [-1]

        elif optimizer_idx == 1:
            # Train discriminator: Measure discriminator's ability to classify real from generated samples

            # How well can it label as real?
            real = torch.ones(imgs.size(0), 1).cuda(imgs.device.index) if self.on_gpu else torch.ones(imgs.size(0), 1)
            real_output = self.discriminator(imgs_rot_0)
            real_loss = self.calc_adversarial_loss(real_output, real)

            # How well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1).cuda(imgs.device.index) if self.on_gpu else torch.zeros(imgs.size(0), 1)
            fake_output = self.discriminator(gen_imgs.detach())
            fake_loss = self.calc_adversarial_loss(fake_output, fake)

            self.d_loss = 0.5 * (real_loss + fake_loss)
            loss = self.d_loss

            # Discriminator accuracy
            pred = fake_output.round()
            is_correct_fake = pred.eq(fake.view_as(pred))
            pred = real_output.round()
            is_correct_real = pred.eq(real.view_as(pred))
            is_correct = torch.cat((is_correct_fake, is_correct_real), 0).float()
        else:
            raise ValueError(f'unexpected optimizer_idx={optimizer_idx}')

        output = OrderedDict({'loss': loss,
                              'd_loss': self.d_loss,
                              'g_loss': self.g_loss,
                              'img_loss': self.img_loss,
                              'angle_loss': self.angle_loss,
                              'is_correct': is_correct})

        if batch_nb in self.batch_idx_to_viz and self.trainer.current_epoch % self.viz_epoch_rate == 0:
            self.visualize_batch(imgs.cpu().numpy(), imgs_rot_0.cpu().numpy(), np.copy(gen_imgs.detach().cpu().numpy()),
                                 rot, z_rot, batch_nb, prefix='train_')
        return output

    def training_epoch_end(self, outputs):
        # Get statistic on generator and discriminator losses
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        logs = {'d_loss': torch.stack([out['d_loss'] for out in outputs]).mean(),
                'g_loss': torch.stack([out['g_loss'] for out in outputs]).mean(),
                'img_loss': torch.stack([out['img_loss'] for out in outputs]).mean(),
                'angle_loss': torch.stack([out['angle_loss'] for out in outputs]).mean(),
                'd_acc': torch.cat([out['is_correct'] for out in outputs]).mean()}

        # Print
        epoch_curr = self.trainer.current_epoch
        logger.info(
            '[{:03d}/{}] Train [d_acc d_loss]=[{:.3f} {:.3f}] [g_loss img_loss angle_loss]=[{:.3f} {:.3f} {:.3f}]'.format(
                epoch_curr, self.trainer.max_nb_epochs - 1,
                logs['d_acc'], logs['d_loss'], logs['g_loss'], logs['img_loss'], logs['angle_loss']))
        for k, v in logs.items():
            self.logger.experiment.add_scalar('train_' + k, v, epoch_curr)
        return {'loss': loss}

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # update generator
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

        # update discriminator opt every generator_idle_freq steps
        if optimizer_idx == 1:
            if batch_idx % self.generator_idle_freq == 0:
                optimizer.step()
                optimizer.zero_grad()

        # todo: worm up

    def validation_step(self, batch, batch_idx):
        imgs, rot, imgs_rot_0 = batch

        # Generator
        imgs, imgs_rot_0 = imgs, imgs_rot_0
        z_rot_input = torch.zeros(len(imgs), 1)  # Force the decoder to generate images with rotation zero
        z_rot_input = z_rot_input.cuda() if torch.cuda.is_available() else z_rot_input
        gen_imgs, z_rot = self.forward(imgs, z_rot_input=z_rot_input)

        # Discriminator prediction
        fake = torch.zeros(imgs.size(0), 1).cuda(imgs.device.index) if self.on_gpu else torch.zeros(imgs.size(0), 1)
        fake_output = self.discriminator(gen_imgs)
        real = torch.ones(imgs.size(0), 1).cuda(imgs.device.index) if self.on_gpu else torch.ones(imgs.size(0), 1)
        real_output = self.discriminator(imgs_rot_0)

        # Discriminator Loss
        fake_loss = self.calc_adversarial_loss(fake_output, fake)
        real_loss = self.calc_adversarial_loss(real_output, real)
        d_loss = 0.5 * (real_loss + fake_loss)

        # Discriminator accuracy
        pred = fake_output.round()  # classification: 0 is fake and 1 is real
        is_correct_fake = pred.eq(fake.view_as(pred)).float()
        pred = real_output.round()  # classification: 0 is fake and 1 is real
        is_correct_real = pred.eq(real.view_as(pred)).float()

        # Generator loss: the generator wants that the fake output would be considered as real
        g_loss = self.calc_adversarial_loss(fake_output, real)
        img_loss = self.calc_img_loss(gen_imgs, imgs_rot_0)
        angle_loss = self.calc_angle_loss(z_rot, rot)

        # Total loss
        loss = d_loss + img_loss + angle_loss

        output = OrderedDict({'loss': loss,
                              'd_loss': d_loss,
                              'g_loss': g_loss,
                              'img_loss': img_loss,
                              'angle_loss': angle_loss,
                              'is_correct_fake': is_correct_fake,
                              'is_correct_real': is_correct_real})

        # Visualize
        if batch_idx in self.batch_idx_to_viz and self.trainer.current_epoch % self.viz_epoch_rate == 0:
            self.visualize_batch(imgs.cpu().numpy(), imgs_rot_0.cpu().numpy(), np.copy(gen_imgs.detach().cpu().numpy()),
                                 rot, z_rot, batch_idx, prefix='val_')
        self.gen_imgs = gen_imgs.detach().clone()
        return output

    def validation_epoch_end(self, outputs):
        logs = {'d_loss': torch.stack([out['d_loss'] for out in outputs]).mean(),
                'g_loss': torch.stack([out['g_loss'] for out in outputs]).mean(),
                'img_loss': torch.stack([out['img_loss'] for out in outputs]).mean(),
                'angle_loss': torch.stack([out['angle_loss'] for out in outputs]).mean(),
                'd_real_acc': torch.cat([out['is_correct_real'] for out in outputs]).mean(),
                'd_fake_acc': torch.cat([out['is_correct_fake'] for out in outputs]).mean()}
        val_loss = logs['g_loss'] + logs['img_loss'] + logs['angle_loss']

        # Get learning rates
        lrs = []
        for optimizer in self.trainer._get_optimizers_iterable():
            for param_group in optimizer[1].param_groups:
                lrs.append(param_group['lr'])

        # Print
        epoch_curr = self.trainer.current_epoch
        logger.info(
            '[{:03d}/{}] Val   [d_real_acc d_fake_acc d_loss]=[{:.3f} {:.3f} {:.3f}] [g_loss img_loss angle_loss]=[{:.3f} {:.3f} {:.3f}] lr={}'.format(
                epoch_curr, self.trainer.max_nb_epochs - 1,
                logs['d_real_acc'], logs['d_fake_acc'], logs['d_loss'], logs['g_loss'], logs['img_loss'],
                logs['angle_loss'], lrs))

        # log sampled images
        if epoch_curr % self.viz_epoch_rate == 0:
            sample_imgs = self.gen_imgs[:12]
            grid = torchvision.utils.make_grid(sample_imgs, 6)
            self.logger.experiment.add_image(f'generated_images_epoch_{epoch_curr}', grid, global_step=epoch_curr)
        for k, v in logs.items():
            self.logger.experiment.add_scalar('val_' + k, v, epoch_curr)
        return {'val_loss': val_loss, 'logs': logs}

    def visualize_batch(self, imgs: np.ndarray, imgs_rot_0: np.ndarray, gen_imgs: np.ndarray,
                        rot, z_rot, batch_idx: int, prefix: str = ''):
        w, h = plt.rcParams.get('figure.figsize')
        fig, axs = plt.subplots(3, self.viz_num_per_batch, figsize=(w * 2, h))
        for i, img_idx in enumerate(np.linspace(0, len(imgs) - 1, self.viz_num_per_batch).astype(int)):
            ax = axs[0, i]
            ax.imshow(imgs_rot_0[img_idx].squeeze(), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax = axs[1, i]
            ax.imshow(imgs[img_idx].squeeze(), cmap='gray')
            ax.set_title(r'$\theta$={:.2f}$^o$'.format(180 * rot[i].item() / np.pi))
            ax.set_xticks([])
            ax.set_yticks([])
            ax = axs[2, i]
            ax.imshow(gen_imgs[img_idx].squeeze(), cmap='gray')
            ax.set_title(r'$\hat \theta $={:.2f}$^o$'.format(180 * z_rot[i] / np.pi))
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0, 0].set_ylabel('Original')
        axs[1, 0].set_ylabel('Input')
        axs[2, 0].set_ylabel('Generated')
        fig.suptitle(f'{prefix} Epoch {self.trainer.current_epoch} Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig('{}epoch_{:03d}_batch_{:03d}.jpg'.format(prefix, self.trainer.current_epoch, batch_idx))
        plt.close()


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
            # nn.Sigmoid()
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
            nn.Linear(256, int(np.prod(img_shape))),
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

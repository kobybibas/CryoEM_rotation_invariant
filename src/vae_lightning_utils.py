"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""
import logging
import os
import os.path as osp

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class VAE(pl.LightningModule):

    def __init__(self, hparams, train_loader: DataLoader, val_loader: DataLoader, img_shape: tuple):
        super(VAE, self).__init__()
        self.hparams = hparams
        self.train_loader = train_loader
        self.val_loader = val_loader

        # coordinates array
        x0_grid, x1_grid = np.linspace(-1, 1, img_shape[1]), np.linspace(1, -1, img_shape[2])
        x0, x1 = np.meshgrid(x0_grid, x1_grid)
        x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
        self.x_coord = torch.from_numpy(x_coord).float()
        self.x_coord = self.x_coord.cuda() if torch.cuda.is_available() else self.x_coord.cpu()

        # Define encoder-decoder
        self.p_net = SpatialGenerator(hparams.z_dim, hparams.hidden_dim, img_shape,
                                      num_layers=hparams.num_layers)  # Decoder
        inf_dim = hparams.z_dim + 1
        self.q_net = InferenceNetwork(np.prod(img_shape), inf_dim, hparams.hidden_dim,
                                      num_layers=hparams.num_layers)  # Encoder

        # Store rest of the hyper-params
        self.theta_prior = hparams.theta_prior
        self.lr = hparams.lr

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        optim = torch.optim.Adam(list(self.p_net.parameters()) + list(self.q_net.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.step_size)
        return [optim], [scheduler]

    def forward(self, x, y, theta_input=None):
        """
        Inference
        :param x: The coordinates of the entire image [0,0],[0,0.01],...[1,1]
        :param y: The batch images
        :param theta_input: enforcing rotation angle
        :return:
        """

        # Expand x to match batch size
        b = y.size(0)
        x = x.expand(b, x.size(0), x.size(1))

        # Encoder
        z_mu, z_logstd = self.q_net(y)
        z_std = torch.exp(z_logstd)
        z_dim = z_mu.size(1)

        # Draw samples from variational posterior to calculate E[p(x|z)]
        r = x.data.new(b, z_dim).normal_()
        z = z_std * r + z_mu

        # In our case, we assume always rotation. equivalent to: if rotate is true in the original code
        theta = z[:, 0]  # z[0] is the rotation
        z = z[:, 1:]

        if theta_input is not None:
            theta = torch.tensor([theta_input]).float()

        # Calculate rotation matrix
        rot = theta.data.new(b, 2, 2).zero_()
        rot[:, 0, 0] = torch.cos(theta)
        rot[:, 0, 1] = torch.sin(theta)
        rot[:, 1, 0] = -torch.sin(theta)
        rot[:, 1, 1] = torch.cos(theta)

        # Coordinate transformation
        x = torch.bmm(x, rot)  # rotate coordinates by theta

        # Decoder
        y_hat = self.p_net(x.contiguous(), z)
        return y_hat, z_mu, z_logstd, z_std

    def calc_log_p_x_g_z(self, y_hat_flat, y_flat, size):
        if self.hparams.dataset == 'mnist':
            log_p_x_g_z = -size * F.binary_cross_entropy_with_logits(y_hat_flat, y_flat)
        elif self.hparams.dataset == '5hdb':
            log_p_x_g_z = -0.5 * torch.sum((y_hat_flat - y_flat) ** 2, 1).mean()
        else:
            raise ValueError(f'{self.hparams.dataset} is not supported')
        return log_p_x_g_z

    def training_step(self, batch, batch_nb):
        y = batch[0]  # get the rotated image
        x_coord = self.x_coord

        y_hat, z_mu, z_logstd, z_std = self.forward(x_coord, y)

        # z[0] is the latent variable that corresponds to the rotation
        theta_mu, theta_std, theta_logstd = z_mu[:, 0], z_std[:, 0], z_logstd[:, 0]
        z_mu, z_std, z_logstd = z_mu[:, 1:], z_std[:, 1:], z_logstd[:, 1:]

        # Calculate the KL divergence term
        sigma = self.theta_prior
        kl_div = -theta_logstd + np.log(sigma) + (theta_std ** 2 + theta_mu ** 2) / 2 / sigma ** 2 - 0.5

        y_flat, y_hat_flat = y.view(y.size(0), -1), y_hat.view(y_hat.size(0), -1)
        size = y_flat.size(1)
        log_p_x_g_z = self.calc_log_p_x_g_z(y_hat_flat, y_flat, size)

        # Unit normal prior over z and translation
        z_kl = -z_logstd + 0.5 * z_std ** 2 + 0.5 * z_mu ** 2 - 0.5
        kl_div = kl_div + torch.sum(z_kl, 1)
        kl_div = kl_div.mean()

        elbo = log_p_x_g_z - kl_div
        loss = -elbo
        return {'loss': loss,
                'log': {'elbo': elbo,
                        'log_p_x_g_z': log_p_x_g_z,
                        'kl_div': kl_div}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log_epoch('Train', avg_loss.item())
        return {'loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        y = batch[0]  # get the rotated image
        x_coord = self.x_coord
        y_hat, z_mu, z_logstd, z_std = self.forward(x_coord, y)

        # z[0] is the rotation
        theta_mu, theta_std, theta_logstd = z_mu[:, 0], z_std[:, 0], z_logstd[:, 0]
        z_mu, z_std, z_logstd = z_mu[:, 1:], z_std[:, 1:], z_logstd[:, 1:]

        # Calculate the KL divergence term
        sigma = self.theta_prior
        kl_div = -theta_logstd + np.log(sigma) + (theta_std ** 2 + theta_mu ** 2) / 2 / sigma ** 2 - 0.5

        y_flat, y_hat_flat = y.view(y.size(0), -1), y_hat.view(y_hat.size(0), -1)
        size = y_flat.size(1)
        log_p_x_g_z = self.calc_log_p_x_g_z(y_hat_flat, y_flat, size)

        # Unit normal prior over z and translation
        z_kl = -z_logstd + 0.5 * z_std ** 2 + 0.5 * z_mu ** 2 - 0.5
        kl_div = kl_div + torch.sum(z_kl, 1)
        kl_div = kl_div.mean()

        elbo = log_p_x_g_z - kl_div
        loss = -elbo
        return {'val_loss': loss, 'elbo': elbo, 'log_p_x_g_z': log_p_x_g_z, 'kl_div': kl_div}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss,
                'avg_elbo': torch.stack([x['elbo'] for x in outputs]).mean(),
                'avg_log_p_x_g_z': torch.stack([x['log_p_x_g_z'] for x in outputs]).mean(),
                'avg_kl_div': torch.stack([x['kl_div'] for x in outputs]).mean()
                }
        self.log_epoch('Val', avg_loss.item(), logs['avg_elbo'].item(), logs['avg_log_p_x_g_z'].item(),
                       logs['avg_kl_div'].item())

        # save model
        save_file = osp.join(os.getcwd(), 'vae_{}_encoder.pth'.format(self.hparams.dataset))
        torch.save(self.q_net.state_dict(), save_file)
        logger.info('Saving model: {}'.format(save_file))
        save_file = osp.join(os.getcwd(), 'vae_{}_decoder.pth'.format(self.hparams.dataset))
        torch.save(self.p_net.state_dict(), save_file)
        logger.info('Saving model: {}'.format(save_file))

        return {'val_loss': avg_loss, 'log': logs}

    def log_epoch(self, print_type: str, loss=-1, elbo=-1, log_p_x_g_z=-1, kl_div=-1):
        logger.info('')
        logger.info('[{}/{}] \t {} \t [loss elbo log_p_x_g_z kl_div]=[{:.3f} {:.3f} {:.3f} {:.3f}]'.format(
            self.trainer.current_epoch, self.trainer.max_nb_epochs - 1,
            print_type, loss, elbo, log_p_x_g_z, kl_div))


class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class InferenceNetwork(nn.Module):
    def __init__(self, n, latent_dim, hidden_dim, num_layers=1, activation=nn.Tanh):
        super(InferenceNetwork, self).__init__()

        self.latent_dim = latent_dim
        self.n = n

        layers = [nn.Linear(n, hidden_dim),
                  activation(),
                  ]
        for _ in range(1, num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, 2 * latent_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x is (batch,num_coords)
        x_flat = x.view(x.size(0), -1)
        z = self.layers(x_flat)

        ld = self.latent_dim
        z_mu = z[:, :ld]
        z_logstd = z[:, ld:]

        return z_mu, z_logstd


class SpatialGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, img_shape, n_out=1, num_layers=1, activation=nn.Tanh):
        super(SpatialGenerator, self).__init__()

        self.img_shape = img_shape

        in_dim = 2
        self.coord_linear = nn.Linear(in_dim, hidden_dim)
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)

        layers = [activation()]
        for _ in range(1, num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, n_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):
        # x is (batch, num_coords, 2)
        # z is (batch, latent_dim)

        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)
        x = x.view(b * n, -1)

        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)

        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)

        h = h_x + h_z  # (batch, num_coords, hidden_dim)
        h = h.view(b * n, -1)

        y = self.layers(h)  # (batch*num_coords, nout)
        y = y.view(b, n, -1)

        img = y.view(y.size(0), *self.img_shape)
        return img


def load_vae_model(out_base_dir: str, train_loader, test_loader, image_shape):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    vae_mnist_config = osp.join(out_base_dir, 'lightning_logs/version_0/hparams.yaml')
    with open(vae_mnist_config) as f:
        hparams_dict = yaml.safe_load(f)
    hparams = AttrDict()
    hparams.update(hparams_dict)
    vae_model = VAE(hparams, train_loader, test_loader, image_shape)

    # Load pretrained model
    encoder_pretrained_path = osp.join(out_base_dir, f'vae_{hparams.dataset}_encoder.pth')
    decoder_pretrained_path = osp.join(out_base_dir, f'vae_{hparams.dataset}_decoder.pth')
    vae_model.q_net.load_state_dict(torch.load(encoder_pretrained_path, map_location=lambda storage, loc: storage))
    vae_model.p_net.load_state_dict(torch.load(decoder_pretrained_path, map_location=lambda storage, loc: storage))
    return vae_model

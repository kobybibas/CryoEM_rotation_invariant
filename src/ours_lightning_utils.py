"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import yaml
from collections import OrderedDict
from torch import optim
from torch.utils.data import DataLoader

from gan_utils import Encoder, Decoder, Discriminator, DiscriminatorCNN, DecoderCNN

logger = logging.getLogger(__name__)


class Ours(pl.LightningModule):

    def __init__(self, hparams, train_loader: DataLoader, val_loader: DataLoader, img_shape: int = 28):
        super(Ours, self).__init__()
        self.hparams = hparams
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Define encoder-decoder
        if hparams.architecture == 'fc':
            self.encoder = Encoder(latent_dim=hparams.z_dim, img_shape=img_shape)
            self.decoder = Decoder(latent_dim=hparams.z_dim, img_shape=img_shape)
            self.discriminator = Discriminator(img_shape=img_shape)
        elif hparams.architecture == 'cnn':
            self.encoder = Encoder(latent_dim=hparams.z_dim, img_shape=img_shape)
            self.decoder = DecoderCNN(latent_dim=hparams.z_dim, img_shape=img_shape,
                                      last_activation=hparams.last_activation)
            self.discriminator = DiscriminatorCNN(img_shape=img_shape, is_linear_output=self.hparams.use_wasserstein)
            self.angle_classifier = DiscriminatorCNN(img_shape=img_shape, is_linear_output=True)
        else:
            raise ValueError(f'{hparams.architecture} is not supported')

        # clamp the model weights
        self.clipper_h = WeightClipper()

        # Store rest of the hyper-params
        self.lr = hparams.lr
        self.lr_disc = hparams.lr_disc

        # Number of batch in which the generator is not updated
        self.generator_idle_freq = hparams.generator_idle_freq

        # Number of batch in which the discriminator is not updated
        self.discriminator_idle_freq = hparams.discriminator_idle_freq

        # whether to use wasserstein gan:
        # https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
        self.use_wasserstein = hparams.use_wasserstein

        # Intermediate results
        self.gen_imgs = None
        self.d_loss = 0
        self.g_loss = 0
        self.img_loss = 0
        self.angle_loss = 0

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

    def calc_adversarial_loss(self, y_hat, y) -> torch.Tensor:
        if self.use_wasserstein is False:
            loss = F.binary_cross_entropy(y_hat, y)
        else:
            loss = torch.mean(y_hat * y)
        return loss

    @staticmethod
    def calc_angle_loss(angle_hat, angle) -> torch.Tensor:
        return torch.exp(F.mse_loss(angle_hat.float(), angle.float())) - 1

    @staticmethod
    def calc_img_loss(img_hat, img):
        return F.l1_loss(img_hat, img) + F.mse_loss(img_hat, img)

    def configure_optimizers(self):
        betas = (0.5, 0.999)

        # Encoder-Decoder (generator)
        enc_dec_p = list(self.encoder.parameters()) + list(self.decoder.parameters())
        # if self.use_wasserstein is False:
        opt_g = optim.Adam(enc_dec_p, lr=self.lr, betas=betas, weight_decay=self.hparams.weight_decay)
        # else:
        #     opt_g = optim.RMSprop(enc_dec_p, lr=self.lr, weight_decay=self.hparams.weight_decay)
        scheduler_g = optim.lr_scheduler.StepLR(opt_g, self.hparams.step_size)

        # Discriminator
        # if self.use_wasserstein is False:
        disc_p = list(self.discriminator.parameters()) + list(self.angle_classifier.parameters())
        opt_d = optim.Adam(disc_p, lr=self.lr_disc, betas=betas,
                           weight_decay=self.hparams.weight_decay)
        # else:
        #     opt_d = optim.RMSprop(self.discriminator.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        scheduler_d = optim.lr_scheduler.StepLR(opt_d, self.hparams.step_size_disc)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def create_labels(self, real_imgs_num: int, fake_imgs_num: int) -> (torch.Tensor, torch.Tensor):
        if self.use_wasserstein is False:
            # generate class labels, 1 for 'real'
            real = torch.ones(real_imgs_num, 1).cuda() if self.on_gpu else torch.ones(real_imgs_num, 1)
            # generate class labels, 0 for 'fake'
            fake = torch.zeros(fake_imgs_num, 1).cuda() if self.on_gpu else torch.zeros(fake_imgs_num, 1)
        else:
            # generate class labels, -1 for 'real'
            real = -torch.ones(real_imgs_num, 1).cuda() if self.on_gpu else -torch.ones(real_imgs_num, 1)
            # generate class labels, 1 for 'fake'
            fake = torch.ones(fake_imgs_num, 1).cuda() if self.on_gpu else torch.ones(fake_imgs_num, 1)
        return real, fake

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, rot, imgs_rot0 = batch
        is_correct = []

        # Generate images
        z_rot_input = torch.zeros(len(imgs), 1)  # Force the decoder to generate images with rotation zero
        gen_imgs, z_rot = self.forward(imgs, z_rot_input=z_rot_input.cuda() if self.on_gpu else z_rot_input)

        if optimizer_idx == 0:  # Train generator

            # The loss between input and output image
            self.img_loss = self.calc_img_loss(gen_imgs, imgs_rot0)

            # angle classier
            self.angle_loss = self.calc_angle_loss(z_rot, rot)

            angle_classifier_output = self.angle_classifier(gen_imgs)
            classifier_angle_loss = self.calc_angle_loss(angle_classifier_output,
                                                         rot * 0)  # we want the the angle will be zero

            # adversarial loss is binary cross-entropy, fake labels are real for generator cost
            disc_output = self.discriminator(gen_imgs)
            real, fake = self.create_labels(imgs.size(0), imgs.size(0))
            self.g_loss = self.calc_adversarial_loss(disc_output, real)  # maximize log(D(G(z)))

            # Overall loss
            loss = (1 / 4) * (self.img_loss + self.angle_loss + self.g_loss + classifier_angle_loss)

        elif optimizer_idx == 1:  # Train discriminator
            self.d_loss, is_correct_real, is_correct_fake, _, _ = self.calc_disc_perf(gen_imgs, imgs_rot0)
            is_correct = torch.cat((is_correct_fake, is_correct_real), 0).float()

            angle_classifier_output = self.angle_classifier(imgs)
            classifier_angle_loss = self.calc_angle_loss(angle_classifier_output, rot)

            # Overall loss
            loss = 0.5 * (self.d_loss + classifier_angle_loss)

        else:
            raise ValueError(f'unexpected optimizer_idx={optimizer_idx}')

        return OrderedDict({'loss': loss,
                            'd_loss': self.d_loss,
                            'g_loss': self.g_loss,
                            'img_loss': self.img_loss,
                            'angle_loss': self.angle_loss,
                            'is_correct': is_correct})

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
        logger.info('')
        logger.info(
            '[{:03d}/{}] Train [d_acc d_loss]=[{:.3f} {:.3f}] [g_loss img_loss angle_loss]=[{:.3f} {:.3f} {:.3f}]'.format(
                epoch_curr, self.trainer.max_nb_epochs - 1,
                logs['d_acc'], logs['d_loss'], logs['g_loss'], logs['img_loss'], logs['angle_loss']))
        for k, v in logs.items():
            self.logger.experiment.add_scalar('train_' + k, v, epoch_curr)
        return {'loss': loss}

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # In the WGAN model, the critic model must be updated more than the generator model.
        # set generator_idle_freq higher
        if optimizer_idx == 0 and batch_idx % self.generator_idle_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

        # update discriminator (critic) opt every discriminator_idle_freq steps
        if optimizer_idx == 1 and batch_idx % self.discriminator_idle_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Clipping allows us to enforce the Lipschitz constraint on the critic’s model
            if self.use_wasserstein:
                self.discriminator.apply(self.clipper_h)

    def validation_step(self, batch, batch_idx):
        imgs, rot, imgs_rot0 = batch

        # Generator
        z_rot_input = torch.zeros(len(imgs), 1).cuda() if self.on_gpu else torch.zeros(len(imgs), 1)
        gen_imgs, z_rot = self.forward(imgs, z_rot_input=z_rot_input)

        # Discriminator prediction
        d_loss, is_correct_real, is_correct_fake, fake_output, _ = self.calc_disc_perf(gen_imgs, imgs_rot0)

        # Generator loss: the generator wants that the fake output would be considered as real
        real = torch.ones(gen_imgs.size(0), 1).cuda() if self.on_gpu else torch.ones(gen_imgs.size(0), 1)
        g_loss = self.calc_adversarial_loss(fake_output, real)
        img_loss = self.calc_img_loss(gen_imgs, imgs_rot0)
        angle_loss = self.calc_angle_loss(z_rot.squeeze(), rot.squeeze())

        # Total loss
        loss = 0.25 * (d_loss + g_loss + img_loss + angle_loss)

        output = OrderedDict({'loss': loss,
                              'd_loss': d_loss,
                              'g_loss': g_loss,
                              'img_loss': img_loss,
                              'angle_loss': angle_loss,
                              'is_correct_fake': is_correct_fake,
                              'is_correct_real': is_correct_real})

        # Visualize
        if batch_idx in self.batch_idx_to_viz and self.trainer.current_epoch % self.viz_epoch_rate == 0:
            self.visualize_batch(imgs.cpu().numpy(), imgs_rot0.cpu().numpy(), np.copy(gen_imgs.detach().cpu().numpy()),
                                 rot, z_rot, batch_idx, prefix='val_')
        self.gen_imgs = gen_imgs.detach().clone()
        return output

    def validation_epoch_end(self, outputs):
        logs = {'loss': torch.stack([out['loss'] for out in outputs]).mean(),
                'd_loss': torch.stack([out['d_loss'] for out in outputs]).mean(),
                'g_loss': torch.stack([out['g_loss'] for out in outputs]).mean(),
                'img_loss': torch.stack([out['img_loss'] for out in outputs]).mean(),
                'angle_loss': torch.stack([out['angle_loss'] for out in outputs]).mean(),
                'd_real_acc': torch.cat([out['is_correct_real'] for out in outputs]).mean(),
                'd_fake_acc': torch.cat([out['is_correct_fake'] for out in outputs]).mean()}

        # Print
        epoch_curr = self.trainer.current_epoch
        logger.info('')
        logger.info(
            '[{:03d}/{}] Val   [d_real_acc d_fake_acc d_loss]=[{:.3f} {:.3f} {:.3f}] [g_loss img_loss angle_loss]=[{:.3f} {:.3f} {:.3f}] lr={}'.format(
                epoch_curr, self.trainer.max_nb_epochs - 1,
                logs['d_real_acc'], logs['d_fake_acc'], logs['d_loss'], logs['g_loss'], logs['img_loss'],
                logs['angle_loss'], self.get_current_learning_rates()))

        # log sampled images in tensorboard
        if epoch_curr % self.viz_epoch_rate == 0:
            sample_imgs = self.gen_imgs[:12]
            grid = torchvision.utils.make_grid(sample_imgs, 6)
            self.logger.experiment.add_image(f'generated_images_epoch_{epoch_curr}', grid, global_step=epoch_curr)
        for k, v in logs.items():
            self.logger.experiment.add_scalar('val_' + k, v, epoch_curr)

        # Save models
        self.save_model()
        return {'val_loss': logs['loss'], 'logs': logs}

    def save_model(self):
        save_file = osp.join(os.getcwd(), 'our_{}_decoder.pth'.format(self.hparams.dataset))
        torch.save(self.decoder.state_dict(), save_file)
        logger.info('Saving model: {}'.format(save_file))
        save_file = osp.join(os.getcwd(), 'our_{}_encoder.pth'.format(self.hparams.dataset))
        torch.save(self.encoder.state_dict(), save_file)
        logger.info('Saving model: {}'.format(save_file))

    def get_current_learning_rates(self) -> list:
        lrs = []
        for optimizer in self.trainer._get_optimizers_iterable():
            for param_group in optimizer[1].param_groups:
                lrs.append(param_group['lr'])
        return lrs

    def calc_disc_perf(self, imgs_fake: torch.Tensor, imgs_real: torch.Tensor):
        """
        Calculate the discriminator performance
        :param imgs_fake: fake images, the generator output
        :param imgs_real: real images, from the given dataset
        :return: discriminator performance
        """
        # create labels
        real, fake = self.create_labels(imgs_real.size(0), imgs_fake.size(0))
        real, fake = real.int(), fake.int()

        # How well can it label as fake?
        fake_output = self.discriminator(imgs_fake)
        fake_loss = self.calc_adversarial_loss(fake_output, fake)

        # How well can it label as real?
        real_output = self.discriminator(imgs_real)
        real_loss = self.calc_adversarial_loss(real_output, real)

        # Discriminator Loss
        d_loss = 0.5 * (real_loss + fake_loss)

        # Discriminator accuracy
        if self.use_wasserstein is False:
            # classification: 0 is fake and 1 is real
            pred_fake, pred_real = fake_output.round(), real_output.round()
        else:
            # classification: 1 is fake and -1 is real, here labels flip (not as in the loss)
            pred_fake, pred_real = -torch.sign(fake_output).int(), -torch.sign(real_output).int()

        is_correct_fake = pred_fake.eq(fake.view_as(pred_fake)).float()
        is_correct_real = pred_real.eq(real.view_as(pred_real)).float()
        return d_loss, is_correct_real, is_correct_fake, fake_output, real_output

    def visualize_batch(self, imgs: np.ndarray, imgs_rot_0: np.ndarray, gen_imgs: np.ndarray,
                        rot, z_rot, batch_idx: int, prefix: str = ''):
        w, h = plt.rcParams.get('figure.figsize')
        fig, axs = plt.subplots(3, self.viz_num_per_batch, figsize=(w * 2, h))
        for i, img_idx in enumerate(np.linspace(0, len(imgs) - 1, self.viz_num_per_batch).astype(int)):
            ax = axs[0, i]
            ax.imshow(imgs_rot_0[img_idx].squeeze(), cmap='gray')
            ax = axs[1, i]
            ax.imshow(imgs[img_idx].squeeze(), cmap='gray')
            ax.set_title(r'$\theta$={:.2f}$^o$'.format(180 * rot[i].item() / np.pi))
            ax = axs[2, i]
            ax.imshow(gen_imgs[img_idx].squeeze(), cmap='gray')
            ax.set_title(r'$\hat \theta $={:.2f}$^o$'.format(180 * z_rot[i] / np.pi))

        [ax.set_xticks([]) for ax in axs.flatten()]
        [ax.set_yticks([]) for ax in axs.flatten()]

        axs[0, 0].set_ylabel('Original')
        axs[1, 0].set_ylabel('Input')
        axs[2, 0].set_ylabel('Generated')
        fig.suptitle(f'{prefix} Epoch {self.trainer.current_epoch} Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig('{}epoch_{:03d}_batch_{:03d}.jpg'.format(prefix, self.trainer.current_epoch, batch_idx))
        plt.close()


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-0.01, 0.01)


def load_our_model(out_base_dir: str, train_loader, test_loader, image_shape):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    vae_mnist_config = osp.join(out_base_dir, 'lightning_logs/version_0/hparams.yaml')
    with open(vae_mnist_config) as f:
        hparams_dict = yaml.safe_load(f)
    hparams = AttrDict()
    hparams.update(hparams_dict)
    our_model = Ours(hparams, train_loader, test_loader, image_shape)

    # Load pretrained model
    encoder_pretrained_path = osp.join(out_base_dir, f'our_{hparams.dataset}_encoder.pth')
    decoder_pretrained_path = osp.join(out_base_dir, f'our_{hparams.dataset}_decoder.pth')
    our_model.encoder.load_state_dict(torch.load(encoder_pretrained_path, map_location=lambda storage, loc: storage))
    our_model.decoder.load_state_dict(torch.load(decoder_pretrained_path, map_location=lambda storage, loc: storage))
    return our_model

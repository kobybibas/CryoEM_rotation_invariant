import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F


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
    def __init__(self, latent_dim, img_shape, last_activ='sigmoid'):
        super(Decoder, self).__init__()
        self.img_shape = img_shape
        self.last_activation = last_activ

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
        if self.last_activation == 'tanh':
            img = torch.tanh(img)
        elif self.last_activation == 'sigmoid':  # values between 0 and 1
            img = torch.sigmoid(img)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, is_linear_output: bool = False):
        super(Discriminator, self).__init__()
        self.is_linear_output = is_linear_output

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        validity = torch.sigmoid(validity) if self.is_linear_output is False else validity
        return validity


# --------------------------- #
# CNN based Encoder-Decoder
# --------------------------- #


class DecoderCNN(nn.Module):
    # based on DCGAN

    def __init__(self, latent_dim, img_shape, featmap_dim=64, n_channel=1, last_activ='sigmoid'):
        super(DecoderCNN, self).__init__()
        self.featmap_dim = featmap_dim
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.last_activation = last_activ

        self.fc1 = nn.Linear(self.latent_dim, 4 * 4 * self.featmap_dim)
        self.conv1 = nn.ConvTranspose2d(self.featmap_dim, int(self.featmap_dim / 2), 5, stride=2, padding=2)

        self.BN1 = nn.BatchNorm2d(int(self.featmap_dim / 2))
        self.conv2 = nn.ConvTranspose2d(int(self.featmap_dim / 2), int(self.featmap_dim / 4), 6, stride=2, padding=2)

        self.BN2 = nn.BatchNorm2d(int(self.featmap_dim / 4))

        # Determine the kernel size such that the output is as in img_shape
        # The formula: h_out = (h_in - 1)* stride - 2 * pad + 1 + dilation*(kernel_size-1)
        h_in = 14
        stride = 2
        pad = 2
        dilation = 1
        h_out = self.img_shape[-1]
        kernel_size = (1 / dilation) * (h_out - (h_in - 1) * stride + 2 * pad - 1) + 1
        kernel_size = int(kernel_size)  # todo: add warning if not int
        self.conv3 = nn.ConvTranspose2d(int(self.featmap_dim / 4), n_channel, kernel_size, stride=stride, padding=pad)

    def forward(self, z_content, z_rot):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after conv but not at output layer,
        ReLU activation function.
        """
        x = torch.cat((z_content, z_rot), 1)
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        x = F.leaky_relu(self.BN1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = self.conv3(x)

        if self.last_activation == 'tanh':  # values between -1 to 1
            x = torch.tanh(x)
        elif self.last_activation == 'sigmoid':  # values between 0 and 1
            x = torch.sigmoid(x)

        return x


class DiscriminatorCNN(nn.Module):

    def __init__(self, img_shape, feature_map_dim=8, n_channel=1, is_linear_output: bool = False):
        super(DiscriminatorCNN, self).__init__()
        self.img_shape = img_shape
        self.feature_map_dim = feature_map_dim
        self.is_linear_output = is_linear_output

        # Define the layers
        self.conv1 = nn.Conv2d(n_channel, int(self.feature_map_dim / 4), 5, stride=2, padding=2)

        self.conv2 = nn.Conv2d(int(self.feature_map_dim / 4), int(self.feature_map_dim / 2), 5, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(int(self.feature_map_dim / 2))

        self.conv3 = nn.Conv2d(int(self.feature_map_dim / 2), self.feature_map_dim, 5, stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(self.feature_map_dim)

        self.fc = nn.Linear(int(self.feature_map_dim * 4 * 4), 1)
        if self.img_shape[-1] == 40:  # in case of 5hdb data
            self.fc = nn.Linear(200, 1)

    def forward(self, x_in):
        """
        Strided conv layers,
        Batch Normalization after conv but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """

        x = F.leaky_relu(self.conv1(x_in), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)
        x = torch.sigmoid(x) if self.is_linear_output is False else x
        return x

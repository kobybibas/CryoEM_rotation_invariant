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


# --------------------------- #
# CNN based Encoder-Decoder
# --------------------------- #


class DecoderCNN(nn.Module):
    # based on DCGAN

    def __init__(self, latent_dim, img_shape, featmap_dim=1024, n_channel=1):
        super(DecoderCNN, self).__init__()
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(latent_dim, 4 * 4 * featmap_dim)
        self.conv1 = nn.ConvTranspose2d(featmap_dim, int(featmap_dim / 2), 5, stride=2, padding=2)

        self.BN1 = nn.BatchNorm2d(int(featmap_dim / 2))
        self.conv2 = nn.ConvTranspose2d(int(featmap_dim / 2), int(featmap_dim / 4), 6, stride=2, padding=2)

        self.BN2 = nn.BatchNorm2d(int(featmap_dim / 4))
        self.conv3 = nn.ConvTranspose2d(int(featmap_dim / 4), n_channel, 6, stride=2, padding=2)

    def forward(self, z_content, z_rot):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after conv but not at output layer,
        ReLU activation function.
        """
        x = torch.cat((z_content, z_rot), 1)
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = torch.tanh(self.conv3(x))

        return x


class DiscriminatorCNN(nn.Module):

    def __init__(self, img_shape, feature_map_dim=512, n_channel=1):
        super(DiscriminatorCNN, self).__init__()
        self.img_shape = img_shape
        self.featmap_dim = feature_map_dim
        self.conv1 = nn.Conv2d(n_channel, int(feature_map_dim / 4), 5, stride=2, padding=2)

        self.conv2 = nn.Conv2d(int(feature_map_dim / 4), int(feature_map_dim / 2), 5, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(int(feature_map_dim / 2))

        self.conv3 = nn.Conv2d(int(feature_map_dim / 2), feature_map_dim, 5, stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(feature_map_dim)

        self.fc = nn.Linear(feature_map_dim * 4 * 4, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after conv but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        x = torch.sigmoid(self.fc(x))
        return x

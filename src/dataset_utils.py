import logging
import os.path as osp
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import MNIST

logger = logging.getLogger(__name__)


def get_dataset(dataset_name: str,
                batch_size: int = 128,
                num_workers: int = 4,
                data_base_dir: str = osp.join('..', '..', 'data')):
    # Trainset
    if dataset_name == 'mnist_rotated':
        logger.info('Training on rotated MNIST')
        mnist_train = np.load(osp.join(data_base_dir, 'mnist_rotated', 'images_train.npy'))
        mnist_test = np.load(osp.join(data_base_dir, 'mnist_rotated', 'images_test.npy'))
        mnist_train = torch.from_numpy(mnist_train).float() / 255
        mnist_test = torch.from_numpy(mnist_test).float() / 255
        image_shape = n = m = 28
        y_train = mnist_train.view(-1, n * m)
        y_test = mnist_test.view(-1, n * m)
        trainset = torch.utils.data.TensorDataset(y_train)
        testset = torch.utils.data.TensorDataset(y_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    # elif dataset_name == 'mnist':
    #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    #     trainset = MNIST(data_base_dir, train=True, download=True, transform=transform)
    #     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
    #                                                num_workers=num_workers)
    #     testset = MNIST(data_base_dir, train=False, download=True, transform=transform)
    #     test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    #     image_shape = (1, 28, 28)
    elif dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        trainset = MnistRotate(data_base_dir, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)
        testset = MnistRotate(data_base_dir, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
        image_shape = (1, 28, 28)
    else:
        raise ValueError('Dataset {} is not supported'.format(dataset_name))

    return train_loader, test_loader, image_shape


def visualize_dataset(dataloader: torch.utils.data.DataLoader, dataset_index: int, image_shape: int):
    # Get data
    img = dataloader.dataset[dataset_index][0]

    # Reshape to 2D
    img_np = img.view(image_shape, image_shape).numpy()
    return img_np


class MnistRotate(MNIST):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        # rotation between -90 to 90
        self.rot_deg = 0
        self.rotations_legit = torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, 1000)).float()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, _ = self.data[index], int(self.targets[index])

        # Rotate
        if self.train is False:
            angle = self.rotations_legit[index % len(self.rotations_legit)]
        else:
            angle = random.choice(self.rotations_legit)
        angle_deg = angle * 180 / np.pi
        img = Image.fromarray(img.numpy(), mode='L')
        img_rot = TF.rotate(img, angle_deg)
        # img = TF.to_tensor(img)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
            img_rot = self.transform(img_rot)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img_rot, angle, img

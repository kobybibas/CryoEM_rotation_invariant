import logging
import os.path as osp

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_dataset(dataset_name: str,
                batch_size: int=128,
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
    else:
        raise ValueError('Dataset {} is not supported'.format(dataset_name))

    return train_loader, test_loader, image_shape

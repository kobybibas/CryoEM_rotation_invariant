import cv2
import logging
import mrc
import numpy as np
import os.path as osp
import pickle
import random
import scipy.ndimage
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

transform_mnist = transforms.Compose([transforms.ToTensor()])
transform_particles = transforms.Compose([transforms.CenterCrop(100),
                                          transforms.Resize(40, interpolation=2),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.3907], [0.0948 / 2.3510475]),  # match std
                                          transforms.Normalize([-0.0065 - 0.9820243], [1])]  # match mean
                                         )


def get_dataset(dataset_name: str,
                batch_size: int = 128,
                n_workers: int = 4,
                data_base_dir: str = osp.join('..', '..', 'data')):
    if dataset_name == 'mnist':
        image_shape = (1, 28, 28)

        # Trainset
        trainset = MnistRotate(data_base_dir, train=True, download=True, transform=transform_mnist)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

        # Testset
        testset = MnistRotate(data_base_dir, train=False, download=True, transform=transform_mnist)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=n_workers)

    elif dataset_name == '5hdb':
        image_shape = (1, 40, 40)

        # Trainset
        trainset = EM_5HDB(data_base_dir, train=True, download=True, transform=transform_particles)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

        # Testset
        testset = EM_5HDB(data_base_dir, train=False, download=True, transform=transform_particles)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=n_workers)

    else:
        raise ValueError('Dataset {} is not supported'.format(dataset_name))

    return train_loader, test_loader, image_shape


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

        if self.transform is not None:
            img = self.transform(img)
            img_rot = self.transform(img_rot)
        return img_rot, angle, img


class EM_5HDB(VisionDataset):
    particles_train = [str(x) + '_' for x in range(0, 16)]
    particles_test = [str(x) + '_' for x in range(16, 20)]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.processed_data_path = osp.join(root, '5HDB', 'ours_processed_{}.npy'.format('train' if train else 'test'))

        if not osp.exists(self.processed_data_path):
            particle_folders = self.particles_train if train is True else self.particles_test
            imgs, rots_deg, imgs_rot = self.process_data(root, particle_folders)
            np.save(self.processed_data_path, {'imgs': imgs, 'rots_deg': rots_deg, 'imgs_rot': imgs_rot})
        else:
            logger.info('{} exists'.format(self.processed_data_path))

        loaded_dict = np.load(self.processed_data_path, allow_pickle=True).item()

        self.imgs = loaded_dict['imgs']
        self.imgs_rot = loaded_dict['imgs_rot']
        self.targets = loaded_dict['rots_deg']

    @staticmethod
    def load_particle_images(base_dir: str):

        # Load image
        mrc_file = osp.join(base_dir, 'imgdata.mrc')
        imgs = mrc.imread(mrc_file)
        imgs = np.array(imgs)

        # Load rotation angle
        pkl = pickle.load(open(osp.join(base_dir, 'pardata.pkl'), "rb"))
        rot_rad = pkl['i']
        rot_deg = 180 * np.array(rot_rad) / np.pi
        return imgs, rot_deg

    def process_data(self, root: str, particle_folders: list) -> (np.ndarray, np.ndarray, np.ndarray):

        # Load particles data
        logger.info('Load particles data: {}'.format('train' if self.train else 'test'))
        imgs_rot_list, rot_deg_list = [], []
        for particle_folder in tqdm(particle_folders):
            imgs, rot_deg = self.load_particle_images(osp.join(root, '5HDB', particle_folder))

            # Normalize each particle between 0 and 255 like regular image
            imgs = 255 * (imgs - imgs.min()) / (imgs.max() - imgs.min())
            imgs = np.uint8(imgs)

            # Store
            imgs_rot_list.append(imgs)
            rot_deg_list.append(rot_deg)

        imgs_rot = np.concatenate(imgs_rot_list, axis=0)
        rots_deg = np.concatenate(rot_deg_list, axis=0)

        # Process data
        logger.info('Process data: {}'.format('train' if self.train else 'test'))
        img_list = []
        for img_rot, angle_deg in tqdm(zip(imgs_rot, rots_deg), total=len(imgs_rot)):
            # Rotate to canonic angle
            img = scipy.ndimage.rotate(img_rot, -angle_deg, mode='reflect')

            # de-noising
            img = cv2.fastNlMeansDenoising(img, None, 21, 2, 21)

            img_list.append(img)
        imgs = np.array(img_list)
        return imgs, rots_deg, imgs_rot

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_rot, angle_deg, img = self.imgs_rot[index], self.targets[index], self.imgs[index]

        # from angle between 0-360 to angle between -180 to 180
        if angle_deg > 180:
            angle_deg = -(360 - angle_deg)

        img = Image.fromarray(img, mode='P')
        img_rot = Image.fromarray(img_rot, mode='P')

        if self.transform is not None:
            img = self.transform(img)
            img_rot = self.transform(img_rot)

        angle_rad = np.pi * angle_deg / 180
        return img_rot, angle_rad, img

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = EM_5HDB('../data')
    print(dataset[0])

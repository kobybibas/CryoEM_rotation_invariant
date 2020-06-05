import logging
import os
import os.path as osp

import hydra
import pytorch_lightning as pl
import torch

from dataset_utils import get_dataset
from vae_lightning_utils import VAE

logger = logging.getLogger(__name__)


@hydra.main(config_path=osp.join('..', 'configs', 'vae_mnist.yaml'))
def train_vae(cfg):
    logger.info(f"Run config:\n{cfg.pretty()}")
    out_dir = os.getcwd()
    logger.info('Working directory {}'.format(out_dir))

    # Dataset
    train_loader, test_loader, image_shape = get_dataset(cfg.dataset, cfg.batch_size, cfg.num_workers)

    # Model definition
    vae_model = VAE(cfg, train_loader=train_loader, val_loader=test_loader, img_shape=image_shape)

    # Train
    trainer = pl.Trainer(early_stop_callback=False,
                         checkpoint_callback=False,
                         max_nb_epochs=cfg.num_epochs,
                         fast_dev_run=cfg.fast_dev_run,
                         gpus=[0] if torch.cuda.is_available() else 0)
    trainer.fit(vae_model)
    logger.info('Finished. Save to: {}'.format(os.getcwd()))

    # Save models
    models_save_dir = osp.join('..', '..', 'models')
    os.makedirs(models_save_dir, exist_ok=True)
    save_file = osp.join(models_save_dir, 'vae_{}_decoder.pth'.format(cfg.dataset))
    torch.save(vae_model.q_net.state_dict(), save_file)
    save_file = osp.join(models_save_dir, 'vae_{}_encoder.pth'.format(cfg.dataset))
    torch.save(vae_model.p_net.state_dict(), save_file)
    save_file = osp.join(models_save_dir, 'vae_{}_decoder_pickle.pth'.format(cfg.dataset))
    torch.save(vae_model.q_net, save_file)
    save_file = osp.join(models_save_dir, 'vae_{}_encoder_pickle.pth'.format(cfg.dataset))
    torch.save(vae_model.p_net, save_file)


if __name__ == "__main__":
    train_vae()

import logging
import os
import os.path as osp

import hydra
import pytorch_lightning as pl

from dataset_utils import get_dataset
from vae_lightning_utils import VAE

logger = logging.getLogger(__name__)


@hydra.main(config_path=osp.join('..', 'configs', 'vae_mnist.yaml'))
def train_vae(cfg):
    logger.info(f"Run config:\n{cfg.pretty()}")
    out_dir = os.getcwd()
    logger.info('Working directory {}'.format(out_dir))

    # Dataset
    train_loader, test_loader, image_shape = get_dataset(cfg.dataset, cfg.batch_size)

    # Model definition
    vae_model = VAE(cfg, train_loader=train_loader, val_loader=test_loader, image_shape=image_shape)

    # Train
    trainer = pl.Trainer(early_stop_callback=False,
                         max_nb_epochs=cfg.num_epochs,
                         fast_dev_run=cfg.fast_dev_run)
    trainer.fit(vae_model)
    logger.info('Finished. Save to: {}'.format(os.getcwd()))

    # torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    train_vae()

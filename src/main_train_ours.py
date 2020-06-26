import hydra
import logging
import os
import os.path as osp
import pytorch_lightning as pl
import torch

from dataset_utils import get_dataset
from ours_lightning_utils import Ours

logger = logging.getLogger(__name__)


@hydra.main(config_path=osp.join('..', 'configs', 'ours_mnist.yaml'))
def train_ours(cfg):
    logger.info(f"Run config:\n{cfg.pretty()}")
    out_dir = os.getcwd()
    logger.info('Working directory {}'.format(out_dir))
    logger.info('tensorboard: http://localhost:6006/')

    # To ensure reproducibility
    pl.seed_everything(123)

    # Dataset
    train_loader, test_loader, image_shape = get_dataset(cfg.dataset, cfg.batch_size, cfg.num_workers)

    # Model definition
    our_model = Ours(cfg, train_loader=train_loader, val_loader=test_loader, img_shape=image_shape)

    # Train
    trainer = pl.Trainer(early_stop_callback=False,
                         checkpoint_callback=False,
                         max_nb_epochs=cfg.num_epochs,
                         fast_dev_run=cfg.fast_dev_run,
                         progress_bar_refresh_rate=1,
                         gpus=[0] if torch.cuda.is_available() else 0)
    trainer.fit(our_model)
    logger.info('Finished. Save to: {}'.format(os.getcwd()))

    # Save models
    save_file = osp.join(os.getcwd(), 'our_{}_decoder.pth'.format(cfg.dataset))
    torch.save(our_model.decoder.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))
    save_file = osp.join(os.getcwd(), 'our_{}_encoder.pth'.format(cfg.dataset))
    torch.save(our_model.encoder.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))
    save_file = osp.join(os.getcwd(), 'our_{}_discriminator.pth'.format(cfg.dataset))
    torch.save(our_model.discriminator.state_dict(), save_file)
    logger.info('Saving model: {}'.format(save_file))


if __name__ == "__main__":
    train_ours()

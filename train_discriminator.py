import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger

from src.lightning_modules import LightningAttributeModel
from src.utils.utils import read_discriminator_configs
from src.dataset import get_dataloaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_name', default='dd_topics_baseline.yaml', type=str)
    parser.add_argument('-fast_dev_run', default=True, type=bool)
    parser.add_argument('-ddp', default='ddp_cpu', type=str, choices=['ddp', 'dp', 'ddp_cpu', 'none'])
    parser.add_argument('-use_comet', default=False, type=bool)

    parser_args = parser.parse_args()
    conf = read_discriminator_configs(parser_args.config_name)

    pl.seed_everything(conf.seed)

    model = LightningAttributeModel(conf)

    train_loader, valid_loader = get_dataloaders(
        model.tokenizer,
        conf.model,
        conf.batch_size,
        conf.num_workers,
        **conf['datasets']
    )

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )
    if 'logger' in conf and parser_args.use_comet is True:
        comet_logger = CometLogger(**conf['logger'], experiment_name=conf.experiment_name)
        logger = [logger, comet_logger]
    # TODO: write custom callback to load ckpt to comet

    ddp = None
    if parser_args.ddp != 'none':
        ddp = parser_args.ddp

    trainer = pl.Trainer(
        deterministic=True,
        fast_dev_run=parser_args.fast_dev_run,
        distributed_backend=ddp,
        logger=logger,
        **conf['trainer']
    )

    trainer.fit(model, train_dataloader=train_loader)


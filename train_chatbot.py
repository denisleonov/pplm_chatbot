import argparse
import os
#import logging
#logging.basicConfig(level=logging.INFO)

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger

from src.lightning_modules import LightningDoubleHeadsModel
from src.utils.utils import read_chatbot_configs
from src.dataset import get_dataloaders

from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_name', default='bart_baseline.yaml', type=str)
    parser.add_argument('-fast_dev_run', default=False, type=bool)
    parser.add_argument('-ddp', default='ddp', type=str, choices=['ddp', 'dp', 'ddp_cpu', 'none'])
    parser.add_argument('-use_comet', default=False, type=bool)
    parser_args = parser.parse_args()
    conf = read_chatbot_configs(parser_args.config_name)
    pl.seed_everything(conf.seed)

    model = LightningDoubleHeadsModel(conf)
    num_neg_samples = conf.num_neg_samples if conf.use_cls_head else 0
    
    print('Training model: ', model.model.model_name)
    print('Using classification head: ', conf.use_cls_head)
    
    train_loader, valid_loader = get_dataloaders(
        model.tokenizer,
        model.model.model_type,
        conf.batch_size,
        num_neg_samples,
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

    checkpoint_callback = ModelCheckpoint(
        filepath='./model_weights/{}.ckpt'.format(conf.experiment_name),
        save_top_k=2,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )


    ddp = None
    if parser_args.ddp != 'none':
        ddp = parser_args.ddp

    if conf.checkpoint_name:
        print('Using checkpoint: ', conf.checkpoint_name)
        trainer = pl.Trainer(resume_from_checkpoint=f'model_weights/{conf.checkpoint_name}')
    else:
        trainer = pl.Trainer(deterministic=True,
                            fast_dev_run=parser_args.fast_dev_run,
                            distributed_backend=ddp,
                            logger=logger,
                            checkpoint_callback=checkpoint_callback,
                            **conf['trainer']
                            )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)

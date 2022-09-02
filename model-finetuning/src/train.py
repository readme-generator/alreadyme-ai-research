import argparse
import os
import warnings
from typing import Optional

from lightning import MyLightningDataModule, MyLightningModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    config: DictConfig,
    resume_from: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    checkpoint = ModelCheckpoint(
        save_last=True,
        every_n_train_steps=config.train.save_every_n_train_steps,
    )
    logger = WandbLogger(
        project="alreadyme-model-finetuning",
        name=config.model.pretrained_model_name_or_path,
        id=resume_id,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision=16,
        amp_backend="apex",
        log_every_n_steps=config.train.log_every_n_steps,
        max_steps=config.optim.scheduler.num_training_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(
        MyLightningModule(config), MyLightningDataModule(config), ckpt_path=resume_from
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config, args.resume_from, args.resume_id)

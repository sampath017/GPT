from .data_module import ShakespearDataModule
from .module import ShakespeareModule

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L
import wandb

from .settings import (
    logs_path,
    data_path,
    num_words,
    context_size,
    batch_size,
    project_name,
    debug,
    epochs,
    limit_train_batches,
    limit_val_batches
)

dm = ShakespearDataModule(
    data_path, context_size=context_size, batch_size=batch_size)
model = ShakespeareModule(num_words, context_size=context_size)

callbacks = [ModelCheckpoint(
    filename="best", monitor="val_loss",
)]

logger = WandbLogger(project=project_name, save_dir=logs_path)
trainer = L.Trainer(
    max_epochs=epochs,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    fast_dev_run=debug,
    callbacks=callbacks,  # type: ignore
    logger=logger
)


def train():
    trainer.fit(model, dm)
    wandb.finish()

    return logger.version

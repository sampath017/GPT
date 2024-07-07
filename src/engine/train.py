from data_module import NamesDataModule
from module import NamesModule

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L
import wandb

from settings import logs_path, data_path, num_chars, block_size, batch_size, project_name

dm = NamesDataModule(data_path, block_size=block_size, batch_size=batch_size)
model = NamesModule(num_chars, block_size=3)

callbacks = [ModelCheckpoint(
    filename="[{epoch}] [{train_loss:.2f}] [{val_loss:.2f}]", monitor="val_loss",
    save_top_k=3
)]

logger = WandbLogger(project=project_name, save_dir=logs_path)
trainer = L.Trainer(
    max_epochs=50,
    fast_dev_run=False,
    callbacks=callbacks,  # type: ignore
    logger=logger
)


def train():
    trainer.fit(model, dm)
    wandb.finish()

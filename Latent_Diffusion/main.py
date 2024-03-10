"""Run this file to train the model"""
import datetime
import os

import lightning as L
import wandb
from archisound import ArchiSound
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import load
from torch.utils.data import DataLoader

from constants import (
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_DIRECTORY,
    TRAINING_CONFIG,
    WANDB_ENCODER_DECODER_PROJECT_NAME,
    RAW_MP3_DIR,
)
from Data_Processing import pre_processor
from lightning_diffusion import LitAudioEncoder


def check_num_workers(test_dataloader):
    total = 0
    print(next(iter(test_dataloader)).shape)
    for i in range(0, 20):
        start = datetime.datetime.now()
        next(iter(test_dataloader))
        total = total + (datetime.datetime.now() - start).total_seconds()

    print(f"Average: {total/20}")


def execute_training_pipeline(
    pre_process_data=False,
    train_split_prop=0.99,
    save_top_k_models=3,
    save_on_n_epochs=5,
    model_path=None,
):
    """
    Function for executing entire training pipeline
    """
    # Pipeline
    preprocessor = pre_processor.PreProcessor()
    if pre_process_data:
        preprocessor.preprocess(verbose=True)
    train_set, val_set = preprocessor.split_into_train_val(train_prop=train_split_prop)
    train_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )
    if not model_path:
        autoencoder = ArchiSound.from_pretrained(
            "dmae1d-ATC32-v3",
        )
    else:
        autoencoder = load(model_path)

    config_dict = TRAINING_CONFIG.copy()
    config_dict["Learning_Rate"] = LEARNING_RATE
    # Initializing Wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandblogger = WandbLogger(
        # set the wandb project where this run will be logged
        project=WANDB_ENCODER_DECODER_PROJECT_NAME,
        name=f"{WANDB_ENCODER_DECODER_PROJECT_NAME}-run-test",
        # track hyperparameters and run metadata
        config=config_dict,
    )
    lightning_model = LitAudioEncoder(model=autoencoder)
    config_dict["num_val_sample_steps"] = lightning_model.val_sample_steps

    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    check_point_callback = ModelCheckpoint(
        dirpath=MODEL_DIRECTORY,
        save_top_k=save_top_k_models,
        every_n_epochs=save_on_n_epochs,
        monitor="val_loss",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.01, patience=8, verbose=False, mode="min"
    )

    trainer = L.Trainer(
        **TRAINING_CONFIG,
        callbacks=[check_point_callback, early_stop_callback],
        logger=wandblogger,
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
    wandb.finish()


if __name__ == "__main__":
    execute_training_pipeline(pre_process_data=True)

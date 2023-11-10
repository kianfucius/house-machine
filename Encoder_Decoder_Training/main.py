import datetime

import lightning as L
from archisound import ArchiSound
from torch.utils.data import DataLoader

from constants import WANDB_ENCODER_DECODER_PROJECT_NAME, TRAINING_CONFIG
from Data_Processing import pre_processor
from lightning_torch import LitAudioEncoder


def check_num_workers(test_dataloader):
    total = 0
    print(next(iter(test_dataloader)).shape)
    for i in range(0, 20):
        start = datetime.datetime.now()
        next(iter(test_dataloader))
        total = total + (datetime.datetime.now() - start).total_seconds()

    print(f"Average: {total/20}")


if __name__ == "__main__":
    preprocessor = pre_processor.PreProcessor()
    preprocessor.preprocess(verbose=True)
    train_set, val_set = preprocessor.split_into_train_val(train_prop=0.7)
    train_dataloader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
    )

    autoencoder = ArchiSound.from_pretrained(
        "dmae1d-ATC32-v3",
    )
    lightning_model = LitAudioEncoder(
        model=autoencoder, project_name=WANDB_ENCODER_DECODER_PROJECT_NAME
    )
    trainer = L.Trainer(**TRAINING_CONFIG)
    trainer.fit(
        lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

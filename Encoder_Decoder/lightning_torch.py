import os
from typing import Any

import lightning as L
import torch.nn.functional as F
import torchaudio
from torch import randint
from torch.optim import AdamW, lr_scheduler

from constants import LEARNING_RATE, TRAINING_CONFIG, VAL_DIR


class LitAudioEncoder(L.LightningModule):
    """Torch Lightning Module for Audio Encoder-Decoder Model."""

    def __init__(
        self,
        model,
        val_sample_dir=VAL_DIR,
        num_saved_samples_per_val_step=1,
        num_validation_sample_steps=50,
        learning_rate_scheduler=None,
    ):
        super().__init__()
        self.lr_scheduler = learning_rate_scheduler
        self.val_sample_steps = num_validation_sample_steps
        self.val_dir = val_sample_dir
        self.model = model
        self.num_val_samples = num_saved_samples_per_val_step

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LEARNING_RATE)
        if not self.lr_scheduler:
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, cooldown=1, factor=0.5, mode="min"
            )
        lr_training_config = {
            # REQUIRED: The scheduler instance
            "scheduler": self.lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,  # <- I do this to make sure the validation loss and training is aligned.
            # Metric to to monitor for schedulers.
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_training_config}

    def validation_step(self, val_batch, batch_idx):
        encoded = self.model.encode(val_batch)
        decoded = self.model.decode(encoded, num_steps=self.val_sample_steps)
        loss = F.mse_loss(val_batch, decoded)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Generating and saving audio samples.
        random_index = randint(0, val_batch.shape[0], (self.num_val_samples,))
        print("Saving Compressed-Decompressed Examples into directory: " + self.val_dir)
        base_path = os.path.join(f"{self.val_dir}", f"epoch_{self.current_epoch}")
        val_batch = val_batch.to("cpu")
        decoded = decoded.to("cpu")
        for i in list(random_index):
            # Making directory if doesn't exist.
            sample_dir = os.path.join(base_path, f"Audio_Example_{i}")
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            torchaudio.save(
                os.path.join(sample_dir, "original.wav"),
                val_batch[i, :, :],
                sample_rate=48000,
                channels_first=True,
            )
            torchaudio.save(
                os.path.join(sample_dir, "reconstructed.wav"),
                decoded[i, :, :],
                sample_rate=48000,
                channels_first=True,
            )


def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError:
        if not os.path.isdir(dir_name):
            raise
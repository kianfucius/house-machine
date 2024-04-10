import os

import lightning as L
import torch
import torchaudio
from archisound import ArchiSound
from audio_diffusion_pytorch import VDiffusion
from constants import LEARNING_RATE, VAL_DIR
from Custom_Losses import CustomFrequencyLoss
from torch import nn, randint
from torch.optim import AdamW, lr_scheduler


class CustomDiffusionModel(VDiffusion):
    def __init__(self, base_diffusion: VDiffusion, custom_loss: nn.Module):
        super().__init__(base_diffusion.net, base_diffusion.sigma_distribution)
        self.custom_loss = custom_loss
        self.tuple_loss = False

    def extend_dim(self, x: torch.Tensor, dim: int):
        # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
        return x.view(*x.shape + (1,) * (dim - x.ndim))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = self.extend_dim(sigmas, dim=x.ndim)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)

        # Computing x_pred:
        x_pred = alphas * x_noisy - sigmas_batch * v_pred
        return self.custom_loss(
            v_pred,
            v_target,
            x_pred,
            x,
            alphas,
            sigmas_batch,
            tuple_loss=self.tuple_loss,
        )


class LitDiffusionAudioEncoder(L.LightningModule):
    """Torch Lightning Module for Audio Encoder-Decoder Model."""

    def __init__(
        self,
        model_path=None,
        val_sample_dir=VAL_DIR,
        num_saved_samples_per_val_step=3,
        num_validation_sample_steps=50,
        learning_rate_scheduler=None,
        loss_fn="custom",
        quantize_model=True,
        frequency_weight=0.001,
    ):
        """
        if loss_fn == mean: keep v-objective diffusion loss.

        if loss_fn == 'custom': overwrite forward method of diffusion element.

        alpha only used if loss function is custom.
        alpha is the frequency loss term weight.
        if quantize_model is true, then
        """
        super().__init__()
        self.lr_scheduler = learning_rate_scheduler
        self.val_sample_steps = num_validation_sample_steps
        self.val_dir = val_sample_dir
        self.num_val_samples = num_saved_samples_per_val_step
        self.frequency_weight = frequency_weight

        if not model_path:
            self.model = ArchiSound.from_pretrained(
                "dmae1d-ATC32-v3",
            )
        else:
            self.model = torch.load(model_path)

        if loss_fn == "custom":
            # Changing forward pass of diffusion model to incorporate custom loss function.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            custom_loss = CustomFrequencyLoss(frequency_weight=frequency_weight)
            self.model.model.diffusion = CustomDiffusionModel(
                self.model.model.diffusion, custom_loss
            ).to(device)

        if quantize_model:
            self.model = torch.quantization.quantize_dynamic(self.model)

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LEARNING_RATE)
        if not self.lr_scheduler:
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=50, cooldown=10, factor=0.5, mode="min"
            )
        lr_training_config = {
            # REQUIRED: The scheduler instance
            "scheduler": self.lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,  # Update the learning rate after every step.
            "monitor": "train_loss",  # Monitoring the training loss.
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_training_config}

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            # -------------------------------------------------------------
            print("Computing Batched generation")
            model_output = self.model.encode(val_batch)
            # with torch.autocast(device_type="cuda"):
            # model_output = model_output.to(dtype = torch.float)
            model_output = self.model.decode(
                model_output,
                num_steps=self.val_sample_steps,
            )
            # -------------------------------------------------------------
            print("Computing Validation Loss")
            # Enabling Tuple loss to examine both frequency and MSE loss.
            self.model.model.diffusion.tuple_loss = True
            diffusion_loss, frequency_loss = self.model(val_batch)
        # ---------------------------------------------------------------------
        self.log(
            "val_loss", diffusion_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "validation frequency loss",
            frequency_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Generating and saving audio samples.
        random_index = randint(0, val_batch.shape[0], (self.num_val_samples,))
        print("Saving Compressed-Decompressed Examples into directory: " + self.val_dir)
        base_path = os.path.join(f"{self.val_dir}", f"epoch_{self.current_epoch}")
        val_batch = val_batch.to("cpu")
        model_output = model_output.to("cpu")
        for i in list(random_index):
            # Making directory if doesn't exist.
            sample_dir = os.path.join(base_path, f"Audio_Example_{i}")
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            torchaudio.save(
                os.path.join(sample_dir, "original.wav"),
                val_batch[i, :, :],
                sample_rate=44100,
                channels_first=True,
            )
            torchaudio.save(
                os.path.join(sample_dir, "reconstructed.wav"),
                model_output[i, :, :],
                sample_rate=44100,
                channels_first=True,
            )
        # Disabling Tuple loss to only get final loss score.
        self.model.model.diffusion.tuple_loss = False

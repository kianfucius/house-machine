import torch
from torch import nn
from torch import randint
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F

from typing import List

import os
from constants import LEARNING_RATE, VAL_DIR

import lightning as L
import torchaudio
from audio_diffusion_pytorch import VDiffusion
from archisound import ArchiSound
from auraloss.freq import SumAndDifferenceSTFTLoss


class CustomDiffusionModel(VDiffusion):
    def __init__(self, base_diffusion:VDiffusion, custom_loss:nn.Module):
        super().__init__(base_diffusion.net, base_diffusion.sigma_distribution)
        self.custom_loss = custom_loss
        self.tuple_loss =False

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
        x_pred = alphas*x_noisy - sigmas*v_pred
        # Feeding everything into loss functions:

        if self.tuple_loss:
            return self.custom_loss(v_pred, 
                    v_target, 
                    x_pred,
                    x,
                    alphas,
                    sigmas,
                    tuple_loss = True)
        else:
            return self.custom_loss(v_pred, 
                    v_target, 
                    x_pred,
                    x,
                    alphas,
                    sigmas)


class CustomFrequencyLoss(nn.Module):
    """
    Class for custom pytorch loss with Custom frequency loss penalty term.
    """

    def __init__(self, alpha = 0.5,        
                fft_sizes: List[int] = [2048, 4096, 1024],
                hop_sizes: List[int] = [120, 240, 50],
                win_lengths: List[int] = [600, 1200, 240], 
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frequency_loss = SumAndDifferenceSTFTLoss(fft_sizes=fft_sizes,
                                                       hop_sizes = hop_sizes,
                                                       win_lengths=win_lengths)
        self.weighting =alpha
        

    def forward(self, 
                v_pred:torch.Tensor, 
                v_true:torch.Tensor, 
                x_pred:torch.Tensor, 
                x_true:torch.Tensor,
                alphas:torch.Tensor,
                sigmas:torch.Tensor,
                tuple_loss= False):
        
        mean_loss = F.mse_loss(v_pred,v_true) 
        frequency_loss = self.frequency_loss(x_true, x_pred)
        if tuple_loss:
            return mean_loss, frequency_loss*(1+ (alphas/sigmas)**2)
        else:
            return mean_loss + frequency_loss*(1+ (alphas/sigmas)**2)*self.weighting
    
class LitDiffusionAudioEncoder(L.LightningModule):
    """Torch Lightning Module for Audio Encoder-Decoder Model."""

    def __init__(
        self,
        model_path,
        val_sample_dir=VAL_DIR,
        num_saved_samples_per_val_step=3,
        num_validation_sample_steps=50,
        learning_rate_scheduler=None,
        loss_fn = 'custom',
        alpha = 0.5
    ):
        """
        if loss_fn == mean: keep v-objective diffusion loss.

        if loss_fn == 'custom': overwrite forward method of diffusion element.

        alpha only used if loss function is custom.
        alpha is the frequency loss term weight.
        """
        super().__init__()
        self.lr_scheduler = learning_rate_scheduler
        self.val_sample_steps = num_validation_sample_steps
        self.val_dir = val_sample_dir
        self.num_val_samples = num_saved_samples_per_val_step
        self.alpha = alpha

        if not model_path:
            self.model = ArchiSound.from_pretrained(
                "dmae1d-ATC32-v3",
            )
        else:
            self.model = torch.load(model_path)
        
        if loss_fn =='custom':
            # Changing forward pass of diffusion model to compute
            # Custom Loss function.
            custom_loss = CustomFrequencyLoss(alpha= alpha)
            self.model.diffusion = CustomDiffusionModel(self.model.diffusion, custom_loss)

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
        with torch.no_grad():
            print('Computing Batched generation')
            model_output = self.model.encode(val_batch)
            model_output = self.model.decode(model_output, num_steps=self.val_sample_steps)
            print('Computing Validation Loss')
            self.model.diffusion.tuple_loss = True
            diffusion_loss, frequency_loss = self.model(val_batch)

        self.log("val_loss", diffusion_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation frequency loss", frequency_loss, on_step=False, on_epoch=True, prog_bar=True)

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
                sample_rate=48000,
                channels_first=True,
            )
            torchaudio.save(
                os.path.join(sample_dir, "reconstructed.wav"),
                model_output[i, :, :],
                sample_rate=48000,
                channels_first=True,
            )

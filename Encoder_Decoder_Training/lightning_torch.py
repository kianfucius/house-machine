import os
import torchaudio
from torch.optim import AdamW, lr_scheduler 
from torch import randint
from typing import Any, Optional, Union
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion 
import lightning as L
import constants
import torch.functional as F

class LitAudioEncoder(L.LightningModule):
    def __init__(self, model, project_name, val_sample_dir= constants.VAL_DIR, 
                 num_saved_samples_per_val_step = 1,
                 num_validation_sample_steps =50):
        super().__init__()
        self.val_sample_steps = num_validation_sample_steps
        self.val_dir = val_sample_dir
        # Making directory if doesn't exist.
        make_dir(self.val_dir)
        self.model = model
        self.num_val_samples = num_saved_samples_per_val_step

        config_dict = constants.TRAINER_CONFIG
        config_dict['Learning_Rate'] = constants.LEARNING_RATE
        config_dict['num_val_sample_steps'] = self.val_sample_steps

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("train_loss", loss,prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=constants.LEARNING_RATE)
    
    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        return super().lr_scheduler_step(scheduler, metric)
    
    def validation_step(self,val_batch, batch_idx):
        wandb_logger = self.logger.experiment
        encoded = self.model.encode(val_batch)
        decoded = self.model.decode(encoded, num_steps = self.val_sample_steps)
        loss = F.mse_loss(val_batch,decoded)
        wandb_logger.log('Validation Loss', loss)
        
        # Generating and saving audio samples.
        random_index = randint(0, val_batch.shape[0], (self.num_val_samples,))
        print('Saving Compressed-Decompressed Examples into directory: ' + self.val_dir)
        base_path = os.path.join(self.val_dir + ':',self.current_epoch)
        for i in list(random_index):
            # Making directory if doesn't exist.
            sample_dir = os.path.join(base_path,'Audio_Example_1')
            torchaudio.save(os.path.join(sample_dir,'original.wav'),val_batch[i,:,:],
                                sample_rate=48000, channels_first=True)
            torchaudio.save(os.path.join(sample_dir, 'reconstructed.wav'),decoded[i,:,:],
                                sample_rate=48000, channels_first=True)
    

def make_dir(dir_name):
    try: 
        os.makedirs(dir_name)
    except OSError:
        if not os.path.isdir(dir_name):
            raise


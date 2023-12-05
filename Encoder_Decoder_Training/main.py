import datetime

import lightning as L
from archisound import ArchiSound
from torch.utils.data import DataLoader
from torch import load

from constants import WANDB_ENCODER_DECODER_PROJECT_NAME, TRAINING_CONFIG
from Data_Processing import pre_processor
from lightning_torch import LitAudioEncoder
from pytorch_lightning.callbacks import ModelCheckpoint


def check_num_workers(test_dataloader):
    total = 0
    print(next(iter(test_dataloader)).shape)
    for i in range(0, 20):
        start = datetime.datetime.now()
        next(iter(test_dataloader))
        total = total + (datetime.datetime.now() - start).total_seconds()

    print(f"Average: {total/20}")

def execute_training_pipeline(pre_process_data = False, 
                              train_split_prop = 0.99, 
                              save_top_k_models = 3,
                              save_on_n_epochs = 5,
                              model_path = None):
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
    if not model_path:
        autoencoder = ArchiSound.from_pretrained(
            "dmae1d-ATC32-v3",
        )
    else:
        autoencoder = load(model_path)
    lightning_model = LitAudioEncoder(model=autoencoder, project_name=WANDB_ENCODER_DECODER_PROJECT_NAME)
    check_point_callback = ModelCheckpoint(dirpath = 'Model_Directory',
                                           save_top_k=save_top_k_models,
                                           every_n_epochs= save_on_n_epochs,
                                           monitor='val_loss')
    trainer = L.Trainer(**TRAINING_CONFIG,callbacks = [check_point_callback])
    trainer.fit(
        lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

if __name__ == "__main__":
    execute_training_pipeline(pre_process_data = True)
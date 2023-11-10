import os
RAW_MP3_DIR = os.path.join('Data','Unprocessed_songs')
ENCODER_DECODER_PROCESSED_DIR = os.path.join('Data','Chunked_songs')
VAL_DIR = os.path.join('Data','Val_Samples')
WANDB_ENCODER_DECODER_PROJECT_NAME = 'Lofi-Encoder'

TRAINING_CONFIG = {'accelerator':"gpu",
  'devices':1,
  'precision':"16-mixed",
  'accumulate_grad_batches':2,
  'max_steps':1e10,
  'check_val_every_n_epoch':2,
  'max_time':"00:24:00:00",
  'enable_checkpointing':True,
  'gradient_clip_val':1.5,
  'limit_train_batches':30,
  'profiler':"advanced"}

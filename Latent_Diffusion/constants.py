"""Various constants used throughout the project."""

import os

RAW_MP3_DIR = os.path.join("..", "..", "data")
ENCODER_DECODER_PROCESSED_DIR = os.path.join("data", "chunked_songs")
VAL_DIR = os.path.join("data", "val_samples")
WANDB_ENCODER_DECODER_PROJECT_NAME = "lofi-encoder"
LEARNING_RATE = 1e-5
TRAINING_CONFIG = {
    "accelerator": "gpu",
    "devices": 1,
    "precision": "16-mixed",
    "accumulate_grad_batches": 20,
    "enable_checkpointing": True,
    "gradient_clip_val": 1.5,
    "profiler": "advanced",
    "limit_val_batches": 2,
    "max_epochs": 30,
    "log_every_n_steps": 1,
    "val_check_interval": 0.5,
}
BATCH_SIZE = 8
MODEL_DIRECTORY = "model"

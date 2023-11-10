"""Various constants used throughout the project."""
import os

RAW_MP3_DIR = os.path.join("data", "unprocessed_songs")
ENCODER_DECODER_PROCESSED_DIR = os.path.join("data", "chunked_songs")
VAL_DIR = os.path.join("data", "val_samples")
WANDB_ENCODER_DECODER_PROJECT_NAME = "lofi-encoder"
LEARNING_RATE = 1e-4
TRAINING_CONFIG = {
    "accelerator": "gpu",
    "devices": 1,
    "precision": "16-mixed",
    "accumulate_grad_batches": 2,
    "max_steps": 1e10,
    "check_val_every_n_epoch": 2,
    "max_time": "00:24:00:00",
    "enable_checkpointing": True,
    "gradient_clip_val": 1.5,
    "limit_train_batches": 30,
    "profiler": "advanced",
    "limit_val_batches": 1.0,
}

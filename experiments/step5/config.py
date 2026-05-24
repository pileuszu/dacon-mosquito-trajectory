import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "open"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Metadata
METADATA_PATH = BASE_DIR / "step5" / "metadata.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 1024 # Keep high for GPU utilization
EPOCHS = 150
LEARNING_RATE = 0.0005
PATIENCE = 20
TARGET_SCALE = 100.0 # Work in cm

# Multi-modal GMM Params
NUM_MODES = 6 # Predict 6 possible future paths
INPUT_SIZE = 12 # Same as step4
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Data Augmentation
AUG_POSITION_SHIFT = 0.005 # 5mm random shift
AUG_NOISE_STD = 0.001     # 1mm gaussian noise

# Logging
WANDB_PROJECT = "mosquito-trajectory"
WANDB_RUN_NAME = "step5-multimodal-gmm"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

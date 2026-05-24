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
METADATA_PATH = BASE_DIR / "step6" / "metadata.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 1024
EPOCHS = 150
LEARNING_RATE = 0.0005 # Peak LR for OneCycle
PATIENCE = 30
TARGET_SCALE = 1.0

# Discretization Params
GRID_RANGE = 0.05
BIN_SIZE = 0.005
NUM_BINS = int((GRID_RANGE * 2) / BIN_SIZE) + 1
CENTER_BIN = NUM_BINS // 2

# Model Params (Upgraded)
INPUT_SIZE = 18 # Extended with Bio-Physics Invariants
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Logging
WANDB_PROJECT = "mosquito-trajectory"
WANDB_RUN_NAME = "step6-tadic-physics-onecycle"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

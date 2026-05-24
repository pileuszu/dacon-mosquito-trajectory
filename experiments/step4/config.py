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
METADATA_PATH = BASE_DIR / "step4" / "metadata.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 2048 # Maximize GPU utilization
EPOCHS = 150
LEARNING_RATE = 0.001 # Scaled up for larger batch size
PATIENCE = 20
TARGET_SCALE = 100.0 # Convert meters to centimeters for better gradient flow

# Model Params
INPUT_SIZE = 12 # xyz(3) + vel(3) + accel(3) + curvature(1) + angular_vel(2)
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.15

# Biomechanical Constraints (from EDA/Report)
MAX_VELOCITY = 3.0 # m/s
MAX_ACCEL = 40.0   # m/s^2

# Logging
WANDB_PROJECT = "mosquito-trajectory"
WANDB_RUN_NAME = "step4-biomech-equivariant"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

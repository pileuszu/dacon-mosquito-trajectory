import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "open"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Metadata and Splits
METADATA_PATH = BASE_DIR / "step2" / "metadata.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15 # Increased patience for better convergence

# Model Params
INPUT_SIZE = 9 # x,y,z + vx,vy,vz + ax,ay,az
HIDDEN_SIZE = 128
NUM_LAYERS = 3

# Logging
WANDB_PROJECT = "mosquito-trajectory"
WANDB_RUN_NAME = "step2-lstm-physics-robust"

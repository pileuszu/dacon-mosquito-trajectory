import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "open"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "step8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metadata
METADATA_PATH = BASE_DIR / "step8" / "metadata.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 512
EPOCHS = 150
LEARNING_RATE = 0.0005
PATIENCE = 30

# Candidate Params
NUM_CANDIDATES = 27
R_HIT = 0.01 # 1cm

# Model Params
INPUT_SIZE = 18 # Using our best bio-physics features
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Logging
WANDB_PROJECT = "mosquito-trajectory"
WANDB_RUN_NAME = "step8-physics-candidate-selector"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

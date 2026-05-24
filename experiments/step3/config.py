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
METADATA_PATH = BASE_DIR / "step3" / "metadata.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 64 # Transformer benefits from smaller/moderate batches for sharper gradients
EPOCHS = 150
LEARNING_RATE = 0.0005 # Transformer often needs a lower initial LR
PATIENCE = 20

# Model Params (Transformer specific)
INPUT_SIZE = 9
D_MODEL = 64 # Dimension of Transformer model
NHEAD = 4    # Number of Attention Heads
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# Logging
WANDB_PROJECT = "mosquito-trajectory"
WANDB_RUN_NAME = "step3-transformer-physics"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

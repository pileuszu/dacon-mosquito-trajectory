import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Enforce VENV usage
def _ensure_venv():
    project_root = Path(__file__).parent.parent
    if sys.platform == 'win32':
        venv_python = (project_root / 'venv' / 'Scripts' / 'python.exe').resolve()
    else:
        venv_python = (project_root / 'venv' / 'bin' / 'python').resolve()
    
    curr_python = Path(sys.executable).resolve()
    
    if venv_python.exists() and curr_python != venv_python:
        print(f"Switching to venv: {venv_python}", flush=True)
        code = subprocess.call([str(venv_python)] + sys.argv)
        sys.exit(code)

_ensure_venv()

# Load environment variables from .env file
load_dotenv(override=True)

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Current Step Info
STEP_NAME = Path(__file__).parent.name  # 'step1'

# Data Paths
DATA_DIR = PROJECT_ROOT / 'data' / 'open'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
TRAIN_LABELS_PATH = DATA_DIR / 'train_labels.csv'
SAMPLE_SUBMISSION_PATH = DATA_DIR / 'sample_submission.csv'

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / STEP_NAME
METADATA_PATH = PROJECT_ROOT / STEP_NAME / 'metadata.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metrics Configuration
R_HIT = float(os.getenv("R_HIT", 0.01))
R_HIT_3CM = float(os.getenv("R_HIT_3CM", 0.03))
R_HIT_5CM = float(os.getenv("R_HIT_5CM", 0.05))

# Data Split Configuration
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# WandB Configuration
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mosquito-trajectory")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "pileuszu-study")
WANDB_GROUP = STEP_NAME  # Automatically 'step1'
WANDB_NAME = f"{STEP_NAME}-lstm-residual"
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2

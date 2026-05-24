import os
import sys
import subprocess
from pathlib import Path

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
        # Use subprocess.call and sys.exit to completely replace the process
        code = subprocess.call([str(venv_python)] + sys.argv)
        sys.exit(code)

_ensure_venv()

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
PROJECT_ROOT = Path(__file__).parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / 'data' / 'open'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
TRAIN_LABELS_PATH = DATA_DIR / 'train_labels.csv'
SAMPLE_SUBMISSION_PATH = DATA_DIR / 'sample_submission.csv'

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'step0'
METADATA_PATH = PROJECT_ROOT / 'step0' / 'metadata.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metrics Configuration
R_HIT = float(os.getenv("R_HIT", 0.01))    # 1cm
R_HIT_3CM = float(os.getenv("R_HIT_3CM", 0.03))
R_HIT_5CM = float(os.getenv("R_HIT_5CM", 0.05))

# Data Split Configuration
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# WandB Configuration
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mosquito-trajectory")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_NAME = os.getenv("WANDB_NAME", "baseline-cv")
WANDB_GROUP = os.getenv("WANDB_GROUP", "step0")
WANDB_JOB_TYPE = os.getenv("WANDB_JOB_TYPE", "evaluation")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)

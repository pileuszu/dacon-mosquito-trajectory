import numpy as np
import pandas as pd
import wandb
from datetime import datetime
from config import (
    TRAIN_LABELS_PATH, R_HIT, R_HIT_3CM, R_HIT_5CM,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME, WANDB_GROUP, WANDB_JOB_TYPE
)
from model import ConstantVelocityModel
from dataset import get_splits

def calculate_train_loss(true_xyz, pred_xyz):
    """
    Calculate loss-like metrics for the training split.
    Focus on convergence-related values like MSE/MAE.
    """
    errors = true_xyz - pred_xyz
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    
    return {
        "train/loss": mse,
        "train/mae": mae,
    }

def calculate_test_metrics(true_xyz, pred_xyz):
    """
    Calculate performance metrics for the test/eval split.
    Focus on competition-specific metrics.
    """
    errors = true_xyz - pred_xyz
    distance = np.linalg.norm(errors, axis=1)
    
    metrics = {
        "test/hit_rate@1cm": np.mean(distance <= R_HIT),
        "test/hit_rate@3cm": np.mean(distance <= R_HIT_3CM),
        "test/hit_rate@5cm": np.mean(distance <= R_HIT_5CM),
        "test/mean_dist_error": np.mean(distance),
        "test/max_dist_error": np.max(distance),
        "test/rmse": np.sqrt(np.mean(distance**2)),
    }
    return metrics

def main():
    # 1. Setup
    train_files, test_files = get_splits()
    train_labels = pd.read_csv(TRAIN_LABELS_PATH)
    model = ConstantVelocityModel()
    
    # 2. WandB Initialization
    run_name = f"{WANDB_NAME}-{datetime.now().strftime('%m%d-%H%M%S')}"
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        group=WANDB_GROUP,
        job_type="train_eval_pipeline",
        config={
            "model": "ConstantVelocity",
            "train_size": len(train_files),
            "test_size": len(test_files),
        }
    )
    
    # 3. Evaluate Train Split (Track Loss)
    print(f"\nTracking 'Loss' on training split ({len(train_files)} samples)...")
    train_pred = model.predict_batch(train_files)
    train_eval = train_labels.merge(train_pred, on='id', suffixes=('_true', '_pred'))
    train_loss = calculate_train_loss(
        train_eval[['x_true', 'y_true', 'z_true']].to_numpy(),
        train_eval[['x_pred', 'y_pred', 'z_pred']].to_numpy()
    )
    
    # 4. Evaluate Test Split (Track Metrics)
    print(f"\nEvaluating metrics on test split ({len(test_files)} samples)...")
    test_pred = model.predict_batch(test_files)
    test_eval = train_labels.merge(test_pred, on='id', suffixes=('_true', '_pred'))
    test_metrics = calculate_test_metrics(
        test_eval[['x_true', 'y_true', 'z_true']].to_numpy(),
        test_eval[['x_pred', 'y_pred', 'z_pred']].to_numpy()
    )
    
    # 5. Log to WandB
    wandb.log({**train_loss, **test_metrics})
    
    print(f"\nTrain Loss: {train_loss['train/loss']:.6f}")
    print(f"Test Hit@1cm: {test_metrics['test/hit_rate@1cm']:.4f}")
    
    run.finish()
    print("\nTraining and Evaluation records completed in WandB.")

if __name__ == "__main__":
    main()

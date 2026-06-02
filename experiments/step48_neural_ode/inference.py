import os
import sys
import json
import csv
import torch
import numpy as np
import argparse
import traceback
from pathlib import Path

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import SimpleNeuralODEModel, extract_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--out-dir", type=str, default="outputs/step48_neural_ode", help="Output directory")
    args = parser.parse_args()
    
    # Send start notification
    send_discord_notification(
        None,
        f"🚀 Started: [Step 48 inference.py] Generating final Test predictions using 5-fold Neural ODE models..."
    )
    
    try:
        # Determine device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # Make output directory
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load compiled test data
        data_dir = Path("step48_neural_ode/data")
        test_x = np.load(data_dir / "test_x.npy")
        
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        print(f"Loaded test data. test_x: {test_x.shape}, test_ids: {len(test_ids)}")
        
        models_dir = Path("step48_neural_ode/models")
        
        # We need to run inference across all available models
        # Find which folds have trained models
        available_folds = []
        for f in range(5):
            model_path = models_dir / f"model_fold_{f}.pt"
            stats_path = models_dir / f"stats_fold_{f}.pt"
            if model_path.exists() and stats_path.exists():
                available_folds.append(f)
                
        if not available_folds:
            raise FileNotFoundError("No trained Neural ODE models or stats files found in step48_neural_ode/models.")
            
        print(f"Found trained models for folds: {[f+1 for f in available_folds]}")
        
        test_tensor = torch.tensor(test_x, dtype=torch.float32)
        batch_size = 256
        
        fold_predictions = []
        
        for fold in available_folds:
            print(f"Running inference for Fold {fold + 1}...")
            
            # Load statistics
            stats = torch.load(models_dir / f"stats_fold_{fold}.pt", map_location=device)
            mean_stats = stats["mean"].to(device)
            std_stats = stats["std"].to(device)
            
            # Rebuild model
            model = SimpleNeuralODEModel(input_dim=24, latent_dim=64).to(device)
            model.load_state_dict(torch.load(models_dir / f"model_fold_{fold}.pt", map_location=device))
            model.eval()
            
            preds_list = []
            
            with torch.no_grad():
                for i in range(0, len(test_tensor), batch_size):
                    Xb = test_tensor[i:i+batch_size].to(device)
                    ft, df, plt_, tht, _, _, _, Rt, spt, _, _ = extract_features(Xb, mean_stats, std_stats)
                    pred = model(ft, df, plt_, tht, spt, Rt)
                    preds_list.append(pred.cpu().numpy())
                    
            fold_pred = np.concatenate(preds_list, axis=0)
            fold_predictions.append(fold_pred)
            
        # Ensemble predictions by averaging coordinates
        final_preds = np.mean(fold_predictions, axis=0)
        
        # Write submission file
        sub_path = out_dir / "submission_ode.csv"
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, row in zip(test_ids, final_preds):
                writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])
                
        # Calculate displacement statistics
        p0 = test_x[:, -1]
        disp_cm = np.linalg.norm(final_preds - p0, axis=1).mean() * 100
        
        success_msg = (
            f"✅ Finished: [Step 48 inference.py] Test predictions successfully generated using {len(available_folds)}-fold ensemble!\n"
            f"Submission file saved at: `{sub_path}`\n"
            f"Average Displacement: **{disp_cm:.4f} cm**"
        )
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 48 inference.py] Inference ERROR:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()

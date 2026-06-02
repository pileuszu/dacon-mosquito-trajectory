import os
import sys
import json
import csv
import torch
import numpy as np
from pathlib import Path
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import extract_features, SotaCFMTrajectoryModel

def main():
    send_discord_notification(
        None,
        "🚀 Started: [Step 56 inference.py] Generating CFM model test predictions..."
    )
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        data_dir = Path("step56_flow_matching/data")
        models_dir = Path("step56_flow_matching/models")
        out_dir = Path("outputs/step56_flow_matching")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load test data and candidates
        test_x = np.load(data_dir / "test_x.npy")
        test_candidates = np.load(data_dir / "test_candidates.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        print(f"Loaded test input. Shape: {test_x.shape}, candidates: {test_candidates.shape}")
        
        # 2. Perform 5-fold ensemble predictions with 2-step Euler integration
        print("Running 5-fold ensemble inference...")
        test_tensor = torch.tensor(test_x, dtype=torch.float32)
        
        ensemble_preds_2step = np.zeros((len(test_x), 3))
        ensemble_preds_1step = np.zeros((len(test_x), 3))
        
        for fold in range(5):
            print(f"  Predicting with Fold {fold + 1}...")
            # Load normalization stats
            stats = torch.load(models_dir / f"stats_fold_{fold}.pt")
            tr_mean = stats["mean"]
            tr_std = stats["std"]
            
            # Load CFM model
            model = SotaCFMTrajectoryModel(feature_dim=38, latent_dim=64, num_candidates=36).to(device)
            model.load_state_dict(torch.load(models_dir / f"model_fold_{fold}.pt", map_location=device))
            model.eval()
            
            with torch.no_grad():
                test_tensor_dev = test_tensor.to(device)
                ft_te, df_te, _, _, _ = extract_features(test_tensor_dev, tr_mean.to(device), tr_std.to(device))
                candidates_te = torch.tensor(test_candidates, dtype=torch.float32).to(device)
                
                # Predict
                fold_pred_2step = model.predict(ft_te, df_te, candidates_te, steps=2)
                fold_pred_1step = model.predict(ft_te, df_te, candidates_te, steps=1)
                
                ensemble_preds_2step += fold_pred_2step.cpu().numpy() / 5.0
                ensemble_preds_1step += fold_pred_1step.cpu().numpy() / 5.0
                
        # 3. Save predicted coordinates as numpy arrays for blending
        np.save(data_dir / "test_preds_cfm_2step.npy", ensemble_preds_2step)
        np.save(data_dir / "test_preds_cfm_1step.npy", ensemble_preds_1step)
        print("Saved test predictions as numpy arrays.")
        
        # 4. Save single-model CFM submission csv for fallback/submission
        sub_path = out_dir / "submission_cfm_2step.csv"
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, row in zip(test_ids, ensemble_preds_2step):
                writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])
                
        p_last_te = test_x[:, -1]
        disp_orig = np.linalg.norm(ensemble_preds_2step - p_last_te, axis=1).mean() * 100
        
        success_msg = (
            f"✅ Finished: [Step 56 inference.py] CFM model test predictions generated successfully!\n"
            f"Submission saved to: `{sub_path}`\n"
            f"Average Displacement: **{disp_orig:.4f} cm**"
        )
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 56 inference.py] Inference failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()

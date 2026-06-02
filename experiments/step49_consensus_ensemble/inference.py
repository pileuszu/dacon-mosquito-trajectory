import os
import sys
import json
import csv
import numpy as np
import argparse
import traceback
from pathlib import Path

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from train_automl import classify_regimes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="outputs/step49_consensus_ensemble", help="Output directory")
    args = parser.parse_args()
    
    # Send start notification
    send_discord_notification(
        None,
        f"🚀 Started: [Step 49 inference.py] Generating blended final consensus Test predictions..."
    )
    
    try:
        data_dir = Path("step49_consensus_ensemble/data")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load optimal weights configuration
        weights_path = data_dir / "optimal_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"Optimal weights config not found at {weights_path}. Please run train_automl.py first.")
            
        with open(weights_path, "r") as f:
            optimal_config = json.load(f)
            
        blend_type = optimal_config["blend_type"]
        w_cruising = optimal_config["w_cruising"]
        w_gliding = optimal_config["w_gliding"]
        w_steering = optimal_config["w_steering"]
        
        print(f"Optimal weights configuration loaded:")
        print(f"  - Blend Type: {blend_type.upper()}")
        print(f"  - w_cruising (Step 47 weight): {w_cruising:.2f}")
        print(f"  - w_gliding (Step 47 weight): {w_gliding:.2f}")
        print(f"  - w_steering (Step 47 weight): {w_steering:.2f}")
        
        # Load test inputs
        test_x = np.load(data_dir / "test_x.npy")
        
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        # Select predictions based on blend type
        # Step 47 has soft and argmax submissions
        if blend_type == "soft":
            s47_test = np.load(data_dir / "step47_test.npy") # s47 soft test predictions
        else:
            # Load argmax submission for Step 47
            s47_argmax_csv = Path("outputs/step47_physics_ladder/submission_argmax.csv")
            if not s47_argmax_csv.exists():
                raise FileNotFoundError(f"Step 47 argmax csv not found at {s47_argmax_csv}")
            import pandas as pd
            df_s47 = pd.read_csv(s47_argmax_csv)
            s47_dict = {row["id"]: np.array([row["x"], row["y"], row["z"]]) for _, row in df_s47.iterrows()}
            s47_test = np.array([s47_dict[sid] for sid in test_ids])
            
        s48_test = np.load(data_dir / "step48_test.npy")
        
        print(f"Loaded test predictions. s47_test: {s47_test.shape}, s48_test: {s48_test.shape}")
        
        # Classify test regimes
        print("Classifying test trajectories into 3 physical regimes...")
        regimes, _, _ = classify_regimes(test_x)
        
        # Perform coordinate blending
        final_preds = np.zeros_like(s47_test)
        
        cruising_idx = np.where(regimes == 0)[0]
        gliding_idx = np.where(regimes == 1)[0]
        steering_idx = np.where(regimes == 2)[0]
        
        final_preds[cruising_idx] = w_cruising * s47_test[cruising_idx] + (1 - w_cruising) * s48_test[cruising_idx]
        final_preds[gliding_idx] = w_gliding * s47_test[gliding_idx] + (1 - w_gliding) * s48_test[gliding_idx]
        final_preds[steering_idx] = w_steering * s47_test[steering_idx] + (1 - w_steering) * s48_test[steering_idx]
        
        # Write blended submission
        sub_path = out_dir / "submission_ensemble.csv"
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, row in zip(test_ids, final_preds):
                writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])
                
        # Calculate displacement statistics
        p0 = test_x[:, -1]
        disp_cm = np.linalg.norm(final_preds - p0, axis=1).mean() * 100
        
        success_msg = (
            f"✅ Finished: [Step 49 inference.py] Blended final consensus Test predictions successfully generated!\n"
            f"Submission file saved at: `{sub_path}`\n"
            f"Average Displacement: **{disp_cm:.4f} cm**"
        )
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 49 inference.py] Blended inference failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()

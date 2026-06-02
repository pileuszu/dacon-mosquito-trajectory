import os
import sys
import json
import csv
import numpy as np
from pathlib import Path

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification

EPS = 1e-8

def classify_regimes(X):
    N = X.shape[0]
    last_v = (X[:, -1] - X[:, -2]) / 0.04  # (N, 3)
    speeds = np.linalg.norm(last_v, axis=1) # (N,)
    
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    last_a = (last_v - prev_v) / 0.04 # (N, 3)
    
    t_dir = last_v / (speeds[:, None] + EPS)
    acc_par_scalar = np.sum(last_a * t_dir, axis=1)
    acc_perp = last_a - acc_par_scalar[:, None] * t_dir
    acc_perp_norm = np.linalg.norm(acc_perp, axis=1)
    
    cross_prod = np.cross(last_v, last_a, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curvature = cross_norm / (speeds ** 3 + EPS)
    
    is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
    
    regimes = np.zeros(N, dtype=int)
    for i in range(N):
        if is_steering[i]:
            regimes[i] = 2  # Steering
        elif speeds[i] <= 0.50:
            regimes[i] = 0  # Cruising
        else:
            regimes[i] = 1  # Gliding
            
    return regimes

def main():
    send_discord_notification(
        None,
        "🚀 Started: [Step 51 inference.py] Generating final Test predictions using Consensus Blending v2..."
    )
    
    try:
        data_dir = Path("step51_consensus_blend_v2/data")
        out_dir = Path("outputs/step51_consensus_blend_v2")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load optimal weights
        weights_path = data_dir / "optimal_weights.json"
        with open(weights_path, "r") as f:
            config = json.load(f)
            
        blend_type = config["blend_type"]
        w_cruising = config["w_cruising"]
        w_gliding = config["w_gliding"]
        w_steering = config["w_steering"]
        
        print("Optimal Blending Configuration:")
        print(f"  - Blend Type: {blend_type.upper()}")
        print(f"  - w_cruising: {w_cruising}")
        print(f"  - w_gliding: {w_gliding}")
        print(f"  - w_steering: {w_steering}")
        
        # 2. Load datasets
        test_x = np.load(data_dir / "test_x.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        print(f"Loaded test input. Shape: {test_x.shape}")
        
        # 3. Load model predictions
        if blend_type == "soft":
            s47_test = np.load(data_dir / "step47_test_soft.npy")
        else:
            s47_test = np.load(data_dir / "step47_test_argmax.npy")
            
        s50_test = np.load(data_dir / "step50_test.npy")
        
        print(f"Loaded Step 47 test prediction. Shape: {s47_test.shape}")
        print(f"Loaded Step 50 test prediction. Shape: {s50_test.shape}")
        
        # 4. Classify test regimes
        print("Classifying test trajectories...")
        regimes = classify_regimes(test_x)
        
        # 5. Blend predictions
        final_preds = np.zeros_like(s50_test)
        
        for i in range(len(test_x)):
            r = regimes[i]
            if r == 0:
                final_preds[i] = w_cruising * s47_test[i] + (1 - w_cruising) * s50_test[i]
            elif r == 1:
                final_preds[i] = w_gliding * s47_test[i] + (1 - w_gliding) * s50_test[i]
            else:
                final_preds[i] = w_steering * s47_test[i] + (1 - w_steering) * s50_test[i]
                
        # 6. Write final blended submission file
        sub_path = out_dir / "submission_frenet_ode_blend.csv"
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, row in zip(test_ids, final_preds):
                writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])
                
        # Calculate statistics
        p0 = test_x[:, -1]
        disp_cm = np.linalg.norm(final_preds - p0, axis=1).mean() * 100
        
        # Compare with baseline
        s47_disp_cm = np.linalg.norm(s47_test - p0, axis=1).mean() * 100
        s50_disp_cm = np.linalg.norm(s50_test - p0, axis=1).mean() * 100
        
        success_msg = (
            f"✅ Finished: [Step 51 inference.py] Blended test predictions successfully generated!\n"
            f"Submission file saved at: `{sub_path}`\n"
            f"Average Displacement stats:\n"
            f"  - Step 47 baseline: {s47_disp_cm:.4f} cm\n"
            f"  - Step 50 Frenet ODE baseline: {s50_disp_cm:.4f} cm\n"
            f"  - Step 51 Blended Coordinate: **{disp_cm:.4f} cm**"
        )
        print(success_msg)
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 51 inference.py] Blended inference failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()

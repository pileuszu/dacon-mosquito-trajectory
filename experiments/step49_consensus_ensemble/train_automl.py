import os
import sys
import json
import numpy as np
import argparse
import traceback
from pathlib import Path

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification

EPS = 1e-8

def classify_regimes(X):
    # X shape: (N, 11, 3)
    N = X.shape[0]
    
    # Compute terminal speed (between step 10 and 11)
    # dt = 40ms = 0.04s per step
    last_v = (X[:, -1] - X[:, -2]) / 0.04  # (N, 3)
    speeds = np.linalg.norm(last_v, axis=1) # (N,)
    
    # Compute curvature or acceleration to identify turning (Steering)
    # prev_v is speed vector from step 9 to 10
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    
    # centripetal acceleration (approx last acceleration perpendicular component)
    last_a = (last_v - prev_v) / 0.04 # (N, 3)
    
    t_dir = last_v / (speeds[:, None] + EPS)
    acc_par_scalar = np.sum(last_a * t_dir, axis=1, keepdims=True)
    acc_perp = last_a - acc_par_scalar * t_dir
    acc_perp_norm = np.linalg.norm(acc_perp, axis=1)
    
    # Cross product for curvature calculation
    cross_prod = np.cross(last_v, last_a, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curvature = cross_norm / (speeds ** 3 + EPS)
    
    # Routing criteria based on physical distributions:
    # Steering (Turning): curvature > 6.0 OR perpendicular acceleration > 1.8 m/s^2
    is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
    
    # Speed threshold for slow vs fast: 0.50 m/s (50 cm/s)
    # If not steering, split by speed
    regimes = np.zeros(N, dtype=int)  # 0: Cruising, 1: Gliding, 2: Steering
    
    for i in range(N):
        if is_steering[i]:
            regimes[i] = 2  # Steering (Turning)
        elif speeds[i] <= 0.50:
            regimes[i] = 0  # Cruising (Slow-Straight)
        else:
            regimes[i] = 1  # Gliding (Fast-Straight)
            
    return regimes, speeds, curvature

def grid_search_blending(s47_preds, s48_preds, train_y, regimes):
    N = len(train_y)
    best_hr = 0.0
    best_weights = {}
    
    # 3-Regime indices
    cruising_idx = np.where(regimes == 0)[0]
    gliding_idx = np.where(regimes == 1)[0]
    steering_idx = np.where(regimes == 2)[0]
    
    # We will search weights for Step 47 model (w) and Neural ODE (1-w)
    # search space: 0.0 to 1.0 in 0.05 increments
    weights_space = np.linspace(0.0, 1.0, 21)
    
    print(f"Sample distribution by regime:")
    print(f"  - Cruising (Slow-Straight): {len(cruising_idx)} samples ({len(cruising_idx)/N*100:.2f}%)")
    print(f"  - Gliding (Fast-Straight): {len(gliding_idx)} samples ({len(gliding_idx)/N*100:.2f}%)")
    print(f"  - Steering (Turning): {len(steering_idx)} samples ({len(steering_idx)/N*100:.2f}%)")
    
    # Find best weight for Cruising
    best_w_cruising = 0.0
    best_hr_cruising = 0.0
    for w in weights_space:
        preds = w * s47_preds[cruising_idx] + (1 - w) * s48_preds[cruising_idx]
        dists = np.linalg.norm(preds - train_y[cruising_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_cruising:
            best_hr_cruising = hr
            best_w_cruising = w
            
    # Find best weight for Gliding
    best_w_gliding = 0.0
    best_hr_gliding = 0.0
    for w in weights_space:
        preds = w * s47_preds[gliding_idx] + (1 - w) * s48_preds[gliding_idx]
        dists = np.linalg.norm(preds - train_y[gliding_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_gliding:
            best_hr_gliding = hr
            best_w_gliding = w
            
    # Find best weight for Steering
    best_w_steering = 0.0
    best_hr_steering = 0.0
    for w in weights_space:
        preds = w * s47_preds[steering_idx] + (1 - w) * s48_preds[steering_idx]
        dists = np.linalg.norm(preds - train_y[steering_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_steering:
            best_hr_steering = hr
            best_w_steering = w
            
    # Reconstruct final predictions
    final_preds = np.zeros_like(train_y)
    final_preds[cruising_idx] = best_w_cruising * s47_preds[cruising_idx] + (1 - best_w_cruising) * s48_preds[cruising_idx]
    final_preds[gliding_idx] = best_w_gliding * s47_preds[gliding_idx] + (1 - best_w_gliding) * s48_preds[gliding_idx]
    final_preds[steering_idx] = best_w_steering * s47_preds[steering_idx] + (1 - best_w_steering) * s48_preds[steering_idx]
    
    final_dists = np.linalg.norm(final_preds - train_y, axis=1)
    overall_hr = np.mean(final_dists <= 0.01)
    
    best_weights = {
        "w_cruising": float(best_w_cruising),
        "w_gliding": float(best_w_gliding),
        "w_steering": float(best_w_steering),
        "hr_cruising": float(best_hr_cruising),
        "hr_gliding": float(best_hr_gliding),
        "hr_steering": float(best_hr_steering),
        "overall_hr": float(overall_hr)
    }
    
    return best_weights, final_preds

def main():
    # Send start notification
    send_discord_notification(
        None,
        f"🚀 Started: [Step 49 train_automl.py] Optimizing Consensus Blending weight coefficients..."
    )
    
    try:
        data_dir = Path("step49_consensus_ensemble/data")
        
        # Load compiled inputs
        train_x = np.load(data_dir / "train_x.npy")
        train_y = np.load(data_dir / "train_y.npy")
        
        # Load OOF predictions
        s47_soft = np.load(data_dir / "step47_oof_soft.npy")
        s47_argmax = np.load(data_dir / "step47_oof_argmax.npy")
        s48_oof = np.load(data_dir / "step48_oof.npy")
        
        print(f"Loaded datasets. s47_soft: {s47_soft.shape}, s48_oof: {s48_oof.shape}, train_y: {train_y.shape}")
        
        # Classify regimes
        print("Classifying trajectories into 3 physical regimes...")
        regimes, _, _ = classify_regimes(train_x)
        
        # Baseline single model scores
        s47_soft_hr = np.mean(np.linalg.norm(s47_soft - train_y, axis=1) <= 0.01)
        s47_argmax_hr = np.mean(np.linalg.norm(s47_argmax - train_y, axis=1) <= 0.01)
        s48_hr = np.mean(np.linalg.norm(s48_oof - train_y, axis=1) <= 0.01)
        
        print(f"Single model baselines:")
        print(f"  - Step 47 OOF (Soft): {s47_soft_hr*100:.3f}%")
        print(f"  - Step 47 OOF (Argmax): {s47_argmax_hr*100:.3f}%")
        print(f"  - Step 48 OOF (Neural ODE): {s48_hr*100:.3f}%")
        
        # Search 1: Blending Step 47 Soft & Step 48 ODE
        print("\nOptimizing blend between Step 47 Soft & Step 48 ODE...")
        soft_weights, soft_blended_preds = grid_search_blending(s47_soft, s48_oof, train_y, regimes)
        
        # Search 2: Blending Step 47 Argmax & Step 48 ODE
        print("\nOptimizing blend between Step 47 Argmax & Step 48 ODE...")
        argmax_weights, argmax_blended_preds = grid_search_blending(s47_argmax, s48_oof, train_y, regimes)
        
        # Determine the absolute best blend
        if soft_weights["overall_hr"] >= argmax_weights["overall_hr"]:
            best_blend_type = "soft"
            optimal_config = soft_weights
            optimal_preds = soft_blended_preds
            print(f"\n🏆 Best Blend: Step 47 (Soft) + Step 48 (ODE)")
        else:
            best_blend_type = "argmax"
            optimal_config = argmax_weights
            optimal_preds = argmax_blended_preds
            print(f"\n🏆 Best Blend: Step 47 (Argmax) + Step 48 (ODE)")
            
        optimal_config["blend_type"] = best_blend_type
        
        print("\n==========================================")
        print(f"Optimal Consensus Blending Configuration:")
        print(f"  - Blend Type: {optimal_config['blend_type'].upper()}")
        print(f"  - w_cruising (Step 47 weight): {optimal_config['w_cruising']:.2f} (Val Hit: {optimal_config['hr_cruising']*100:.2f}%)")
        print(f"  - w_gliding (Step 47 weight): {optimal_config['w_gliding']:.2f} (Val Hit: {optimal_config['hr_gliding']*100:.2f}%)")
        print(f"  - w_steering (Step 47 weight): {optimal_config['w_steering']:.2f} (Val Hit: {optimal_config['hr_steering']*100:.2f}%)")
        print(f"  - Blended OOF Hit Rate@1cm: {optimal_config['overall_hr']*100:.3f}%")
        print("==========================================")
        
        # Save optimal weights configuration
        weights_path = data_dir / "optimal_weights.json"
        with open(weights_path, "w") as f:
            json.dump(optimal_config, f, indent=4)
        print(f"Saved optimal weights to {weights_path}")
        
        # Save blended OOF predictions
        np.save(data_dir / "blended_oof_predictions.npy", optimal_preds)
        
        # Send finish notification
        send_discord_notification(
            None,
            f"✅ Finished: [Step 49 train_automl.py] Weight Optimization Completed Successfully!\n"
            f"Best Blend Type: **{optimal_config['blend_type'].upper()}**\n"
            f"Optimal Weights: w_cruising={optimal_config['w_cruising']:.2f}, w_gliding={optimal_config['w_gliding']:.2f}, w_steering={optimal_config['w_steering']:.2f}\n"
            f"Overall Blended OOF Hit@1cm: **{optimal_config['overall_hr']*100:.3f}%**\n"
            f"Improvement over Step 47 Soft OOF: **+{(optimal_config['overall_hr'] - s47_soft_hr)*100:.3f}%p**"
        )
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 49 train_automl.py] Weight optimization failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()

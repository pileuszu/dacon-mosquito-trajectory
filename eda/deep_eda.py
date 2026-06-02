import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from step49_consensus_ensemble.train_automl import classify_regimes

def main():
    data_dir = Path("step49_consensus_ensemble/data")
    artifact_dir = Path("C:/Users/pilla/.gemini/antigravity/brain/b0f5a587-e592-4e6d-953e-191b7ce04510")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for EDA visualization...")
    # Load raw coordinates & targets
    train_x = np.load(data_dir / "train_x.npy")
    train_y = np.load(data_dir / "train_y.npy")
    
    # Load model OOF predictions
    s47_oof = np.load(data_dir / "step47_oof_soft.npy")
    s48_oof = np.load(data_dir / "step48_oof.npy")
    blended_oof = np.load(data_dir / "blended_oof_predictions.npy")
    
    # 1. Classify regimes and get statistics
    regimes, speeds, curvature = classify_regimes(train_x)
    
    # Calculate error distances
    s47_errors = np.linalg.norm(s47_oof - train_y, axis=1) * 100 # in cm
    s48_errors = np.linalg.norm(s48_oof - train_y, axis=1) * 100 # in cm
    blended_errors = np.linalg.norm(blended_oof - train_y, axis=1) * 100 # in cm
    
    print("Generating Chart 1: Regime Speed-Curvature Scatter Plot...")
    # -------------------------------------------------------------
    # Chart 1: Regime Distribution (Speed vs Curvature)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    colors = ['#1f77b4', '#2ca02c', '#d62728'] # HSL/hex custom colors
    labels = ['Cruising (Slow-Straight)', 'Gliding (Fast-Straight)', 'Steering (Turning)']
    
    for r in range(3):
        mask = regimes == r
        # Use log scale for curvature since it has extreme outliers
        plt.scatter(speeds[mask], np.log1p(curvature[mask]), c=colors[r], label=labels[r], alpha=0.6, s=15, edgecolors='none')
        
    # Draw boundary thresholds
    plt.axvline(x=0.50, color='#e08214', linestyle='--', linewidth=1.5, label='Speed Threshold (0.50 m/s)')
    plt.axhline(y=np.log1p(6.0), color='#762a83', linestyle='--', linewidth=1.5, label='Curvature Threshold (log(6+1))')
    
    plt.title("Physical Flight Regimes of Mosquito Trajectories", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Velocity magnitude (m/s)", fontsize=11)
    plt.ylabel("Log-Curvature log(1 + Curvature)", fontsize=11)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, loc='upper right')
    plt.tight_layout()
    chart1_path = artifact_dir / "regime_scatter.png"
    plt.savefig(chart1_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Chart 1 to {chart1_path}")
    
    print("Generating Chart 2: Error Distribution KDE comparison...")
    # -------------------------------------------------------------
    # Chart 2: Model Error KDE Comparison across Regimes
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=150)
    regime_names = ["Cruising (Slow-Straight)", "Gliding (Fast-Straight)", "Steering (Turning)"]
    
    for r in range(3):
        mask = regimes == r
        ax = axes[r]
        
        # Plot KDE for Step 47, Step 48, and Blended
        # To avoid dependency on seaborn, we can use histogram outline or standard box plots
        # Boxplot is cleaner and more robust
        data_to_plot = [s47_errors[mask], s48_errors[mask], blended_errors[mask]]
        
        box = ax.boxplot(data_to_plot, patch_artist=True, labels=['Step 47', 'Step 48 (ODE)', 'Step 49 (Blend)'], sym='')
        
        # Colors
        colors_box = ['#a1d99b', '#9ecae1', '#fdae6b']
        for patch, color in zip(box['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            
        ax.set_title(regime_names[r], fontsize=12, fontweight='bold')
        ax.set_ylabel("Error Distance (cm)" if r == 0 else "")
        # Draw 1cm boundary line
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.2, label='1.0cm Hit Threshold')
        if r == 0:
            ax.legend()
            
    plt.suptitle("Model Prediction Error Distribution Comparison per Regime", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart2_path = artifact_dir / "error_comparison.png"
    plt.savefig(chart2_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Chart 2 to {chart2_path}")
    
    print("Generating Chart 3: Trajectory Solver Path Comparison...")
    # -------------------------------------------------------------
    # Chart 3: 3D Trajectory Path Comparison
    # -------------------------------------------------------------
    # Pick a high-curvature turning (Steering) sample with distinct predictions
    # Filter for Steering and where s48 had a large error (>1.5cm) but s47 was closer
    steering_mask = (regimes == 2) & (s48_errors > 1.5) & (s47_errors < 1.0)
    candidates = np.where(steering_mask)[0]
    
    if len(candidates) > 0:
        idx = candidates[0]
        print(f"Selecting trajectory index {idx} for 3D path comparison.")
        
        history = train_x[idx] # (11, 3)
        target = train_y[idx] # (3,)
        pred_s47 = s47_oof[idx] # (3,)
        pred_s48 = s48_oof[idx] # (3,)
        pred_blend = blended_oof[idx] # (3,)
        
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot past history (last 5 points for clarity, or all 11)
        ax.plot(history[:, 0], history[:, 1], history[:, 2], 'ko-', label='Past Trajectory (History)', linewidth=2, markersize=5)
        
        # Plot target
        ax.scatter(target[0], target[1], target[2], color='red', s=100, label='True Future Position (Target)', marker='*', zorder=5)
        
        # Plot predictions
        ax.scatter(pred_s47[0], pred_s47[1], pred_s47[2], color='#2ca02c', s=80, label='Step 47 (Local Frame MLP)', marker='o', zorder=4)
        ax.scatter(pred_s48[0], pred_s48[1], pred_s48[2], color='#1f77b4', s=80, label='Step 48 (Neural ODE)', marker='^', zorder=4)
        ax.scatter(pred_blend[0], pred_blend[1], pred_blend[2], color='#ff7f0e', s=80, label='Step 49 (Consensus Blend)', marker='s', zorder=4)
        
        # Draw lines from last point to target/predictions
        last_pt = history[-1]
        ax.plot([last_pt[0], target[0]], [last_pt[1], target[1]], [last_pt[2], target[2]], 'r--', alpha=0.5)
        ax.plot([last_pt[0], pred_s47[0]], [last_pt[1], pred_s47[1]], [last_pt[2], pred_s47[2]], 'g--', alpha=0.5)
        ax.plot([last_pt[0], pred_s48[0]], [last_pt[1], pred_s48[1]], [last_pt[2], pred_s48[2]], 'b--', alpha=0.5)
        
        # Draw 1.0cm sphere boundary around target for visual reference
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs = target[0] + 0.01 * np.cos(u) * np.sin(v)
        ys = target[1] + 0.01 * np.sin(u) * np.sin(v)
        zs = target[2] + 0.01 * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.08, label='1.0cm Hit Boundary')
        
        ax.set_title(f"3D Trajectory & Solver Paths (Turning Case - Index {idx})", fontsize=13, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(frameon=True, facecolor='white', shadow=True)
        
        # Save plot
        chart3_path = artifact_dir / "trajectory_rk4_path.png"
        plt.savefig(chart3_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Chart 3 to {chart3_path}")
    else:
        print("⚠️ No turning candidates met the criteria for Chart 3 visual comparison.")
        
    print("All deep EDA charts successfully generated.")

if __name__ == "__main__":
    main()

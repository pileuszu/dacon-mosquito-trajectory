import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step1.config import TRAIN_LABELS_PATH

def load_full_trajectory(uid, train_dir, labels_df):
    # Load history
    history_path = os.path.join(train_dir, f"{uid}.csv")
    hist_df = pd.read_csv(history_path)
    
    # Load target
    target_row = labels_df[labels_df['id'] == uid]
    target_df = pd.DataFrame({
        'id': [uid],
        'x': [target_row['x'].values[0]],
        'y': [target_row['y'].values[0]],
        'z': [target_row['z'].values[0]]
    })
    
    # Combine
    full_df = pd.concat([hist_df, target_df], ignore_index=True)
    # Add simple time index (0.04s intervals)
    full_df['Time'] = np.arange(len(full_df)) * 0.04
    return full_df

def main():
    train_dir = "data/open/train"
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    unique_ids = labels_df['id'].unique()
    
    # Create images directory if not exists
    os.makedirs('eda/images', exist_ok=True)
    
    print(f"Analyzing {len(unique_ids)} trajectories...")
    
    # 1. 3D Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    samples = np.random.choice(unique_ids, 5, replace=False)
    
    all_velocities = []
    all_accels = []
    
    for uid in samples:
        traj = load_full_trajectory(uid, train_dir, labels_df)
        ax.plot(traj['x'], traj['y'], traj['z'], marker='o', markersize=3, label=f'ID: {uid}')
        ax.scatter(traj['x'].iloc[-1], traj['y'].iloc[-1], traj['z'].iloc[-1], color='red', s=100, edgecolors='black')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Sample Mosquito Trajectories (Blue: History, Red: Target)')
    ax.legend()
    plt.savefig('eda/images/01_trajectories_3d.png')
    print("Saved: eda/images/01_trajectories_3d.png")
    
    # 2. Physics Analysis
    print("Calculating physics stats on 100 samples...")
    physics_samples = np.random.choice(unique_ids, 100, replace=False)
    for uid in physics_samples:
        traj = load_full_trajectory(uid, train_dir, labels_df)
        dx = traj['x'].diff(); dy = traj['y'].diff(); dz = traj['z'].diff()
        dt = traj['Time'].diff()
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        vel = dist / dt
        all_velocities.extend(vel.dropna().values)
        accel = vel.diff() / dt
        all_accels.extend(accel.dropna().values)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(all_velocities, kde=True, color='blue')
    plt.title('Velocity Distribution (m/s)')
    plt.subplot(1, 2, 2)
    sns.histplot(all_accels, kde=True, color='green')
    plt.title('Acceleration Distribution (m/s²)')
    plt.tight_layout()
    plt.savefig('eda/images/01_physics_dist.png')
    print("Saved: eda/images/01_physics_dist.png")
    
    stats = {
        "Avg Velocity": np.mean(all_velocities),
        "Max Velocity": np.max(all_velocities),
        "95th Pctl Velocity": np.quantile(all_velocities, 0.95),
        "Avg Accel": np.mean(all_accels),
        "Max Accel": np.max(all_accels)
    }
    
    # Write report
    with open('eda/01_basic_analysis.md', 'w') as f:
        f.write("# 01. Preliminary Physics Analysis\n\n")
        f.write("## Physics Statistics (100 samples)\n")
        for k, v in stats.items():
            f.write(f"- **{k}**: {v:.4f}\n")
        f.write("\n## Visualizations\n")
        f.write("### 3D Trajectories\n")
        f.write("![3D Trajectories](images/01_trajectories_3d.png)\n\n")
        f.write("### Physics Distributions\n")
        f.write("![Physics Dist](images/01_physics_dist.png)\n")

    print("\nEDA Report generated: eda/01_basic_analysis.md")

if __name__ == "__main__":
    main()

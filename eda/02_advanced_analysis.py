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
    history_path = os.path.join(train_dir, f"{uid}.csv")
    hist_df = pd.read_csv(history_path)
    target_row = labels_df[labels_df['id'] == uid]
    target_df = pd.DataFrame({
        'id': [uid],
        'x': [target_row['x'].values[0]],
        'y': [target_row['y'].values[0]],
        'z': [target_row['z'].values[0]]
    })
    full_df = pd.concat([hist_df, target_df], ignore_index=True)
    full_df['Time'] = np.arange(len(full_df)) * 0.04
    return full_df

def analyze_cv_baseline(uid, train_dir, labels_df):
    traj = load_full_trajectory(uid, train_dir, labels_df)
    p1 = traj.iloc[-3][['x', 'y', 'z']].values
    p2 = traj.iloc[-2][['x', 'y', 'z']].values
    target = traj.iloc[-1][['x', 'y', 'z']].values
    cv_pred = p2 + (p2 - p1)
    error = np.linalg.norm(target - cv_pred)
    return error

def analyze_turning_angles(uid, train_dir, labels_df):
    traj = load_full_trajectory(uid, train_dir, labels_df)
    coords = traj[['x', 'y', 'z']].values
    vectors = np.diff(coords, axis=0)
    angles = []
    for i in range(len(vectors)-1):
        v1 = vectors[i]; v2 = vectors[i+1]
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_theta = np.dot(v1, v2) / (n1 * n2)
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            angles.append(angle)
    return angles

def main():
    train_dir = "data/open/train"
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    unique_ids = labels_df['id'].unique()
    os.makedirs('eda/images', exist_ok=True)
    
    print("Running Advanced EDA...")
    samples = np.random.choice(unique_ids, 200, replace=False)
    cv_errors = []; all_angles = []
    for uid in samples:
        cv_errors.append(analyze_cv_baseline(uid, train_dir, labels_df))
        all_angles.extend(analyze_turning_angles(uid, train_dir, labels_df))
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(cv_errors, kde=True, color='purple')
    plt.axvline(np.mean(cv_errors), color='red', linestyle='--', label=f'Mean: {np.mean(cv_errors):.4f}')
    plt.title('CV Baseline Error (m)')
    plt.subplot(1, 2, 2)
    sns.histplot(all_angles, kde=True, color='orange')
    plt.title('Turning Angles (degrees)')
    plt.tight_layout()
    plt.savefig('eda/images/02_advanced_analysis.png')
    print("Saved: eda/images/02_advanced_analysis.png")

    with open('eda/02_advanced_analysis.md', 'w') as f:
        f.write("# 02. Advanced Maneuverability & Baseline Analysis\n\n")
        f.write("### 1. CV Baseline Performance\n")
        f.write(f"- **Average CV Error**: {np.mean(cv_errors):.4f} m\n")
        f.write(f"- **R-Hit@1cm Potential**: {(np.array(cv_errors) < 0.01).mean()*100:.2f}%\n\n")
        f.write("### 2. Maneuverability Analysis\n")
        f.write(f"- **Average Turning Angle**: {np.mean(all_angles):.2f}°\n")
        f.write(f"- **90th Percentile Angle**: {np.quantile(all_angles, 0.90):.2f}°\n\n")
        f.write("### 3. Visualizations\n")
        f.write("![Advanced Analysis](images/02_advanced_analysis.png)\n")

    print(f"Report generated at eda/02_advanced_analysis.md")

if __name__ == "__main__":
    main()

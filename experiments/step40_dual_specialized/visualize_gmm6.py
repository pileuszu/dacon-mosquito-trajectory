import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
import sys
import os

sys.path.append(os.getcwd())
from step40_dual_specialized.prepare_data import extract_context_features, CLUSTER_FEATURES, REGIME_MAPPING

def main():
    print("=== Step 40: GMM-6 Outlier (95%) Visualizer ===")
    
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv").set_index('id')
    sample_ids = labels_df.index.unique()
    
    # Load GMM-6 scaler and model
    models_dir = Path("step39_six_regime/models")
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "gmm_model.pkl", "rb") as f:
        gmm = pickle.load(f)
        
    print("Extracting features and predicting GMM-6 probabilities...")
    traj_ids = []
    features_list = []
    
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        ctx = extract_context_features(xyz)
        
        feat_vector = [ctx[feat] for feat in CLUSTER_FEATURES]
        features_list.append(feat_vector + [ctx["ctx_speed"], ctx["smooth_curv_w5"], ctx["ctx_lat_accel"]])
        traj_ids.append(fid)
        
    features_arr = np.array(features_list, dtype=np.float32)
    X_train = features_arr[:, :len(CLUSTER_FEATURES)]
    
    # Scale and predict
    X_scaled = scaler.transform(X_train)
    probs = gmm.predict_proba(X_scaled)
    clusters = gmm.predict(X_scaled)
    max_probs = probs.max(axis=1)
    
    # Save statistics and DataFrame
    df_analysis = pd.DataFrame({
        "id": traj_ids,
        "cluster": clusters,
        "max_prob": max_probs,
        "speed": features_arr[:, -3], # ctx_speed
        "curv": features_arr[:, -2],  # smooth_curv_w5
        "lat_acc": features_arr[:, -1] # ctx_lat_accel
    })
    
    # Exclude under 95% probability as outliers
    df_analysis["is_clear"] = df_analysis["max_prob"] >= 0.95
    num_clear = df_analysis["is_clear"].sum()
    num_outliers = len(df_analysis) - num_clear
    
    print(f"\n--- 95% Outlier Threshold Stats ---")
    print(f"Total Trajectories: {len(df_analysis)}")
    print(f"Clear (>= 95%): {num_clear} ({num_clear/len(df_analysis):.2%})")
    print(f"Outliers (< 95%): {num_outliers} ({num_outliers/len(df_analysis):.2%})")
    
    # Regime breakdown
    print("\nRegime breakdown for Clear trajectories:")
    for cluster_id, name in REGIME_MAPPING.items():
        c_clear = df_analysis[(df_analysis["cluster"] == cluster_id) & df_analysis["is_clear"]]
        c_outliers = df_analysis[(df_analysis["cluster"] == cluster_id) & ~df_analysis["is_clear"]]
        print(f"  * {name}: Clear = {len(c_clear)} (Outliers = {len(c_outliers)})")
        
    # Beautiful Visual Scatter Plot with PCA
    print("\nPerforming PCA for 2D projection...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_analysis["pca1"] = X_pca[:, 0]
    df_analysis["pca2"] = X_pca[:, 1]
    
    print("\nPlotting GMM-6 scatter comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    
    # HSL-tailored premium colors for 6 clusters
    colors = {
        0: '#3498db', # Blue
        1: '#1abc9c', # Teal
        2: '#e67e22', # Orange
        3: '#2ecc71', # Green
        4: '#e74c3c', # Red
        5: '#9b59b6'  # Purple
    }
    
    regime_labels = {
        0: "Fast-Straight-Low (0)",
        1: "Slow-Moderate-Turning (1)",
        2: "Fast-Moderate-Turning (2)",
        3: "Fast-Straight-High (3)",
        4: "Fast-Extreme-Turning (4)",
        5: "Slow-Extreme-Turning (5)"
    }
    
    # 1. Subplot 1: Speed vs Curvature
    ax = axes[0]
    outliers_df = df_analysis[~df_analysis["is_clear"]]
    ax.scatter(
        outliers_df["speed"] * 100, # cm/s
        outliers_df["curv"],
        c='#bdc3c7',
        alpha=0.15,
        s=12,
        label="Ambiguous Outliers (<95% prob)"
    )
    
    for cluster_id in range(6):
        c_df = df_analysis[(df_analysis["cluster"] == cluster_id) & df_analysis["is_clear"]]
        ax.scatter(
            c_df["speed"] * 100, # cm/s
            c_df["curv"],
            c=colors[cluster_id],
            alpha=0.85,
            s=22,
            label=regime_labels[cluster_id]
        )
        
    ax.set_yscale('log')
    ax.set_title("Speed vs. Curvature (Log Scale)\n(Clear >= 95% Confidence vs. Grey Outliers)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Speed (cm/s)", fontsize=11)
    ax.set_ylabel("Curvature (log)", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='gray')
    
    # 2. Subplot 2: Speed vs Lateral Acceleration
    ax = axes[1]
    ax.scatter(
        outliers_df["speed"] * 100,
        outliers_df["lat_acc"] * 100, # cm/s^2
        c='#bdc3c7',
        alpha=0.15,
        s=12
    )
    
    for cluster_id in range(6):
        c_df = df_analysis[(df_analysis["cluster"] == cluster_id) & df_analysis["is_clear"]]
        ax.scatter(
            c_df["speed"] * 100,
            c_df["lat_acc"] * 100,
            c=colors[cluster_id],
            alpha=0.85,
            s=22
        )
        
    ax.set_title("Speed vs. Lateral Acceleration\n(Clear >= 95% Confidence vs. Grey Outliers)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Speed (cm/s)", fontsize=11)
    ax.set_ylabel("Lateral Acceleration (cm/s²)", fontsize=11)
    ax.grid(True, ls=":", alpha=0.5)
    
    # 3. Subplot 3: PCA 2D Feature Space
    ax = axes[2]
    ax.scatter(
        outliers_df["pca1"],
        outliers_df["pca2"],
        c='#bdc3c7',
        alpha=0.15,
        s=12
    )
    
    for cluster_id in range(6):
        c_df = df_analysis[(df_analysis["cluster"] == cluster_id) & df_analysis["is_clear"]]
        ax.scatter(
            c_df["pca1"],
            c_df["pca2"],
            c=colors[cluster_id],
            alpha=0.85,
            s=22
        )
        
    ax.set_title("Mathematically Optimal 8D Feature Space (PCA PC1 vs PC2)\n(Clear >= 95% Confidence vs. Grey Outliers)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Principal Component 2", fontsize=11)
    ax.grid(True, ls=":", alpha=0.5)
    
    plt.suptitle(f"Mosquito Trajectory Prediction: GMM-6 Clustering & 95% Boundary Outlier Filtering\nTotal Trajectories = 10,000 | Clear (>=95%) = {num_clear} ({num_clear/len(df_analysis):.2%}) | Outliers (<95%) = {num_outliers}", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save image
    brain_dir = Path("C:/Users/pilla/.gemini/antigravity-ide/brain/f6c70bbe-a99c-48e2-b0c6-bcb2f3002879")
    img_path = brain_dir / "gmm6_outlier_95_distribution.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"Beautiful scatter plot saved to {img_path}")
    
    # Save the dataframe of is_clear to a pickle file for prepare_data.py to consume
    with open(brain_dir / "gmm6_clear_mask_95.pkl", "wb") as f:
        pickle.dump(df_analysis[["id", "is_clear"]].set_index("id")["is_clear"].to_dict(), f)
    print("Saved gmm6_clear_mask_95.pkl mapping to artifacts.")

if __name__ == "__main__":
    main()

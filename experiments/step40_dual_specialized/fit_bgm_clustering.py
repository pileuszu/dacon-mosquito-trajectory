import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import os

sys.path.append(os.getcwd())
from step40_dual_specialized.prepare_data import extract_context_features, CLUSTER_FEATURES

def main():
    print("=== Step 40: Bayesian Gaussian Mixture (BGM) Automatic Clustering ===")
    
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv").set_index('id')
    sample_ids = labels_df.index.unique()
    
    print("Extracting features for 10,000 trajectories...")
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
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Fit Bayesian Gaussian Mixture (BGM) with a high upper bound of components (e.g. 12)
    max_components = 12
    print(f"Fitting BayesianGaussianMixture with max_components={max_components} (Dirichlet Process Prior)...")
    bgm = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.01,  # Lower prior values force sparser components (fewer active clusters)
        max_iter=300,
        random_state=42,
        verbose=1
    )
    bgm.fit(X_scaled)
    
    # Filter active components (weight > 1%)
    weights = bgm.weights_
    active_mask = weights > 0.01
    active_indices = np.where(active_mask)[0]
    num_active = len(active_indices)
    
    print(f"\n--- BGM Component Weights ---")
    for i in range(max_components):
        marker = "⭐ [ACTIVE]" if active_mask[i] else "[INACTIVE]"
        print(f"  * Component {i:2d}: Weight = {weights[i]:.4f} {marker}")
        
    print(f"\nTotal Active Clusters Found: {num_active}")
    
    # Save the fitted BGM model and scaler
    bgm_models_dir = Path("step40_dual_specialized/models/bgm")
    bgm_models_dir.mkdir(parents=True, exist_ok=True)
    with open(bgm_models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(bgm_models_dir / "bgm_model.pkl", "wb") as f:
        pickle.dump(bgm, f)
    print(f"Saved BGM models to {bgm_models_dir}")
    
    # Predict clusters and soft probabilities
    clusters = bgm.predict(X_scaled)
    probs = bgm.predict_proba(X_scaled)
    max_probs = probs.max(axis=1)
    
    # Store predictions in a DataFrame
    df_bgm = pd.DataFrame({
        "id": traj_ids,
        "cluster": clusters,
        "max_prob": max_probs,
        "speed": features_arr[:, -3],
        "curv": features_arr[:, -2],
        "lat_acc": features_arr[:, -1]
    })
    
    # Filter clear vs ambiguous (using a strict 95% threshold for reporting boundary ambiguity)
    df_bgm["is_clear"] = df_bgm["max_prob"] >= 0.95
    num_clear = df_bgm["is_clear"].sum()
    num_outliers = len(df_bgm) - num_clear
    print(f"\nBoundary Ambiguity (95% threshold): Clear = {num_clear} ({num_clear/len(df_bgm):.2%}), Outliers = {num_outliers} ({num_outliers/len(df_bgm):.2%})")
    
    # Print cluster centroids in physical space
    print("\nActive Cluster Centroids in Physical Space:")
    centroids_dict = {}
    for idx in active_indices:
        c_df = df_bgm[df_bgm["cluster"] == idx]
        mean_speed = c_df["speed"].mean() * 100 # cm/s
        mean_curv = c_df["curv"].mean()
        mean_lat_acc = c_df["lat_acc"].mean() * 100 # cm/s^2
        
        print(f"  * Cluster {idx:2d} (N={len(c_df)}, Weight={weights[idx]:.2%}): Speed = {mean_speed:.2f} cm/s, Curv = {mean_curv:.2f}, LatAcc = {mean_lat_acc:.2f} cm/s²")
        centroids_dict[idx] = {
            "size": len(c_df),
            "weight": weights[idx],
            "speed": mean_speed,
            "curv": mean_curv,
            "lat_acc": mean_lat_acc
        }
        
    # Fit PCA for 2D Projection
    print("\nComputing PCA for optimal 2D projection...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_bgm["pca1"] = X_pca[:, 0]
    df_bgm["pca2"] = X_pca[:, 1]
    
    # Save the PCA-transformed coordinate data to generate a report later
    brain_dir = Path("C:/Users/pilla/.gemini/antigravity-ide/brain/f6c70bbe-a99c-48e2-b0c6-bcb2f3002879")
    with open(brain_dir / "bgm_centroids_data.pkl", "wb") as f:
        pickle.dump((centroids_dict, num_clear, num_outliers), f)
        
    # Plotting BGM Visualizations
    print("\nPlotting BGM-Auto scatter comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    
    # Generate unique premium colors for active clusters, grey for inactive / outliers
    active_colors = sns.color_palette("husl", num_active)
    color_map = {}
    for i, idx in enumerate(active_indices):
        color_map[idx] = active_colors[i]
        
    outliers_df = df_bgm[~df_bgm["is_clear"]]
    
    # 1. Subplot 1: Speed vs Curvature
    ax = axes[0]
    ax.scatter(
        outliers_df["speed"] * 100,
        outliers_df["curv"],
        c='#bdc3c7',
        alpha=0.15,
        s=12,
        label="Ambiguous Outliers (<95% prob)"
    )
    for idx in active_indices:
        c_df = df_bgm[(df_bgm["cluster"] == idx) & df_bgm["is_clear"]]
        ax.scatter(
            c_df["speed"] * 100,
            c_df["curv"],
            color=color_map[idx],
            alpha=0.85,
            s=22,
            label=f"Cluster {idx} (W={weights[idx]:.1%})"
        )
    ax.set_yscale('log')
    ax.set_title("BGM Automatic Clusters: Speed vs. Curvature (Log Scale)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Speed (cm/s)", fontsize=11)
    ax.set_ylabel("Curvature (log)", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='gray')
    
    # 2. Subplot 2: Speed vs Lateral Acceleration
    ax = axes[1]
    ax.scatter(
        outliers_df["speed"] * 100,
        outliers_df["lat_acc"] * 100,
        c='#bdc3c7',
        alpha=0.15,
        s=12
    )
    for idx in active_indices:
        c_df = df_bgm[(df_bgm["cluster"] == idx) & df_bgm["is_clear"]]
        ax.scatter(
            c_df["speed"] * 100,
            c_df["lat_acc"] * 100,
            color=color_map[idx],
            alpha=0.85,
            s=22
        )
    ax.set_title("BGM Automatic Clusters: Speed vs. Lateral Acceleration", fontsize=13, fontweight='bold')
    ax.set_xlabel("Speed (cm/s)", fontsize=11)
    ax.set_ylabel("Lateral Acceleration (cm/s²)", fontsize=11)
    ax.grid(True, ls=":", alpha=0.5)
    
    # 3. Subplot 3: PCA 2D Feature Space (PC1 vs PC2)
    ax = axes[2]
    ax.scatter(
        outliers_df["pca1"],
        outliers_df["pca2"],
        c='#bdc3c7',
        alpha=0.15,
        s=12
    )
    for idx in active_indices:
        c_df = df_bgm[(df_bgm["cluster"] == idx) & df_bgm["is_clear"]]
        ax.scatter(
            c_df["pca1"],
            c_df["pca2"],
            color=color_map[idx],
            alpha=0.85,
            s=22
        )
    ax.set_title("Optimal Feature Space Projection (PCA PC1 vs PC2)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Principal Component 2", fontsize=11)
    ax.grid(True, ls=":", alpha=0.5)
    
    plt.suptitle(f"Bayesian Gaussian Mixture (BGM) Auto-Clustering Result\nActive Clusters = {num_active} | Clear (>=95%) = {num_clear} ({num_clear/len(df_bgm):.2%}) | Outliers (<95%) = {num_outliers}", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save visualization to brain artifacts
    img_path = brain_dir / "bgm_auto_clustering_distribution.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"BGM visualization saved to {img_path}")

if __name__ == "__main__":
    main()

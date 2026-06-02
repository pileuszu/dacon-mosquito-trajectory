import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
import sys
import os

sys.path.append(os.getcwd())
from step36_three_regime.physics import extract_multi_scale_derivatives, EPS
from step36_three_regime.prepare_data import extract_context_features, CLUSTER_FEATURES

def main():
    print("=== Step 39: 6-Regime GMM Clustering & Error Analysis ===")
    
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv").set_index('id')
    
    # 1. Load Step 36 OOF Predictions for all 10,000 trajectories
    print("\nLoading Step 36 OOF predictions to calculate errors...")
    
    # We need to map trajectory IDs to their OOF error from Step 36
    traj_errors = {}
    
    # GMM 4-Regime mapping from Step 35 (to load the correct Step 36 predictor per trajectory)
    with open("experiments/step35_four_regime/models/scaler.pkl", "rb") as f:
        scaler_gmm4 = pickle.load(f)
    with open("experiments/step35_four_regime/models/gmm_model.pkl", "rb") as f:
        gmm4 = pickle.load(f)
    with open("experiments/step35_four_regime/models/regime_mapping.pkl", "rb") as f:
        mapping_gmm4 = pickle.load(f)
        
    sample_ids = labels_df.index.unique()
    
    # Pre-load predictors of Step 36
    regimes_gmm4 = ["slow_straight", "fast_straight", "slow_extreme_turning", "fast_turning"]
    predictors_36 = {}
    for r in regimes_gmm4:
        p_path = f"experiments/step36_four_regime/models/ranker_v36_{r}"
        print(f"  Loading Step 36 Predictor for {r}...")
        predictors_36[r] = TabularPredictor.load(p_path)
        
    print("Extracting features and predicting Step 36 OOF errors...")
    
    features_list = []
    traj_ids = []
    
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        ctx = extract_context_features(xyz)
        
        feat_vector = [ctx[feat] for feat in CLUSTER_FEATURES]
        features_list.append(feat_vector)
        traj_ids.append(fid)
        
        # Predict GMM4 regime to route to correct Step 36 predictor
        feat_scaled_gmm4 = scaler_gmm4.transform([feat_vector])
        cluster_idx_gmm4 = gmm4.predict(feat_scaled_gmm4)[0]
        regime_gmm4 = mapping_gmm4[cluster_idx_gmm4]
        
        # Load the training CSV for this GMM4 regime to get candidate indices and target distances
        # Wait, instead of loading the full CSV every time, let's load step 36 train files once
        
    print("\nBuilding OOF predictions from Step 36 datasets...")
    # Load step 36 datasets
    for r in regimes_gmm4:
        train_path = f"experiments/step36_four_regime/train_ranker_v36_{r}.csv"
        r_df = pd.read_csv(train_path)
        predictor = predictors_36[r]
        
        # Predict OOF probabilities
        print(f"  Predicting OOF for {r} dataset...")
        oof_proba = predictor.predict_proba_oof()
        score_col = 1 if 1 in oof_proba.columns else oof_proba.columns[0]
        r_df['oof_prob'] = oof_proba[score_col].values
        
        # Select best candidate per ID
        idx_best = r_df.groupby('id')['oof_prob'].idxmax()
        best_cands = r_df.loc[idx_best]
        
        for idx, row in best_cands.iterrows():
            traj_errors[row['id']] = row['reg_target']
            
    # 2. Fit 6-Regime GMM Clustering on context features
    print("\nFitting 6-Regime GMM clustering on context features...")
    X_train = np.array(features_list, dtype=np.float32)
    scaler_gmm6 = StandardScaler()
    X_scaled = scaler_gmm6.fit_transform(X_train)
    
    gmm6 = GaussianMixture(n_components=6, covariance_type='full', random_state=42)
    gmm6.fit(X_scaled)
    clusters6 = gmm6.predict(X_scaled)
    
    # Save the models
    models_dir = Path("step39_six_regime/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_gmm6, f)
    with open(models_dir / "gmm_model.pkl", "wb") as f:
        pickle.dump(gmm6, f)
        
    # 3. Analyze the 6 GMM cluster centroids to assign physical names
    print("\nAnalyzing GMM 6-Regime Cluster Centroids...")
    means = gmm6.means_
    
    # Let's map back to unscaled feature space for inspection
    unscaled_means = scaler_gmm6.inverse_transform(means)
    
    regime_names = {}
    
    # Feature columns corresponding to CLUSTER_FEATURES:
    # 0: ctx_speed, 1: ctx_acc, 2: smooth_curv_w5, 3: ctx_lat_accel, 
    # 4: ctx_z_vel, 5: ctx_z_acc, 6: ctx_p_saccade, 7: roll_speed_cv_all
    
    # Let's print out features for human classification
    print("\nCluster Centroids (Unscaled Mean Values):")
    for i in range(6):
        speed = unscaled_means[i, 0]
        acc = unscaled_means[i, 1]
        curv = unscaled_means[i, 2]
        lat_acc = unscaled_means[i, 3]
        p_sacc = unscaled_means[i, 6]
        volatility = unscaled_means[i, 7]
        
        # Simple logical naming
        if speed < 0.015:
            # Slow classes
            if curv > 15.0:
                name = f"Cluster {i}: Slow-Extreme-Turning (Curv={curv:.1f})"
                regime_names[i] = "slow_extreme_turning"
            elif curv > 5.0:
                name = f"Cluster {i}: Slow-Moderate-Turning (Curv={curv:.1f})"
                regime_names[i] = "slow_moderate_turning"
            else:
                name = f"Cluster {i}: Slow-Straight (Curv={curv:.1f})"
                regime_names[i] = "slow_straight"
        else:
            # Fast classes
            if curv > 15.0:
                name = f"Cluster {i}: Fast-Extreme-Turning (Curv={curv:.1f})"
                regime_names[i] = "fast_extreme_turning"
            elif curv > 5.0:
                name = f"Cluster {i}: Fast-Moderate-Turning (Curv={curv:.1f})"
                regime_names[i] = "fast_moderate_turning"
            else:
                name = f"Cluster {i}: Fast-Straight (Curv={curv:.1f})"
                regime_names[i] = "fast_straight"
        
        print(f"  * {name} | Speed: {speed:.4f}, Curv: {curv:.1f}, LatAcc: {lat_acc:.4f}, Volatility: {volatility:.3f}")
        
    with open(models_dir / "regime_mapping.pkl", "wb") as f:
        pickle.dump(regime_names, f)
        
    # 4. Group Step 36 errors by the new 6 GMM Regimes
    analysis_df = pd.DataFrame({
        'id': traj_ids,
        'cluster': clusters6
    })
    analysis_df['err_step36'] = analysis_df['id'].map(traj_errors)
    
    # 5. Plot OOF Error distributions for the 6 new GMM Regimes
    print("\nGenerating 3x2 error distribution plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.ravel()
    
    # Color palette for 6 clusters
    colors = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C', '#9B59B6', '#F1C40F']
    
    for i in range(6):
        ax = axes[i]
        c_df = analysis_df[analysis_df['cluster'] == i].dropna()
        errors_cm = c_df['err_step36'] * 100
        
        # Plot KDE
        sns.kdeplot(errors_cm, color=colors[i], shade=True, linewidth=2.5, ax=ax)
        
        # Add threshold line
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='1.0cm Hit Threshold')
        
        # Add median line
        med = errors_cm.median()
        ax.axvline(med, color='purple', linestyle=':', linewidth=1.5, label=f'Median: {med:.2f}cm')
        
        # Metrics
        hit_rate = (errors_cm <= 1.0).mean() * 100
        
        # Title and labels
        c_title = regime_names[i].upper().replace('_', ' ')
        ax.set_title(f"Cluster {i}: {c_title} (N={len(c_df)})", fontsize=13, fontweight='bold')
        ax.set_xlabel("OOF Error (cm)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 3.0)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        
        # Text details
        textstr = '\n'.join((
            f"Hit@1cm: {hit_rate:.2f}%",
            f"Median: {med:.2f}cm"
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.55, 0.75, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
    plt.suptitle("Step 39: OOF Prediction Error Distribution Across 6 GMM Regimes\nAnalyzing Peak Separation in Turning Regimes", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the plot
    docs_dir = Path("docs/images")
    docs_dir.mkdir(parents=True, exist_ok=True)
    img_path = docs_dir / "08_step39_6regime_error_distribution.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"\n6-Regime error distribution plot saved to {img_path}")
    
    # Save also to brain artifacts directory for markdown embedding
    brain_img_path = Path("C:/Users/pilla/.gemini/antigravity-ide/brain/f6c70bbe-a99c-48e2-b0c6-bcb2f3002879/08_step39_6regime_error_distribution.png")
    plt.savefig(brain_img_path, dpi=150, bbox_inches='tight')
    print(f"Copied plot to brain artifacts directory: {brain_img_path}")
    
if __name__ == "__main__":
    main()

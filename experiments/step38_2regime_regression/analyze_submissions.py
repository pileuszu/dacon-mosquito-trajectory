import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_a", default="outputs/step36_four_regime/submission.csv", help="First submission file path")
    parser.add_argument("--sub_b", default="outputs/step38_2regime_regression/submission.csv", help="Second submission file path")
    args = parser.parse_args()
    
    print(f"Comparing A: {args.sub_a} vs B: {args.sub_b}")
    sub36 = pd.read_csv(args.sub_a).set_index('id')
    sub38 = pd.read_csv(args.sub_b).set_index('id')
    
    # 1. Check overlap of index
    common_ids = sub36.index.intersection(sub38.index)
    print(f"Common IDs: {len(common_ids)}")
    
    s36_coords = sub36.loc[common_ids, ['x', 'y', 'z']].to_numpy()
    s38_coords = sub38.loc[common_ids, ['x', 'y', 'z']].to_numpy()
    
    # Compute Euclidean distance between predictions
    distances = np.linalg.norm(s36_coords - s38_coords, axis=1)
    
    print("\n--- Overall Coordinate Difference (Step 38 vs Step 36) ---")
    print(f"  Mean Distance  : {distances.mean() * 100:.4f} cm")
    print(f"  Median Distance: {np.median(distances) * 100:.4f} cm")
    print(f"  Max Distance   : {distances.max() * 100:.4f} cm")
    print(f"  Std Distance   : {distances.std() * 100:.4f} cm")
    print(f"  Fraction of identical predictions (<0.1mm): {(distances < 0.0001).mean():.2%}")
    print(f"  Fraction of predictions > 1cm different: {(distances > 0.01).mean():.2%}")
    print(f"  Fraction of predictions > 3cm different: {(distances > 0.03).mean():.2%}")
    
    # Let's map IDs to GMM regimes
    data_dir = Path("data/open")
    test_dir = data_dir / "test"
    
    # Load GMM regime mapping and scaler
    models_dir = Path("experiments/step35_four_regime/models")
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "gmm_model.pkl", "rb") as f:
        gmm = pickle.load(f)
    with open(models_dir / "regime_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
        
    # We need extract_context_features
    import sys
    import os
    sys.path.append(os.getcwd())
    from step38_2regime_regression.prepare_data import extract_context_features, CLUSTER_FEATURES
    
    print("\nAssigning test IDs to regimes...")
    regime_distances = {r: [] for r in mapping.values()}
    regime_counts = {r: 0 for r in mapping.values()}
    
    for fid in common_ids:
        fpath = test_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        ctx = extract_context_features(xyz)
        
        feat_vector = np.array([[ctx[feat] for feat in CLUSTER_FEATURES]], dtype=np.float32)
        feat_scaled = scaler.transform(feat_vector)
        cluster_idx = gmm.predict(feat_scaled)[0]
        regime = mapping[cluster_idx]
        
        # Distance for this ID
        dist = distances[sub36.index.get_loc(fid)]
        regime_distances[regime].append(dist)
        regime_counts[regime] += 1
        
    print("\n--- Regime-wise Coordinate Difference (Step 38 vs Step 36) ---")
    for r, dists in regime_distances.items():
        if len(dists) == 0:
            continue
        dists = np.array(dists)
        print(f"Regime {r:20} (N={len(dists)}):")
        print(f"  Mean Distance  : {dists.mean() * 100:.4f} cm")
        print(f"  Median Distance: {np.median(dists) * 100:.4f} cm")
        print(f"  Max Distance   : {dists.max() * 100:.4f} cm")

if __name__ == "__main__":
    main()

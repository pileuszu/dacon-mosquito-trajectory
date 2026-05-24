import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from step11.physics import make_candidates, CANDIDATES
from step12.prepare_data import extract_context_features, calculate_prior_features

def prepare_augmented_data_v14():
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    s7_preds_df = pd.read_csv("step10/step7_preds_train.csv").set_index('id')
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    rows = []
    
    print(f"Preparing Step 14 Augmented Data (Flexible Windowing)...")
    
    for fid in tqdm(sample_ids):
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz_full = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        s7_pos = s7_preds_df.loc[fid].to_numpy()
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        
        total_len = len(xyz_full)
        if total_len >= 50:
            windows = [xyz_full[0:40], xyz_full[5:45], xyz_full[10:50]]
        elif total_len >= 40:
            windows = [xyz_full[0:30], xyz_full[total_len-40:total_len]]
        else:
            windows = [xyz_full]
        
        for w_idx, xyz in enumerate(windows):
            if len(xyz) < 5: continue # Too short to calculate physics accurately
            
            p0 = xyz[-1]
            try:
                ctx = extract_context_features(xyz)
            except Exception as e:
                continue # Skip if feature extraction fails
                
            cands = make_candidates(xyz, horizon=2)
            dists = np.linalg.norm(cands - target, axis=1)
            best_idx = np.argmin(dists)
            
            neg_indices = np.where(dists > 0.01)[0]
            if len(neg_indices) > 20:
                near_misses = neg_indices[np.argsort(dists[neg_indices])[:10]]
                random_negs = np.random.choice(neg_indices, 10, replace=False)
                selected_indices = list(set([best_idx]) | set(near_misses) | set(random_negs))
            else:
                selected_indices = range(len(cands))
                
            s7_s4_dist = np.linalg.norm(s7_pos - s4_pos)
            
            for idx in selected_indices:
                cand_pos = cands[idx]
                feat_s7 = calculate_prior_features(cand_pos, s7_pos, p0, "s7")
                feat_s4 = calculate_prior_features(cand_pos, s4_pos, p0, "s4")
                
                rows.append({
                    "id": f"{fid}_w{w_idx}",
                    "cand_idx": idx,
                    **feat_s7,
                    **feat_s4,
                    **ctx,
                    "s7_s4_dist": s7_s4_dist,
                    "dist_to_p0": np.linalg.norm(cand_pos - p0),
                    "target": 1 if dists[idx] <= 0.01 else 0
                })
                
    out_df = pd.DataFrame(rows)
    out_path = Path("step14/train_ranker_v14.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Step 14 Augmented Data saved to {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    prepare_augmented_data_v14()

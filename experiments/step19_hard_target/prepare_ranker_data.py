import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from step16.physics import make_candidates, CANDIDATES_GLOBAL
from step13.prepare_data import extract_advanced_context

def prepare_hard_target_data_v19():
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    # 1. Using 100% CLEAN OOF Predictions for BOTH Priors
    s7_oof_df = pd.read_csv("step17_oof/step7_oof_train.csv").set_index('id')
    s4_oof_df = pd.read_csv("step18_oof/step4_oof_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    rows = []
    
    # TARGET THRESHOLD (Hard mode: 0.005m = 0.5cm)
    TARGET_THRESHOLD = 0.005 
    
    print(f"Preparing Step 19 HARD TARGET Data (Threshold: {TARGET_THRESHOLD}m, 100% Clean OOF)...")
    
    for fid in tqdm(sample_ids):
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz_full = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        # Load OOF priors
        s7_pos = s7_oof_df.loc[fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)
        s4_pos = s4_oof_df.loc[fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)
        
        # 3x Augmentation
        total_len = len(xyz_full)
        if total_len >= 50:
            windows = [xyz_full[0:40], xyz_full[5:45], xyz_full[10:50]]
        else:
            windows = [xyz_full]
            
        for w_idx, xyz in enumerate(windows):
            if len(xyz) < 5: continue
            
            p0 = xyz[-1]
            ctx = extract_advanced_context(xyz)
            cands = make_candidates(xyz, priors=[s7_pos, s4_pos], horizon=2)
            
            dists = np.linalg.norm(cands - target, axis=1)
            best_idx = np.argmin(dists)
            
            # Negative sampling logic - adjusted for harder target
            neg_indices = np.where(dists > TARGET_THRESHOLD)[0]
            if len(neg_indices) > 40:
                near_misses = neg_indices[np.argsort(dists[neg_indices])[:20]]
                random_negs = np.random.choice(neg_indices, 20, replace=False)
                selected_indices = list(set([best_idx]) | set(near_misses) | set(random_negs))
            else:
                selected_indices = range(len(cands))
                
            s7_s4_dist = np.linalg.norm(s7_pos - s4_pos)
            
            for idx in selected_indices:
                cand_pos = cands[idx]
                row = {
                    "id": f"{fid}_w{w_idx}",
                    "is_global": 1 if idx < len(CANDIDATES_GLOBAL) else 0,
                    "ctx_speed": ctx['ctx_speed'],
                    "s7_s4_dist": s7_s4_dist,
                    "dist_to_p0": np.linalg.norm(cand_pos - p0),
                    "dist_to_s7": np.linalg.norm(cand_pos - s7_pos),
                    "dist_to_s4": np.linalg.norm(cand_pos - s4_pos),
                    "target": 1 if dists[idx] <= TARGET_THRESHOLD else 0
                }
                
                # Add grid features
                if idx < len(CANDIDATES_GLOBAL):
                    spec = CANDIDATES_GLOBAL[idx]
                    row.update({"spec_par": spec.par, "spec_perp": spec.perp, "spec_ts": spec.ts})
                else:
                    row.update({"spec_par": 0.0, "spec_perp": 0.0, "spec_ts": 1.0})
                    
                rows.append(row)
                
    out_df = pd.DataFrame(rows)
    out_path = Path("step19_hard_target/train_ranker_v19.csv")
    out_df.to_csv(out_path, index=False)
    
    positives = out_df['target'].sum()
    print(f"Step 19 Data saved to {out_path} ({len(out_df)} rows)")
    print(f"Positive samples (<= 0.5cm): {positives} ({positives/len(out_df):.2%})")

if __name__ == "__main__":
    prepare_hard_target_data_v19()

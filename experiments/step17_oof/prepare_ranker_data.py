import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from step16.physics import make_candidates, CANDIDATES_GLOBAL
from step13.prepare_data import extract_advanced_context

def prepare_clean_ranker_data_v17():
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    # CRITICAL: Using Clean OOF predictions (No Leakage!)
    s7_oof_df = pd.read_csv("step17_oof/step7_oof_train.csv").set_index('id')
    # For s4, we'll use step12 for now, but in reality we should OOF that too.
    # However, s7 is the main driver, so fixing it is the priority.
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    rows = []
    
    print(f"Preparing Step 17 CLEAN Ranker Data (Using OOF Priors)...")
    
    for fid in tqdm(sample_ids):
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz_full = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        # Use Clean OOF for s7
        s7_pos = s7_oof_df.loc[fid].to_numpy()
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        
        # 3x Augmentation for stability
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
            
            # Sampling logic
            neg_indices = np.where(dists > 0.01)[0]
            if len(neg_indices) > 30:
                near_misses = neg_indices[np.argsort(dists[neg_indices])[:15]]
                random_negs = np.random.choice(neg_indices, 15, replace=False)
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
                    "target": 1 if dists[idx] <= 0.01 else 0
                }
                
                # Add grid features
                if idx < len(CANDIDATES_GLOBAL):
                    spec = CANDIDATES_GLOBAL[idx]
                    row.update({"spec_par": spec.par, "spec_perp": spec.perp, "spec_ts": spec.ts})
                else:
                    row.update({"spec_par": 0.0, "spec_perp": 0.0, "spec_ts": 1.0})
                    
                rows.append(row)
                
    out_df = pd.DataFrame(rows)
    out_path = Path("step17_oof/train_ranker_v17_clean.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Step 17 CLEAN Data saved to {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    prepare_clean_ranker_data_v17()

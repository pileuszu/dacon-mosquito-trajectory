import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from step12.physics import make_candidates, CANDIDATES_GLOBAL
from step12.prepare_data import extract_context_features, calculate_prior_features

def evaluate_v12():
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    s7_preds_df = pd.read_csv("step10/step7_preds_train.csv").set_index('id')
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    predictor = TabularPredictor.load('step12/models/ranker_v4')
    
    val_ids = labels_df['id'].unique()[9000:10000]
    
    s7_hits = 0
    s12_hits = 0
    oracle_hits = 0
    total = len(val_ids)
    
    print(f"Evaluating Step 12 R-Hit@1cm on {total} samples...")
    for fid in tqdm(val_ids):
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy()[0]
        s7_pos = s7_preds_df.loc[fid].to_numpy()
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        
        if np.linalg.norm(s7_pos - target) <= 0.01:
            s7_hits += 1
            
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        p0 = xyz[-1]
        
        cands = make_candidates(xyz, priors=[s7_pos, s4_pos], horizon=2)
        
        # Oracle check
        dists = np.linalg.norm(cands - target, axis=1)
        if np.any(dists <= 0.01):
            oracle_hits += 1
            
        ctx = extract_context_features(xyz)
        s7_s4_dist = np.linalg.norm(s7_pos - s4_pos)
        
        cand_rows = []
        for idx in range(len(cands)):
            cand_pos = cands[idx]
            feat_s7 = calculate_prior_features(cand_pos, s7_pos, p0, "s7")
            feat_s4 = calculate_prior_features(cand_pos, s4_pos, p0, "s4")
            cand_rows.append({
                "cand_idx": idx,
                "is_local": 1 if idx >= len(CANDIDATES_GLOBAL) else 0,
                **feat_s7,
                **feat_s4,
                **ctx,
                "s7_s4_dist": s7_s4_dist,
                "dist_to_p0": np.linalg.norm(cand_pos - p0)
            })
            
        cand_df = pd.DataFrame(cand_rows)
        probs = predictor.predict_proba(cand_df)
        score_col = 1 if 1 in probs.columns else probs.columns[0]
        
        best_idx = probs[score_col].idxmax()
        max_score = probs.loc[best_idx, score_col]
        
        # Confidence Fallback
        if max_score >= 0.4:
            final_pos = cands[int(cand_df.loc[best_idx, 'cand_idx'])]
        else:
            final_pos = s7_pos
            
        if np.linalg.norm(final_pos - target) <= 0.01:
            s12_hits += 1
            
    print(f"\n[Step 12 Validation Results]")
    print(f"Oracle Rate (Potential): {oracle_hits/total:.4f} ({oracle_hits} hits)")
    print(f"Step 7 R-Hit@1cm: {s7_hits/total:.4f} ({s7_hits} hits)")
    print(f"Step 12 R-Hit@1cm: {s12_hits/total:.4f} ({s12_hits} hits)")
    print(f"Improvement over Baseline: {((s12_hits - s7_hits)/total)*100:+.2f}%p")

if __name__ == "__main__":
    evaluate_v12()

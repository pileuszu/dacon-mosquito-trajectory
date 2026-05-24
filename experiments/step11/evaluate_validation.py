import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from step11.physics import make_candidates, CANDIDATES
from step11.prepare_data import extract_context_features, calculate_advanced_features

def evaluate():
    # 1. Load Setup
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    s7_preds_df = pd.read_csv("step10/step7_preds_train.csv").set_index('id')
    predictor = TabularPredictor.load('step11/models/ranker_v3')
    
    # Use 1000 samples for validation (ID 9000~10000)
    val_ids = labels_df['id'].unique()[9000:10000]
    
    s7_hits = 0
    s11_hits = 0
    total = len(val_ids)
    
    print(f"Evaluating R-Hit@1cm on {total} validation samples...")
    
    for fid in tqdm(val_ids):
        # Ground Truth
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy()[0]
        
        # 1. Step 7 Baseline
        s7_pos = s7_preds_df.loc[fid].to_numpy()
        if np.linalg.norm(s7_pos - target) <= 0.01:
            s7_hits += 1
            
        # 2. Step 11 System
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        p0 = xyz[-1]
        
        cands = make_candidates(xyz, end_idx=-1, horizon=2)
        ctx = extract_context_features(xyz)
        
        cand_rows = []
        for idx in range(len(CANDIDATES)):
            cand_pos = cands[idx]
            adv = calculate_advanced_features(cand_pos, s7_pos, p0)
            cand_rows.append({
                "cand_idx": idx,
                "spec_d1": CANDIDATES[idx].d1,
                "spec_par": CANDIDATES[idx].par,
                "spec_perp": CANDIDATES[idx].perp,
                "spec_ts": CANDIDATES[idx].ts,
                **adv,
                **ctx,
                "dist_to_p0": np.linalg.norm(cand_pos - p0)
            })
            
        cand_df = pd.DataFrame(cand_rows)
        probs = predictor.predict_proba(cand_df)
        score_col = 1 if 1 in probs.columns else probs.columns[0]
        
        best_idx = probs[score_col].idxmax()
        max_score = probs.loc[best_idx, score_col]
        
        if max_score >= 0.4:
            final_pos = cands[int(cand_df.loc[best_idx, 'cand_idx'])]
        else:
            final_pos = s7_pos
            
        if np.linalg.norm(final_pos - target) <= 0.01:
            s11_hits += 1
            
    print(f"\n[Validation Results]")
    print(f"Total Samples: {total}")
    print(f"Step 7 R-Hit@1cm: {s7_hits/total:.4f} ({s7_hits} hits)")
    print(f"Step 11 R-Hit@1cm: {s11_hits/total:.4f} ({s11_hits} hits)")
    print(f"Improvement: {((s11_hits - s7_hits)/total)*100:+.2f}%p")

if __name__ == "__main__":
    evaluate()

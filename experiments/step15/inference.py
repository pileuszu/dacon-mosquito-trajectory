import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from step13.physics import make_candidates, CANDIDATES_GLOBAL
from step13.prepare_data import extract_advanced_context, calculate_physics_features, calculate_prior_features

def run_ultimate_inference():
    data_dir = Path("data/open")
    test_dir = data_dir / "test"
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    
    # Load priors
    s7_preds = pd.read_csv("outputs/step7/submission.csv").set_index('id')
    s4_preds = pd.read_csv("outputs/step12/submission.csv").set_index('id')
    
    # Load Ultimate Ranker
    model_path = 'step15/models/ranker_v7_ultimate'
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Make sure training is complete.")
        return
        
    predictor = TabularPredictor.load(model_path)
    
    results = []
    batch_size = 200  # Balanced batch size
    ids = sample_sub['id'].tolist()
    
    print(f"Running Ultimate Step 15 Inference in batches (Batch Size: {batch_size})...")
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        all_cand_rows = []
        batch_info = []
        
        for fid in tqdm(batch_ids, desc=f"Preparing Batch {i//batch_size + 1}"):
            fpath = test_dir / f"{fid}.csv"
            df = pd.read_csv(fpath)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            p0 = xyz[-1]
            
            s7_pos = s7_preds.loc[fid].to_numpy()
            s4_pos = s4_preds.loc[fid].to_numpy()
            
            ctx = extract_advanced_context(xyz)
            cands = make_candidates(xyz, priors=[s7_pos, s4_pos], horizon=2)
            s7_s4_dist = np.linalg.norm(s7_pos - s4_pos)
            
            start_idx = len(all_cand_rows)
            for idx in range(len(cands)):
                cand_pos = cands[idx]
                phys = calculate_physics_features(cand_pos, p0, ctx)
                feat_s7 = calculate_prior_features(cand_pos, s7_pos, p0, "s7")
                feat_s4 = calculate_prior_features(cand_pos, s4_pos, p0, "s4")
                
                all_cand_rows.append({
                    "cand_idx": idx,
                    "is_local": 1 if idx >= len(CANDIDATES_GLOBAL) else 0,
                    **phys,
                    **feat_s7,
                    **feat_s4,
                    "ctx_speed": ctx['ctx_speed'],
                    "ctx_acc_norm": ctx['ctx_acc_norm'],
                    "s7_s4_dist": s7_s4_dist
                })
            batch_info.append({
                "id": fid,
                "start": start_idx,
                "end": len(all_cand_rows),
                "cands": cands,
                "s7_pos": s7_pos
            })
            
        # Batch Predict
        batch_df = pd.DataFrame(all_cand_rows)
        probs = predictor.predict_proba(batch_df)
        score_col = 1 if 1 in probs.columns else probs.columns[0]
        batch_scores = probs[score_col].values
        
        for info in batch_info:
            fid = info['id']
            scores = batch_scores[info['start'] : info['end']]
            best_idx_in_scores = np.argmax(scores)
            max_score = scores[best_idx_in_scores]
            
            if max_score >= 0.35:
                final_pos = info['cands'][best_idx_in_scores]
            else:
                final_pos = info['s7_pos']
                
            results.append({"id": fid, "x": final_pos[0], "y": final_pos[1], "z": final_pos[2]})
            
    sub_df = pd.DataFrame(results)
    out_path = Path("outputs/step15/submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print(f"ULTIMATE Submission saved to {out_path}")

if __name__ == "__main__":
    run_ultimate_inference()

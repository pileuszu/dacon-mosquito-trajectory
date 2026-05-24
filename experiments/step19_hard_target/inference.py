import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback

sys.path.append(os.getcwd())
from step16.physics import make_candidates, CANDIDATES_GLOBAL
from step13.prepare_data import extract_advanced_context
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def run_hard_inference_v19():
    try:
        send_discord_notification(URL, "🚀 [Step 19] Hard Target (0.5cm) Inference Started...")
        
        data_dir = Path("data/open")
        test_dir = data_dir / "test"
        sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
        
        # Use standard Step 7 & 4 priors for Test (These are honest by default for test set)
        s7_preds = pd.read_csv("outputs/step7/submission.csv").set_index('id')
        s4_preds = pd.read_csv("outputs/step12/submission.csv").set_index('id')
        
        # Load HARD Ranker V19
        model_path = 'step19_hard_target/models/ranker_v19'
        predictor = TabularPredictor.load(model_path)
        
        results = []
        batch_size = 200
        ids = sample_sub['id'].tolist()
        
        print(f"Running Hard Target (0.5cm) Step 19 Inference...")
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            all_cand_rows = []
            batch_info = []
            
            for fid in tqdm(batch_ids, desc=f"Batch {i//batch_size + 1}"):
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
                    row = {
                        "fid": fid,
                        "cand_idx": idx,
                        "is_global": 1 if idx < len(CANDIDATES_GLOBAL) else 0,
                        "ctx_speed": ctx['ctx_speed'],
                        "s7_s4_dist": s7_s4_dist,
                        "dist_to_p0": np.linalg.norm(cand_pos - p0),
                        "dist_to_s7": np.linalg.norm(cand_pos - s7_pos),
                        "dist_to_s4": np.linalg.norm(cand_pos - s4_pos)
                    }
                    if idx < len(CANDIDATES_GLOBAL):
                        spec = CANDIDATES_GLOBAL[idx]
                        row.update({"spec_par": spec.par, "spec_perp": spec.perp, "spec_ts": spec.ts})
                    else:
                        row.update({"spec_par": 0.0, "spec_perp": 0.0, "spec_ts": 1.0})
                    all_cand_rows.append(row)
                batch_info.append({"id": fid, "start": start_idx, "end": len(all_cand_rows), "cands": cands})
                
            batch_df = pd.DataFrame(all_cand_rows)
            pred_data = batch_df.drop(columns=['fid'])
            probs = predictor.predict_proba(pred_data)
            score_col = 1 if 1 in probs.columns else probs.columns[0]
            batch_scores = probs[score_col].values
            
            for info in batch_info:
                scores = batch_scores[info['start'] : info['end']]
                best_idx = np.argmax(scores)
                final_pos = info['cands'][best_idx]
                results.append({"id": info['id'], "x": final_pos[0], "y": final_pos[1], "z": final_pos[2]})
                
        sub_df = pd.DataFrame(results)
        out_path = Path("outputs/step19_hard_target/submission.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(out_path, index=False)
        
        success_msg = f"✅ [Step 19] Hard Target Inference Finished Successfully!\nSaved to: {out_path}"
        send_discord_notification(URL, success_msg)
        print(success_msg)
        
    except Exception as e:
        error_msg = f"❌ [Step 19] Inference ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    run_hard_inference_v19()

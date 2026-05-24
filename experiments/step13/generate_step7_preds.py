import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor

def generate_step7_preds_expanded(limit=30000):
    data_dir = Path("data/open")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    sample_ids = labels_df['id'].unique()[:limit]
    
    # Check existing
    existing_path = Path("step10/step7_preds_train.csv")
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        existing_ids = set(existing_df['id'].tolist())
        todo_ids = [fid for fid in sample_ids if fid not in existing_ids]
    else:
        existing_df = pd.DataFrame()
        todo_ids = sample_ids
        
    if not todo_ids:
        print(f"All {limit} Step 7 predictions already exist.")
        return
        
    print(f"Generating Step 7 predictions for {len(todo_ids)} new samples...")
    
    # Load predictors
    p_x = TabularPredictor.load('step7/models/target_x')
    p_y = TabularPredictor.load('step7/models/target_y')
    p_z = TabularPredictor.load('step7/models/target_z')
    
    results = []
    
    # Features from step7/inference_automl.py
    for fid in tqdm(todo_ids):
        fpath = data_dir / "train" / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy()
        
        # Simple tabular features (matching step 7 training)
        feat = {
            "last_x": xyz[-1, 0], "last_y": xyz[-1, 1], "last_z": xyz[-1, 2],
            "v_x": xyz[-1, 0] - xyz[-2, 0], "v_y": xyz[-1, 1] - xyz[-2, 1], "v_z": xyz[-1, 2] - xyz[-2, 2],
            "a_x": (xyz[-1, 0] - xyz[-2, 0]) - (xyz[-2, 0] - xyz[-3, 0]),
            "a_y": (xyz[-1, 1] - xyz[-2, 1]) - (xyz[-2, 1] - xyz[-3, 1]),
            "a_z": (xyz[-1, 2] - xyz[-2, 2]) - (xyz[-2, 2] - xyz[-3, 2])
        }
        
        test_df = pd.DataFrame([feat])
        pred_x = p_x.predict(test_df)[0]
        pred_y = p_y.predict(test_df)[0]
        pred_z = p_z.predict(test_df)[0]
        
        results.append({"id": fid, "x": pred_x, "y": pred_y, "z": pred_z})
        
    new_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
    out_path = Path("step13/step7_preds_train_30k.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(out_path, index=False)
    print(f"Step 7 expanded predictions saved to {out_path}")

if __name__ == "__main__":
    generate_step7_preds_expanded()

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

class ConstantVelocityModel:
    def __init__(self):
        pass

    def predict_sample(self, sample_path: Path):
        """
        Predicts the position 80ms ahead using constant velocity.
        P_80 = P_0 + 2 * (P_0 - P_{-40})
        """
        df = pd.read_csv(sample_path)
        # The interval is 40ms. 
        # Last two points are at index -1 (0ms) and -2 (-40ms)
        prev_xyz = df.loc[df.index[-2], ['x', 'y', 'z']].to_numpy(dtype=float)
        last_xyz = df.loc[df.index[-1], ['x', 'y', 'z']].to_numpy(dtype=float)
        
        pred_xyz = last_xyz + 2.0 * (last_xyz - prev_xyz)
        return pred_xyz

    def predict_batch(self, sample_files):
        """
        Generates predictions for a list of sample files.
        """
        rows = []
        for sample_path in tqdm(sample_files, desc="Predicting"):
            pred_xyz = self.predict_sample(sample_path)
            rows.append({
                "id": sample_path.stem,
                "x": pred_xyz[0],
                "y": pred_xyz[1],
                "z": pred_xyz[2],
            })
        return pd.DataFrame(rows)

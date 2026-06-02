import pandas as pd
import numpy as np
from pathlib import Path

def analyze():
    regimes = ["slow_straight", "slow_extreme_turning", "fast_straight", "fast_turning"]
    
    print("=== Step 38 Dataset Distance Distribution Analysis ===")
    
    for r in regimes:
        path = Path(f"step38_2regime_regression/train_ranker_v38_{r}.csv")
        if not path.exists():
            print(f"Error: {path} does not exist!")
            continue
            
        print(f"\nAnalyzing regime: {r.upper()} ({path.name})")
        df = pd.read_csv(path)
        print(f"  Total rows: {len(df)}")
        
        # reg_target is in meters, convert to cm
        dists_cm = df['reg_target'] * 100.0
        
        print("  Percentiles of Distance Error (cm):")
        for p in [0, 10, 25, 50, 75, 90, 100]:
            print(f"    {p:3d}th percentile: {np.percentile(dists_cm, p):.4f} cm")
            
        # Distance bins
        bins = [0.0, 0.5, 1.0, 1.5, 3.0, float('inf')]
        bin_labels = ["[0.0, 0.5) cm", "[0.5, 1.0) cm", "[1.0, 1.5) cm", "[1.5, 3.0) cm", ">= 3.0 cm"]
        
        binned = pd.cut(dists_cm, bins=bins, labels=bin_labels, right=False)
        counts = binned.value_counts().sort_index()
        pcts = binned.value_counts(normalize=True).sort_index() * 100.0
        
        print("  Distance Bin Counts and Percentages:")
        for label, count, pct in zip(bin_labels, counts, pcts):
            print(f"    {label:15}: {count:6d} ({pct:.2f}%)")

if __name__ == "__main__":
    analyze()

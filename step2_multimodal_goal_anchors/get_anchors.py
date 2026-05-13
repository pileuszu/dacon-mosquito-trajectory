import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
from glob import glob
from tqdm.auto import tqdm
import pickle

def generate_anchors(data_dir, label_path, n_clusters=6):
    print("Calculating Goal Anchors using K-Means...")
    labels = pd.read_csv(label_path)
    
    all_displacements = []
    
    # We need the last point (t=0) to calculate displacement
    for i, row in tqdm(labels.iterrows(), total=len(labels)):
        file_id = row['id']
        file_path = os.path.join(data_dir, f"{file_id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            last_p = df[['x', 'y', 'z']].values[-1]
            target_p = row[['x', 'y', 'z']].values
            disp = target_p - last_p
            all_displacements.append(disp)
            
    all_displacements = np.array(all_displacements)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(all_displacements)
    
    anchors = kmeans.cluster_centers_
    print(f"Generated {n_clusters} Anchors:")
    print(anchors)
    
    # Save anchors
    save_path = 'step2_multimodal_goal_anchors/anchors.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(anchors, f)
    print(f"Anchors saved to {save_path}")
    return anchors

if __name__ == "__main__":
    DATA_DIR = 'data/open/train/'
    LABEL_PATH = 'data/open/train_labels.csv'
    generate_anchors(DATA_DIR, LABEL_PATH)

import pandas as pd
from sklearn.model_selection import train_test_split
from config import TRAIN_DIR, METADATA_PATH, TEST_SIZE, RANDOM_STATE

def get_splits():
    """
    Returns train and test splits of the training data.
    Ensures consistent splitting by saving/loading to metadata.csv.
    """
    if METADATA_PATH.exists():
        print(f"Loading existing split metadata from {METADATA_PATH}")
        df = pd.read_csv(METADATA_PATH)
    else:
        print(f"Generating new train/test split (test_size={TEST_SIZE})...")
        train_files = sorted(TRAIN_DIR.glob('*.csv'))
        file_ids = [f.stem for f in train_files]
        
        train_ids, test_ids = train_test_split(
            file_ids, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        
        metadata = []
        for fid in train_ids:
            metadata.append({"id": fid, "split": "train"})
        for fid in test_ids:
            metadata.append({"id": fid, "split": "test"})
        
        df = pd.DataFrame(metadata)
        df.to_csv(METADATA_PATH, index=False)
        print(f"Saved split metadata to {METADATA_PATH}")

    train_ids = df[df['split'] == 'train']['id'].tolist()
    test_ids = df[df['split'] == 'test']['id'].tolist()
    
    train_files = [TRAIN_DIR / f"{fid}.csv" for fid in train_ids]
    test_files = [TRAIN_DIR / f"{fid}.csv" for fid in test_ids]
    
    return train_files, test_files

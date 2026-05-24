from autogluon.tabular import TabularPredictor
import pandas as pd

def check():
    path = "step11/models/ranker"
    predictor = TabularPredictor.load(path)
    print("Model names:", predictor.model_names())
    try:
        print("Best model:", predictor.model_best())
    except:
        print("Best model: NOT FOUND")
    
    lb = predictor.leaderboard()
    print("\nLeaderboard:")
    print(lb)

if __name__ == "__main__":
    check()

import pandas as pd
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback
from pathlib import Path

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

def train_regime_model(regime_name, train_data_path, model_path, time_limit):
    try:
        send_discord_notification(None, f"🚀 [Step 37] GMM Regime Regression Ranker ({regime_name}) Training Started...")
        
        print(f"Loading {regime_name} data from {train_data_path}...")
        df = pd.read_csv(train_data_path)
        
        # Drop identifier and classification target for training
        train_df = df.drop(columns=['id', 'target'])
        
        # Memory-safe configuration (No RF/XT to prevent bad allocation on Windows)
        hyperparameters = {
            'GBM': {},
            'CAT': {},
            'XGB': {},
            'NN_TORCH': {},
            'FASTAI': {}
        }
        
        print(f"Starting AutoGluon TabularPredictor fit ('best_quality') in REGRESSION mode for {regime_name}...")
        predictor = TabularPredictor(
            label='reg_target',
            problem_type='regression',
            eval_metric='mean_absolute_error',
            path=model_path
        ).fit(
            train_data=train_df,
            presets='best_quality',
            hyperparameters=hyperparameters,
            num_bag_folds=5,  # 5-fold bagging for memory safety
            time_limit=time_limit,
            ag_args_ensemble={
                'num_folds_parallel': 1,
                'fold_fitting_strategy': 'sequential_local'
            },
            verbosity=2
        )
        
        # Calculate OOF Hit@1cm validation score (Group-wise alignment using minimum predicted distance)
        print("Calculating Out-Of-Fold (OOF) predicted distances...")
        oof_pred = predictor.predict_oof()
        df['oof_pred_dist'] = oof_pred.values
        
        # Select candidate with the MINIMUM predicted distance per trajectory
        idx_best = df.groupby('id')['oof_pred_dist'].idxmin()
        best_cands = df.loc[idx_best]
        
        # Evaluate metrics
        oof_hit_rate_1cm = (best_cands['reg_target'] <= 0.01).mean()
        oof_hit_rate_1_5cm = (best_cands['reg_target'] <= 0.015).mean()
        mean_oof_error = best_cands['reg_target'].mean()
        
        # Get AutoGluon Leaderboard
        leaderboard = predictor.leaderboard(train_df, silent=True)
        leaderboard_str = leaderboard.head(15).to_string()
        print(f"\n=== Model Leaderboard ({regime_name}) ===")
        print(leaderboard_str)
        
        best_model = predictor.model_best
        best_score = leaderboard.loc[leaderboard['model'] == best_model, 'score_val'].values[0]
        
        success_msg = (
            f"✅ [Step 37] GMM Regime Regression Ranker ({regime_name}) Training Finished!\n"
            f"Group-wise **OOF Hit@1.0cm**: **{oof_hit_rate_1cm:.4%}** 🏆\n"
            f"Group-wise **OOF Hit@1.5cm**: **{oof_hit_rate_1_5cm:.4%}** ✨\n"
            f"Mean Selected OOF Error: **{mean_oof_error:.4f} cm** 📏\n"
            f"Best Model: **{best_model}**\n"
            f"Best Validation MAE: **{-best_score:.6f} cm**\n\n"
            f"**Leaderboard Top 10:**\n```\n{leaderboard.head(10)[['model', 'score_val', 'pred_time_val', 'fit_time']].to_string()}\n```"
        )
        send_discord_notification(None, success_msg)
        print(success_msg)
        
    except BaseException as e:
        error_msg = f"❌ [Step 37] Training ERROR ({regime_name}):\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e

def main():
    # Make sure output models directory exists
    Path("step37_turning_refinement/models").mkdir(parents=True, exist_ok=True)
    
    # 1. Slow-Straight
    train_regime_model(
        regime_name="SLOW_STRAIGHT",
        train_data_path="step36_four_regime/train_ranker_v36_slow_straight.csv",
        model_path="step37_turning_refinement/models/ranker_v37_slow_straight",
        time_limit=3600
    )
    
    # 2. Fast-Straight
    train_regime_model(
        regime_name="FAST_STRAIGHT",
        train_data_path="step36_four_regime/train_ranker_v36_fast_straight.csv",
        model_path="step37_turning_refinement/models/ranker_v37_fast_straight",
        time_limit=3600
    )
    
    # 3. Slow-Extreme Turning
    train_regime_model(
        regime_name="SLOW_EXTREME_TURNING",
        train_data_path="step36_four_regime/train_ranker_v36_slow_extreme_turning.csv",
        model_path="step37_turning_refinement/models/ranker_v37_slow_extreme_turning",
        time_limit=3600
    )
    
    # 4. Fast-Turning
    train_regime_model(
        regime_name="FAST_TURNING",
        train_data_path="step36_four_regime/train_ranker_v36_fast_turning.csv",
        model_path="step37_turning_refinement/models/ranker_v37_fast_turning",
        time_limit=3600
    )

if __name__ == "__main__":
    main()

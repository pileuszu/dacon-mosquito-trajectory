import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, LGBMClassifier
import hashlib

EPS = 1e-8

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.cfm_model import SotaSpatiotemporalModel, extract_features, SotaFocalSoftHitLoss
from src.models.neural_ode import FrenetNeuralODEModel, extract_ode_features
from src.models.outlier_classifier import OutlierDampingClassifier, extract_classifier_features
from src.candidate_generator import generate_hybrid_36_grid, generate_hybrid_43_grid

# Check if AutoGluon is installed, otherwise fallback to LightGBM
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

class MosquitoDataset(Dataset):
    def __init__(self, X, y, candidates, regimes):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.candidates = torch.tensor(candidates, dtype=torch.float32)
        self.regimes = torch.tensor(regimes, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.candidates[idx], self.regimes[idx]

def stable_fold_id(sample_id: str, folds: int = 5) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run a fast dry-run / mock training")
    parser.add_argument("--device", type=str, default="auto", help="torch device: cpu, cuda, or auto")
    args = parser.parse_args()
    
    print("=== Step 3: Training Reproduction Pipeline ===")
    if args.fast:
        print("⚡ FAST MODE ACTIVE: Reducing epochs and estimators for dry-run verification.")
        
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else 
                          ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {device}")
    
    data_dir = Path("data/processed")
    models_dir = Path("models_trained")
    models_dir.mkdir(exist_ok=True)
    
    # Load processed datasets
    train_x = np.load(data_dir / "train_x.npy")
    train_y = np.load(data_dir / "train_y.npy")
    test_x = np.load(data_dir / "test_x.npy")
    
    with open(data_dir / "train_ids.json", "r") as f:
        train_ids = json.load(f)
    with open(data_dir / "test_ids.json", "r") as f:
        test_ids = json.load(f)
        
    regimes_tr = np.load(data_dir / "regimes_train.npy")
    regimes_te = np.load(data_dir / "regimes_test.npy")
    speeds_tr = np.load(data_dir / "speeds_train.npy")
    speeds_te = np.load(data_dir / "speeds_test.npy")
    
    fold_ids = np.asarray([stable_fold_id(sid) for sid in train_ids])
    folds = 5
    
    # Set model config parameters
    cfm_epochs = 2 if args.fast else 35
    ode_epochs = 2 if args.fast else 30
    ag_time_limit = 10 if args.fast else 900
    
    # ------------------ 1. Train Spatiotemporal ODE Model ------------------
    print("\n--- Training Frenet Neural ODE Model (5-Fold CV) ---")
    ode_oof = np.zeros_like(train_y)
    ode_test = np.zeros((len(test_x), 3))
    
    for fold in range(folds):
        print(f"Fold {fold+1}/{folds}...")
        train_mask = fold_ids != fold
        val_mask = fold_ids == fold
        
        # We need raw history and derivatives
        X_tr, y_tr = train_x[train_mask], train_y[train_mask]
        X_val, y_val = train_x[val_mask], train_y[val_mask]
        
        # Prepare inputs
        tr_scaled, tr_df, tr_plt, tr_tht, _, tr_last_a, tr_R, tr_spt, mean_s, std_s = extract_ode_features(torch.tensor(X_tr).to(device))
        val_scaled, val_df, val_plt, val_tht, _, val_last_a, val_R, val_spt, _, _ = extract_ode_features(torch.tensor(X_val).to(device), mean_s, std_s)
        te_scaled, te_df, te_plt, te_tht, _, te_last_a, te_R, te_spt, _, _ = extract_ode_features(torch.tensor(test_x).to(device), mean_s, std_s)
        
        torch.save({"mean": mean_s.cpu(), "std": std_s.cpu()}, models_dir / f"ode_stats_fold_{fold}.pt")
        
        # Initialize model
        model = FrenetNeuralODEModel(input_dim=27, latent_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Train
        model.train()
        for epoch in range(ode_epochs):
            optimizer.zero_grad()
            pred = model(tr_scaled, tr_df, tr_plt, tr_tht, tr_spt, tr_R, tr_last_a)
            loss = F.huber_loss(pred, torch.tensor(y_tr).to(device), delta=0.001)
            loss.backward()
            optimizer.step()
            
        # Predict val & test
        model.eval()
        with torch.no_grad():
            val_pred = model(val_scaled, val_df, val_plt, val_tht, val_spt, val_R, val_last_a).cpu().numpy()
            te_pred = model(te_scaled, te_df, te_plt, te_tht, te_spt, te_R, te_last_a).cpu().numpy()
            
        ode_oof[val_mask] = val_pred
        ode_test += te_pred / folds
        torch.save(model.state_dict(), models_dir / f"ode_model_fold_{fold}.pt")
        
    np.save(models_dir / "ode_preds_oof.npy", ode_oof)
    np.save(models_dir / "ode_preds_test.npy", ode_test)
    print(f"ODE OOF Hit@1cm: {np.mean(np.linalg.norm(ode_oof - train_y, axis=1) <= 0.01)*100:.3f}%")
    
    # ------------------ 2. Generate Hybrid Candidates ------------------
    print("\n--- Generating Candidate Grids (36 & 43 Candidates) ---")
    train_cands_36 = generate_hybrid_36_grid(train_x, ode_oof)
    test_cands_36 = generate_hybrid_36_grid(test_x, ode_test)
    
    # ------------------ 3. Train Spatiotemporal Clifford-Mamba CFM Model ------------------
    print("\n--- Training Clifford-Mamba CFM Model (5-Fold CV) ---")
    cfm_oof_1step = np.zeros_like(train_y)
    cfm_oof_2step = np.zeros_like(train_y)
    cfm_test_1step = np.zeros((len(test_x), 3))
    cfm_test_2step = np.zeros((len(test_x), 3))
    
    for fold in range(folds):
        print(f"Fold {fold+1}/{folds}...")
        train_mask = fold_ids != fold
        val_mask = fold_ids == fold
        
        X_tr, y_tr, cand_tr, reg_tr = train_x[train_mask], train_y[train_mask], train_cands_36[train_mask], regimes_tr[train_mask]
        X_val, y_val, cand_val, reg_val = train_x[val_mask], train_y[val_mask], train_cands_36[val_mask], regimes_tr[val_mask]
        
        train_loader = DataLoader(MosquitoDataset(X_tr, y_tr, cand_tr, reg_tr), batch_size=256, shuffle=True)
        val_loader = DataLoader(MosquitoDataset(X_val, y_val, cand_val, reg_val), batch_size=256, shuffle=False)
        
        # Feature scale statistics
        with torch.no_grad():
            _, _, _, mean_s, std_s, _, _ = extract_features(torch.tensor(X_tr, dtype=torch.float32).to(device))
        torch.save({"mean": mean_s.cpu(), "std": std_s.cpu()}, models_dir / f"cfm_stats_fold_{fold}.pt")
        
        # Model
        model = SotaSpatiotemporalModel(feature_dim=47, latent_dim=128, num_candidates=36, max_norm=0.05, d_mamba_in=9, use_cross_attn=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        criterion = SotaFocalSoftHitLoss(delta=0.001026, alpha=200.0, target_dist=0.01).to(device)
        
        model.train()
        for epoch in range(cfm_epochs):
            for Xb, yb, candb, regb in train_loader:
                Xb, yb, candb, regb = Xb.to(device), yb.to(device), candb.to(device), regb.to(device)
                optimizer.zero_grad()
                ft, df_c, df_s, _, _, _, _ = extract_features(Xb, mean_s, std_s)
                pred_v, target_v, x_0, h_context = model.forward_flow(Xb, ft, df_c, df_s, candb, yb, regb, noise_scale=0.01)
                
                loss_cfm = F.huber_loss(pred_v, target_v, delta=0.001)
                
                t0 = torch.zeros(Xb.shape[0], device=device)
                v0 = model.vector_field(x_0, t0, h_context)
                pred_1step = x_0 + v0
                
                dt = 0.5
                x_half = x_0 + v0 * dt
                t05 = torch.ones(Xb.shape[0], device=device) * 0.5
                v05 = model.vector_field(x_half, t05, h_context)
                pred_2step = x_half + v05 * dt
                
                loss_pred = criterion(pred_1step, yb) + criterion(pred_2step, yb)
                loss_cons = F.mse_loss(pred_1step, pred_2step)
                
                loss = 10.0 * loss_cfm + 1.0 * loss_pred + 50.0 * loss_cons
                loss.backward()
                optimizer.step()
                
        # Predict val & test
        model.eval()
        val_preds_1step = []
        val_preds_2step = []
        with torch.no_grad():
            for Xv, yv, candv, regv in val_loader:
                Xv, yv, candv, regv = Xv.to(device), yv.to(device), candv.to(device), regv.to(device)
                ft_v, df_c_v, df_s_v, _, _, _, _ = extract_features(Xv, mean_s, std_s)
                pv_1 = model.predict(Xv, ft_v, df_c_v, df_s_v, candv, regv, steps=1)
                pv_2 = model.predict(Xv, ft_v, df_c_v, df_s_v, candv, regv, steps=2)
                val_preds_1step.append(pv_1.cpu().numpy())
                val_preds_2step.append(pv_2.cpu().numpy())
                
        cfm_oof_1step[val_mask] = np.concatenate(val_preds_1step, axis=0)
        cfm_oof_2step[val_mask] = np.concatenate(val_preds_2step, axis=0)
        
        # Test predictions
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_cand_tensor = torch.tensor(test_cands_36, dtype=torch.float32).to(device)
        test_reg_tensor = torch.tensor(regimes_te, dtype=torch.long).to(device)
        
        with torch.no_grad():
            ft_te, df_c_te, df_s_te, _, _, _, _ = extract_features(test_x_tensor, mean_s, std_s)
            tp_1 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, test_cand_tensor, test_reg_tensor, steps=1)
            tp_2 = model.predict(test_x_tensor, ft_te, df_c_te, df_s_te, test_cand_tensor, test_reg_tensor, steps=2)
            cfm_test_1step += tp_1.cpu().numpy() / folds
            cfm_test_2step += tp_2.cpu().numpy() / folds
            
        torch.save(model.state_dict(), models_dir / f"cfm_model_fold_{fold}.pt")
        
    np.save(models_dir / "cfm_preds_oof_1step.npy", cfm_oof_1step)
    np.save(models_dir / "cfm_preds_oof_2step.npy", cfm_oof_2step)
    np.save(models_dir / "cfm_preds_test_1step.npy", cfm_test_1step)
    np.save(models_dir / "cfm_preds_test_2step.npy", cfm_test_2step)
    print(f"CFM OOF 2-Step Hit@1cm: {np.mean(np.linalg.norm(cfm_oof_2step - train_y, axis=1) <= 0.01)*100:.3f}%")
    
    # ------------------ 4. Generate Expanded 43 Candidate Grid ------------------
    print("\n--- Generating Augmented 43-Candidate Grids ---")
    train_cands_43 = generate_hybrid_43_grid(train_x, cfm_oof_2step, train_cands_36)
    test_cands_43 = generate_hybrid_43_grid(test_x, cfm_test_2step, test_cands_36)
    np.save(models_dir / "train_candidates_hybrid_43.npy", train_cands_43)
    np.save(models_dir / "test_candidates_hybrid_43.npy", test_cands_43)
    
    # ------------------ 5. Train Tabular AutoGluon/LightGBM Ranker ------------------
    # Generate tabular DataFrames
    print("\n--- Building Tabular Feature DataFrames for Rankers ---")
    T_tr, N_tr, B_tr, speeds_tr_last, a_perp_tr_last = get_frenet_unit_vectors(train_x)
    T_te, N_te, B_te, speeds_te_last, a_perp_te_last = get_frenet_unit_vectors(test_x)
    
    # Extract historical super features
    train_feat = np.load(data_dir / "train_feat.npy")
    test_feat = np.load(data_dir / "test_feat.npy")
    
    # Format dataset using fast vectorized numpy operations
    print("Vectorizing train features...")
    N_tr_len = len(train_x)
    sample_idx_val = np.repeat(np.arange(N_tr_len), 43)
    cand_idx_val = np.tile(np.arange(43), N_tr_len)
    fold_id_val = np.repeat(fold_ids, 43)
    
    disp_tr = train_cands_43 - train_x[:, -1, None, :]
    spec_par_tr = np.sum(disp_tr * T_tr[:, None, :], axis=2)
    spec_perp_tr = np.sum(disp_tr * N_tr[:, None, :], axis=2)
    spec_ts_tr = np.sum(disp_tr * B_tr[:, None, :], axis=2)
    dist_to_cfm_tr = np.linalg.norm(train_cands_43 - cfm_oof_2step[:, None, :], axis=2) * 100.0
    dist_target_tr = np.linalg.norm(train_cands_43 - train_y[:, None, :], axis=2) * 100.0
    
    feat_repeated_tr = np.repeat(train_feat, 43, axis=0)
    
    cols_val = {
        "sample_idx": sample_idx_val,
        "cand_idx": cand_idx_val,
        "fold_id": fold_id_val,
        "spec_par": spec_par_tr.flatten(),
        "spec_perp": spec_perp_tr.flatten(),
        "spec_ts": spec_ts_tr.flatten(),
        "dist_to_cfm": dist_to_cfm_tr.flatten(),
        "dist_target": dist_target_tr.flatten()
    }
    for f_idx in range(train_feat.shape[1]):
        cols_val[f"feat_{f_idx}"] = feat_repeated_tr[:, f_idx]
    val_df = pd.DataFrame(cols_val)
    
    # Sub-sample candidates for training to prevent memory overflow
    selected_rows = []
    for idx in range(N_tr_len):
        dists = np.linalg.norm(train_cands_43[idx] - train_y[idx], axis=1)
        best_cand = np.argmin(dists)
        other_indices = [i for i in range(43) if i != best_cand]
        sampled_others = np.random.choice(other_indices, size=7, replace=False)
        selected_indices = np.append(best_cand, sampled_others)
        selected_rows.extend(idx * 43 + selected_indices)
        
    train_df = val_df.iloc[selected_rows].reset_index(drop=True)
    
    print("Vectorizing test features...")
    N_te_len = len(test_x)
    sample_idx_te = np.repeat(np.arange(N_te_len), 43)
    cand_idx_te = np.tile(np.arange(43), N_te_len)
    
    disp_te = test_cands_43 - test_x[:, -1, None, :]
    spec_par_te = np.sum(disp_te * T_te[:, None, :], axis=2)
    spec_perp_te = np.sum(disp_te * N_te[:, None, :], axis=2)
    spec_ts_te = np.sum(disp_te * B_te[:, None, :], axis=2)
    dist_to_cfm_te = np.linalg.norm(test_cands_43 - cfm_test_2step[:, None, :], axis=2) * 100.0
    
    feat_repeated_te = np.repeat(test_feat, 43, axis=0)
    
    cols_te = {
        "sample_idx": sample_idx_te,
        "cand_idx": cand_idx_te,
        "spec_par": spec_par_te.flatten(),
        "spec_perp": spec_perp_te.flatten(),
        "spec_ts": spec_ts_te.flatten(),
        "dist_to_cfm": dist_to_cfm_te.flatten()
    }
    for f_idx in range(test_feat.shape[1]):
        cols_te[f"feat_{f_idx}"] = feat_repeated_te[:, f_idx]
    test_df = pd.DataFrame(cols_te)
    
    print("\n--- Training Tabular Rankers (5-Fold CV) ---")
    ranker_oof = np.zeros_like(train_y)
    ranker_test = np.zeros((len(test_x), 3))
    
    use_autogluon = AUTOGLUON_AVAILABLE and not args.fast
    
    if use_autogluon:
        print("Using AutoGluon TabularPredictor for training.")
        oof_dists = []
        for fold in range(folds):
            print(f"Fold {fold+1}/{folds}...")
            fold_model_path = models_dir / f"ag_predictor_fold_{fold}"
            
            train_fold = train_df[train_df['fold_id'] != fold].drop(columns=['sample_idx', 'cand_idx', 'fold_id'])
            val_fold_full = val_df[val_df['fold_id'] == fold]
            
            predictor = TabularPredictor(
                label='dist_target',
                problem_type='regression',
                eval_metric='mean_absolute_error',
                path=fold_model_path
            )
            predictor.fit(
                train_fold,
                time_limit=ag_time_limit,
                presets='best_quality',
                excluded_model_types=['RF', 'XT'],
                verbosity=0
            )
            
            val_data = val_fold_full.drop(columns=['sample_idx', 'cand_idx', 'fold_id', 'dist_target'])
            pred_dists = predictor.predict(val_data)
            
            fold_eval = val_fold_full[['sample_idx', 'cand_idx']].copy()
            fold_eval['pred_dist'] = pred_dists
            oof_dists.append(fold_eval)
            
            # Predict test set
            test_preds_dists = predictor.predict(test_df.drop(columns=['sample_idx', 'cand_idx']))
            test_eval = test_df[['sample_idx', 'cand_idx']].copy()
            test_eval['pred_dist'] = test_preds_dists
            
            # Convert test distances to coordinates using Argmin
            for s_idx in range(len(test_x)):
                s_test = test_eval[test_eval['sample_idx'] == s_idx]
                best_idx = s_test['pred_dist'].idxmin()
                best_cand_idx = s_test.loc[best_idx, 'cand_idx']
                ranker_test[s_idx] += test_cands_43[s_idx, int(best_cand_idx)] / folds
                
        # Resolve OOF predictions
        oof_dist_df = pd.concat(oof_dists, ignore_index=True)
        for s_idx in range(len(train_x)):
            s_val = oof_dist_df[oof_dist_df['sample_idx'] == s_idx]
            best_idx = s_val['pred_dist'].idxmin()
            best_cand_idx = s_val.loc[best_idx, 'cand_idx']
            ranker_oof[s_idx] = train_cands_43[s_idx, int(best_cand_idx)]
            
    else:
        print("Using LightGBM Regressor as a fast surrogate ranker.")
        for fold in range(folds):
            print(f"Fold {fold+1}/{folds}...")
            train_fold = train_df[train_df['fold_id'] != fold]
            val_fold_full = val_df[val_df['fold_id'] == fold]
            
            X_tr_tab = train_fold.drop(columns=['sample_idx', 'cand_idx', 'fold_id', 'dist_target'])
            y_tr_tab = train_fold['dist_target']
            
            reg = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbosity=-1, random_state=42)
            reg.fit(X_tr_tab, y_tr_tab)
            
            # Predict validation distances
            val_data = val_fold_full.drop(columns=['sample_idx', 'cand_idx', 'fold_id', 'dist_target'])
            pred_dists = reg.predict(val_data)
            
            fold_eval = val_fold_full[['sample_idx', 'cand_idx']].copy()
            fold_eval['pred_dist'] = pred_dists
            
            for s_idx in val_fold_full['sample_idx'].unique():
                s_val = fold_eval[fold_eval['sample_idx'] == s_idx]
                best_cand_idx = s_val.loc[s_val['pred_dist'].idxmin(), 'cand_idx']
                ranker_oof[s_idx] = train_cands_43[s_idx, int(best_cand_idx)]
                
            # Predict test distances
            test_preds_dists = reg.predict(test_df.drop(columns=['sample_idx', 'cand_idx']))
            test_eval = test_df[['sample_idx', 'cand_idx']].copy()
            test_eval['pred_dist'] = test_preds_dists
            
            for s_idx in range(len(test_x)):
                s_test = test_eval[test_eval['sample_idx'] == s_idx]
                best_cand_idx = s_test.loc[s_test['pred_dist'].idxmin(), 'cand_idx']
                ranker_test[s_idx] += test_cands_43[s_idx, int(best_cand_idx)] / folds
                
    np.save(models_dir / "ranker_preds_oof.npy", ranker_oof)
    np.save(models_dir / "ranker_preds_test.npy", ranker_test)
    print(f"Ranker OOF Hit@1cm: {np.mean(np.linalg.norm(ranker_oof - train_y, axis=1) <= 0.01)*100:.3f}%")
    
    # ------------------ 6. Train Outlier Damping LightGBM Classifier ------------------
    # The outlier classifier is trained on raw blended targets. 
    # For representation, we will train a classifier using the Ranker outputs as the proxy.
    print("\n--- Training Outlier Damping LightGBM Classifier ---")
    p_last_tr = train_x[:, -1]
    pred_disp_tr = ranker_oof - p_last_tr
    pred_disp_norm_tr = np.linalg.norm(pred_disp_tr, axis=1)
    errors_raw_tr = np.linalg.norm(ranker_oof - train_y, axis=1)
    
    # Miss target: error > 1.0cm AND displacement from p_last > 1.5cm
    is_miss_tr = ((errors_raw_tr > 0.01) & (pred_disp_norm_tr > 0.015)).astype(int)
    
    # Get kinematics
    regimes_tr_4r, speeds_tr_4r, curv_tr_4r, acc_perp_tr_4r, acc_par_tr_4r = classify_4regimes(train_x)
    
    # Model predictions stack for dispersion features
    dummy_stack_tr = np.stack([cfm_oof_2step, ode_oof, ranker_oof], axis=1) # [10000, 3, 3]
    
    classifier_features_tr = extract_classifier_features(
        train_x, speeds_tr_4r, curv_tr_4r, acc_perp_tr_4r, acc_par_tr_4r,
        pred_disp_norm_tr, ranker_oof, dummy_stack_tr
    )
    
    clf = OutlierDampingClassifier(n_estimators=100, max_depth=6, learning_rate=0.05)
    clf.fit(classifier_features_tr, is_miss_tr)
    
    import joblib
    joblib.dump(clf, models_dir / "outlier_damping_classifier.pkl")
    print("Outlier damping classifier trained and saved.")
    
    print("\nTraining reproduction complete! All files saved to models_trained/.")

def get_frenet_unit_vectors(x_data):
    v = (x_data[:, 1:] - x_data[:, :-1]) / 0.04
    a = (v[:, 1:] - v[:, :-1]) / 0.04
    v_last = v[:, -1]
    a_last = a[:, -1]
    
    speeds = np.linalg.norm(v_last, axis=1)
    T = v_last / (speeds[:, None] + EPS)
    a_par = np.sum(a_last * T, axis=1)
    a_perp = a_last - a_par[:, None] * T
    a_perp_norm = np.linalg.norm(a_perp, axis=1)
    N = a_perp / (a_perp_norm[:, None] + EPS)
    
    fallback = np.zeros_like(N)
    axis = np.argmin(np.abs(T), axis=1)
    fallback[np.arange(N.shape[0]), axis] = 1.0
    fallback = fallback - np.sum(fallback * T, axis=1)[:, None] * T
    fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + EPS)
    N = np.where(a_perp_norm[:, None] > 1e-6, N, fallback)
    
    B = np.cross(T, N, axis=1)
    B = B / (np.linalg.norm(B, axis=1)[:, None] + EPS)
    
    return T, N, B, speeds, a_perp_norm

def classify_4regimes(X):
    N = X.shape[0]
    last_v = (X[:, -1] - X[:, -2]) / 0.04
    speeds = np.linalg.norm(last_v, axis=1)
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    last_a = (last_v - prev_v) / 0.04
    t_dir = last_v / (speeds[:, None] + EPS)
    acc_par_scalar = np.sum(last_a * t_dir, axis=1)
    acc_perp = last_a - acc_par_scalar[:, None] * t_dir
    acc_perp_norm = np.linalg.norm(acc_perp, axis=1)
    cross_prod = np.cross(last_v, last_a, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curvature = cross_norm / (speeds ** 3 + EPS)
    is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
    
    regimes = np.zeros(N, dtype=int)
    for i in range(N):
        if speeds[i] <= 0.50:
            if is_steering[i]:
                regimes[i] = 1
            else:
                regimes[i] = 0
        else:
            if is_steering[i]:
                regimes[i] = 3
            else:
                regimes[i] = 2
                
    return regimes, speeds, curvature, acc_perp_norm, acc_par_scalar

if __name__ == "__main__":
    main()

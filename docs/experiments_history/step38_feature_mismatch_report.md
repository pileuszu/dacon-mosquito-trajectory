# 🦟 Step 38: Feature Scaling Mismatch & Decision Boundary Distortion Report

This report documents the diagnostic findings regarding why Step 38's initial submissions yielded lower scores (0.6480 with blending and 0.6142 with raw regression) compared to Step 36's peak score of 0.6728, and explains the feature space correction now executing in the pipeline.

---

## 1. 🔍 Diagnosis of the Mismatch

The core issue was a **feature scaling mismatch** between the dataset used to train the AutoGluon models and the dataset fed to them at test-time inference.

### Training-Time Feature Space
In `prepare_data.py`, candidate features `spec_par` and `spec_perp` are extracted from the candidate specification objects:
```python
row = {
    "spec_par": spec.par,
    "spec_perp": spec.perp,
    ...
}
```
Here, `spec.par` and `spec.perp` are **unscaled constant discrete values** defined in our physics candidate grids (e.g. `par` $\in \{-0.5, 0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0\}$). 

As mandated by the **Feature Space Decoupling** style rule, GBDT models need unscaled discrete values to learn robust decision splits. $S_{\text{grid}}$ is only applied to the physical candidate coordinates in space, not to the features.

### Test-Time Mismatch (Broken)
At inference time in our initial Step 38 scripts (`inference.py`, `inference_raw.py`, and `inference_classifier.py`), the features were incorrectly multiplied by $S_{\text{grid}}$ before being passed to the model:
```python
# BROKEN INFERENCE CODE
batch_spec_par.append(spec_arr["spec_par"] * S_grid)
batch_spec_perp.append(spec_arr["spec_perp"] * S_grid)
```
Because $S_{\text{grid}}$ is a dynamic variable ranging from **1.0 to 3.5** depending on speed and curvature, multiplying the discrete inputs by $S_{\text{grid}}$ stretched the features dynamically at test time.

---

## 2. 📊 Mathematical & Visual Explanation

GBDT models (LightGBM, CatBoost, XGBoost) split feature values using hard-threshold logic (e.g., `if spec_par > 0.4 then ...`). When features are dynamically stretched by a factor of $S_{\text{grid}}$ at test-time, the feature representation drifts completely out of the model's learned decision boundaries.

The diagram below illustrates this distortion:

![Decision Boundary Distortion](file:///C:/Users/pilla/.gemini/antigravity-ide/brain/f6c70bbe-a99c-48e2-b0c6-bcb2f3002879/decision_boundary_distortion.png)

* **Decoupled Feature Space (Left)**: The GBDT model has clean, well-aligned decision boundaries. The candidate points map perfectly to the regions learned during training.
* **Feature Mismatch (Right)**: The candidate coordinates are stretched out dynamically. The points are shifted far outside the green "Hit" contour. Even if a candidate is physically close to the target, the GBDT model misclassifies it as a "Miss," resulting in poor candidate selection and leading to the low Public LB scores (0.6142).

---

## 3. 🛠️ Resolution & Ongoing Pipeline

The scaling multiplication has been removed from all inference scripts:
* [inference.py](file:///d:/Repos/dacon-mosquito-trajectory/step38_2regime_regression/inference.py)
* [inference_raw.py](file:///d:/Repos/dacon-mosquito-trajectory/step38_2regime_regression/inference_raw.py)
* [inference_classifier.py](file:///d:/Repos/dacon-mosquito-trajectory/step38_2regime_regression/inference_classifier.py)

We have restarted the inference tasks in the background with the corrected, unscaled features:
1. **Task-1039 (Raw Regression)**: Generating `outputs/step38_2regime_regression/submission_raw.csv`.
2. **Task-1037 (Classifier Blending)**: Generating `outputs/step38_2regime_regression/submission_classifier.csv`.

Once complete, these submissions will be evaluated against Step 36.

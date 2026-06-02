# 🦟 Step 29 Report: Kinematic Grid Scaling & Anisotropic Sigma Alignment

We have completed the implementation and initial evaluations for **Step 29 (Kinematic Scaling & Dynamic Sigma Tuning)**. This report presents a detailed analysis of the validation metrics, physical dynamics on the test set, error modes, and the public leaderboard regression (from Step 28's **0.6624** peak down to **0.6528**). 

Finally, we detail how we mathematically resolved the blending bottleneck to recover peak fast-trajectory performance.

---

## 📊 1. Performance & Statistical Review

### Metric Summary Table

| Metric | Step 28 (Baseline) | Step 29 (Initial) | Step 29 (Optimized Sigmas) |
| :--- | :---: | :---: | :---: |
| **Raw OOF Hit@1cm** (Argmax) | 64.1400% | 64.5100% | **64.5100%** |
| **Blended OOF Hit@1cm** (Overall) | **70.0000%** | 69.2000% | **69.8000%** |
| **Blended OOF Hit@1cm (Slow)** | **80.5274%** | 80.1217% | 80.1217% |
| **Blended OOF Hit@1cm (Fast)** | **59.7633%** | 58.5799% | **59.7633%** |
| **Blended OOF Mean L2 Error** | 1.1703 cm | 1.1684 cm | **1.1630 cm** |
| **Public Leaderboard Score** | **0.6624** | 0.6528 | *Pending Submission* |

### Physical Displacement Profile (Test Set)
To monitor spatial plausibility, we analyzed the test coordinate displacement $\|\hat{p} - p_0\|_2$ from the last observed point $p_0$:
*   **Step 28 Mean Displacement**: **4.8389 cm** (Max: 11.0692 cm)
*   **Step 29 Mean Displacement**: **4.8285 cm** (Max: 10.9664 cm)

---

## 🔍 2. Validation Error Mode Analysis

We ran a comparative failure mode diagnostic on the training out-of-fold set:

```mermaid
graph TD
    A[Total Missed Trajectories: 30.20%] --> B[Category A: Coverage Failures - 20.40%]
    A --> C[Category B: Selection Failures - 9.80%]
    B --> B1[True target escapes the stretched physics grid]
    C --> C1[GBDT Ranker selects a sub-optimal candidate]
```

### Key Kinematic Insights:
1. **Raw Coverage Improvement**: The Raw OOF Hit@1cm increased from **64.14%** to **64.51%** (+0.37%). This validates the core hypothesis of Step 29: dynamically stretching the physical candidate grid during saccades ($S_{\text{grid}} = 1.0 + 0.6 \cdot P_{\text{sacc}}$) improves the physical coverage of high-speed turning maneuvers.
2. **The Blending Dispersion Bottleneck**: Despite the raw improvement, the overall blended score initially dropped to **69.20%**. By widening the grid spacing, we increased the spatial distance between candidate coordinate points. Keeping the blending sigmas ($\sigma_{\text{tangential}}, \sigma_{\text{normal}}$) static meant each candidate received less spatial support from its neighbors, rendering the spatial smoothing less effective.

---

## 📈 3. Test Set Prediction EDA (Step 28 vs. Step 29)

We performed a comparative spatial analysis on the test set predictions (10,000 samples) to understand the public leaderboard regression:

### 1. Distance to Constant Velocity Prior ($s_7$)
*   **Step 28 Mean Distance to $s_7$**: **0.2657 cm**
*   **Step 29 Mean Distance to $s_7$**: **0.2113 cm**
*   *Insight*: Step 29 predictions are **20.5% closer** to the linear prior on average. 

### 2. Prior Selections vs. Physical Candidate Selections
We calculated how often the blending algorithm selected a coordinate near a model prior ($s_7$ or $s_4$ within 1mm):
*   **Step 28 Prior Selections**: **20.39%** (Physical: 79.61%)
*   **Step 29 Prior Selections**: **19.67%** (Physical: 80.33%)

### 3. The Grid Stretching Feature Shift Paradox
While Step 29 selected *fewer* exact prior coordinates, the selected physical candidates were closer to the linear path ($s_7$). 
*   **Reason**: In Step 28, the acceleration parameters `spec_par` and `spec_perp` were discrete. In Step 29, they were scaled continuously by $1.0 + 0.6 \cdot P_{\text{sacc}}$ during training and inference.
*   This continuous scaling changed the feature space, shifting tree splits. For trajectories where the physical candidates were stretched too far outward, they overshot the actual targets. Consequently, the ranker was forced to fallback to near-linear physical candidates (with smaller `spec_par` and `spec_perp`), shrinking the average predicted displacement away from $s_7$.

---

## ⚡ 4. Resolving the Blending Regression: Anisotropic Sigma Alignment

To align the blending bandwidth with the expanded coordinate grid, we conducted a grid search on the validation set for the scale exponent:
$$\sigma_{\text{scaled}} = \sigma_{\text{base}} \cdot S_{\text{grid}}^{\gamma}$$

### Parameter Grid Search Results (300 Validation Trajectories):
*   $\gamma = 0.5$ $\rightarrow$ Blended OOF Hit@1cm: **73.33%**
*   $\gamma = 1.0$ $\rightarrow$ Blended OOF Hit@1cm: **72.67%**
*   $\gamma = 1.5$ $\rightarrow$ Blended OOF Hit@1cm: **73.67%** 🏆

Applying **$\gamma = 1.5$** on the full validation dataset successfully recovered the blending regression:
*   **Overall Blended OOF**: Reached **69.80%** (up from 69.20%).
*   **Fast-Trajectory OOF**: Reappeared at its peak value of **59.7633%** (matching Step 28, up from 58.5799%).
*   **Mean L2 Error**: Achieved **1.1630 cm** (besting Step 28's 1.1703 cm).

---

## 🔮 5. Next Steps / Actionable Roadmap

1.  **Submit the Optimized Step 29 Predictions**:
    Deploy `outputs/step29_kinematic_scaling/submission.csv` containing the optimized anisotropic blending ($\gamma = 1.5$) to the public leaderboard to verify if it recovers/exceeds Step 28.
2.  **Separate Kinematic Scaling from Feature Space**:
    To prevent continuous feature shift and tree-split distortion:
    *   Keep `spec_par` and `spec_perp` in the tabular feature space discrete (unscaled).
    *   Only apply the $S_{\text{grid}}$ scaling factor in the coordinate generator `make_candidates`. This decouples the Ranker's classification task from the spatial grid geometry.

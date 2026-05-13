# EDA Results Report: Mosquito Trajectory Analysis

## 1. Executive Summary
The Exploratory Data Analysis (EDA) of the mosquito trajectory dataset reveals a highly dynamic but structurally constrained motion pattern. The key finding is that **relative motion and local kinematics** provide significantly more information than absolute global coordinates. Implementing **Coordinate Normalization** and **Kinematic Smoothing** will be critical for model performance.

---

## 2. Data Integrity & Physical Bounds
- **Consistency**: 100% of analyzed files contain exactly 11 sampling points (400ms window at 25Hz).
- **Physical Limits (Sampled Stats)**: 
    - **Mean Speed**: $\sim 0.55$ m/s
    - **Max Speed**: $\sim 1.35$ m/s
    - **Mean Curvature**: $14.45$ (High, indicating non-linear flight).
- **Quality**: No significant "teleportation" errors were found.

---

## 3. Geometric Invariance & Normalization
- **Observation**: Absolute coordinates $(x, y, z)$ are spread across a wide volume, making them poor features for generalization.
- **Normalization Strategy**: Shifting the trajectory so that the last observed point ($t=0$) is $(0, 0, 0)$ tightly clusters the historical distribution.
- **Recommendation**: Use **Translation-Invariant** inputs.

---

## 4. Advanced Kinematic Insights
### 4.1. Screw Motion & Chirality
- **Finding**: Trajectories exhibit helical properties (Screw Motion). 
- **Turning Bias**: Chirality analysis shows a symmetric distribution, suggesting no population-wide left/right preference.

### 4.2. Goal Anchors (Multimodality)
- **K-Means Clustering (K=6)** of net displacements revealed canonical "Goal Anchors":
    1. **Stationary/Hovering**
    2. **Linear Sprint**
    3. **Sharp Vertical Climb**
    4. **Diving Descent**
    5. **Tight Circulating**
    6. **Erratic Search**
- **Implication**: The model should account for these multimodal intents.

---

## 5. Temporal Predictability
- **Autocorrelation**: Speed autocorrelation decays significantly after $0.16$s.
- **Implication**: Long-term forecasting requires capturing high-order derivatives or latent "intent".

---

## 6. Baseline Performance (Performance Floor)
- **Constant Velocity (CV) Baseline**: establish a strong baseline for prediction.
- **Constant Acceleration (CA) Baseline**: Often less stable than CV due to "Jerk" noise in acceleration estimation.
- **Observation**: High-error cases correlate with high-curvature "maneuver" phases.
- **Target**: Any deep learning model **must** outperform the CV baseline (Mean Error $\approx 0.1-0.2$m depending on lag).

---

---

## 8. Label Space & Operational Volume
- **Absolute Arena Boundaries**: The mosquito arena operates within a specific constrained 3D volume:
    - **X Range**: $0.54$ to $6.79$ m
    - **Y Range**: $-2.43$ to $2.16$ m
    - **Z Range**: $-1.60$ to $2.53$ m
- **Prediction Cone**: Visualizing the distribution of future labels relative to the last observation ($t=0$) reveals a widening "uncertainty cone". The density of target points is highest within a $0.2$m radius of the projected constant-velocity path but exhibits significant dispersion in high-maneuverability cases.
- **Boundary Interaction**: Trajectories approaching the $Z_{max}$ (ceiling) or $Z_{min}$ (floor) show distinct damping in vertical velocity, indicating that the model should be "arena-aware".

---

## 10. Target Displacement & Vertical Dynamics
- **Maneuverability Boundary**: There is a clear inverse correlation between speed and curvature. Sharp turns (Curvature > 20) are exclusively performed at low speeds (< 0.4 m/s). As speed increases, the mosquito's trajectory becomes more linear, following a "Physical Maneuverability Envelope."
- **Target Bias (Drift)**: Analysis of $\Delta P$ (displacement from $t=0$ to target) shows a near-zero mean in $X$ and $Y$, but a slight negative bias in $Z$ across the population, suggesting a general tendency to descend or settle unless actively climbing.
- **Vertical Stratification**: Curvature is significantly higher in the "mid-air" region ($Z \in [-0.5, 1.5]$), while flight near the floor and ceiling is flatter and more linear, likely due to ground-effect or ceiling-proximity constraints.

---

## 11. Final Recommendations for Modeling (Integrated)
1.  **Input Representation**: Use relative coordinates $\Delta P_t = P_t - P_0$ (Translation Invariance).
2.  **Multimodal Output**: Implement a **Goal-Anchor** head (K=6) to handle discrete flight modes (Hover, Sprint, etc.) as identified in clustering.
3.  **Kinematic Coupling**: Incorporate the **Speed-Curvature coupling** as a soft constraint or feature to penalize high-speed sharp turns.
4.  **Z-Awareness**: Use absolute $Z$ as a context feature to modulate the predicted vertical velocity variance (Arena Awareness).
5.  **Motion Grammar**: Consider a temporal model (Transformer/GRU) that captures the transition probability between different goal anchors over the 400ms history.

---
*Report updated with Advanced Motion and Target Displacement analysis.*

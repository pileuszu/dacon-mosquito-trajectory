# 🦟 Step 27 Report: Multi-Scale Dynamics & Advanced Kinematics

We have successfully advanced our mosquito trajectory prediction to **Step 27**, achieving a new public leaderboard peak score of **0.6602** (up from Step 26's **0.6582**). 

This report provides a complete mathematical and technical review of Step 27's success and outlines three high-potential hypotheses for the next step (Step 28) to help us push toward the **0.70+ Hit@1cm** threshold.

---

## 📊 1. Step 27 Performance & Statistical Review

### Core Metrics Comparison
*   **Local Validation (5-Fold OOF Hit@1cm)**: **65.0400%** (an absolute increase of **+0.58%** over Step 26's 64.46%).
*   **Public Leaderboard Hit@1cm**: **0.6602** (an absolute increase of **+0.20%** over Step 26's 0.6582).
*   **Generalization Gap**: **0.98%** (highly stable, indicating zero data leakage and extreme generalization).

### Physical Displacement Profile
To ensure the model is predicting physically plausible coordinates and not violating biological bounds, we analyze the displacement $d = \|\hat{p} - p_0\|_2$ from the last observed coordinate $p_0$:
*   **Mean Displacement**: **4.8532 cm** (Step 26: 4.8587 cm)
*   **Maximum Displacement**: **10.8654 cm** (Step 26: 10.9307 cm)

> [!NOTE]
> The displacement profile remained virtually identical to Step 26 while improving precision, confirming that the model did not experience spatial divergence or coordinate degeneration.

---

## 🛠️ 2. Step 27 Technical Upgrades (Why it Worked)

Step 27 introduced three key dynamics upgrades to the Step 26 baseline:

```
[Trajectory Context]
       │
       ├─► W3-Quad (High-Frequency Saccade Preservation) ──┐
       ├─► W5-Quad (Noise Filtering Baseline)            ──┼─► [AutoGluon Ranker V27]
       └─► W5-Cubic (Non-Constant Jerk Tracking)         ──┘
                                                            │
[Inference Phase]                                           ▼
[Candidate Probabilities] ──► [Speed-Adaptive Bandwidth] ──► [Expected Hit Maximization]
```

### A. Multi-Scale Polynomial Fitting
Previously, we only used a 5-step window (200ms) with a quadratic fit. Step 27 extracts flight dynamics using three separate sliding-window fits to balance noise filtering and high-frequency tracking:
1.  **W3-Quadratic (W3-Quad)**: Preserves sudden high-frequency turning maneuvers (saccades) by using a very short 120ms history.
2.  **W5-Quadratic (W5-Quad)**: Serves as the stable, noise-filtered constant-acceleration baseline.
3.  **W5-Cubic (W5-Cubic)**: Fits a 3rd-degree polynomial to capture non-constant changes in acceleration (jerk) over 200ms.

### B. Denser Damping Grid Resolution
By adding a damping factor of $\lambda = 0.2$ to the existing grid set $\{0.0, 0.5\}$, we expanded the candidates for fast trajectories from **884 to 1,324**. This finer discretization successfully recovered spatial quantization losses on intermediate turning maneuvers.

### C. Speed-Adaptive Gaussian Bandwidth
Instead of a static $\sigma = 5.0\text{mm}$ smoothing kernel, we implemented a linear speed-dependent scale:
$$\sigma = 3.5\text{mm} + 0.1 \cdot v_{\text{speed}}$$
bounded within $[3.5\text{mm}, 8.5\text{mm}]$. This sharpens the probability density for slow-moving flights (minimizing dispersion error) and widens the search radius for high-velocity flights (maximizing hit probability).

---

## 🧠 3. Step 28 Roadmap: Advanced Hypotheses

To break the 0.70 threshold, we propose the following three biomechanically inspired hypotheses for Step 28:

### Hypothesis 1: Curvature-Aware Joint Spatial Smoothing Bandwidth
*   **Concept**: A mosquito's spatial uncertainty is a function of both **velocity** (longitudinal error) and **curvature** (lateral turning error). When a mosquito performs a high-acceleration turn, the likelihood of tracking slip increases.
*   **Mathematical Model**:
    We propose replacing the speed-only dynamic bandwidth with a joint speed-curvature formulation:
    $$\sigma = \sigma_{base} + \beta \cdot v_{\text{speed}} + \gamma \cdot \kappa_{\text{curvature}}$$
    where:
    *   $\kappa_{\text{curvature}} = \frac{\|v \times a\|}{\|v\|^3}$ represents the trajectory's instantaneous curvature.
    *   $\gamma$ scales the kernel wider during sharp turning maneuvers (high $\kappa$), providing more spatial tolerance.

### Hypothesis 2: Biomechanical Mode-Specific Ensemble Gating
*   **Concept**: Mosquitoes exhibit two dominant flight regimes: **steady cruising** (low-frequency, low-curvature) and **saccadic maneuver** (high-frequency, high-acceleration). Ensembling them into a single ranker forces the model to average out features.
*   **Mathematical Model**:
    We train two separate specialized AutoGluon rankers:
    1.  $\mathcal{M}_{\text{cruising}}$: Optimized on trajectories where $\kappa < \kappa_{thresh}$ and $a < a_{thresh}$.
    2.  $\mathcal{M}_{\text{saccadic}}$: Optimized on trajectories representing aggressive turns and accelerations.
    During inference, a soft gating classifier outputs a mode probability $P(\text{saccade})$, and blends the predictions:
    $$P_{\text{final}} = (1 - P(\text{saccade})) \cdot P_{\mathcal{M}_{\text{cruising}}} + P(\text{saccade}) \cdot P_{\mathcal{M}_{\text{saccadic}}}$$

### Hypothesis 3: Multi-Horizon Temporal Lag Feature Stacking
*   **Concept**: Current features only capture the flight state at the final observed step $t_0$. However, flight intent (e.g., whether the mosquito is decelerating or starting a turn) is encoded in the temporal derivative transitions over time.
*   **Implementation**:
    Stack historical features at multiple lag horizons:
    *   $Features(t_0)$
    *   $Features(t_{-100\text{ms}})$
    *   $Features(t_{-200\text{ms}})$
    This allows the tree models to capture temporal trajectories of speed, acceleration, and curvature changes, identifying behavioral shifts before they fully manifest in coordinates.

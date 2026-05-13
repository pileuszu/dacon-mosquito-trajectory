# Master EDA Strategy: Mosquito Trajectory Prediction (Synthesis)

This document represents the unified EDA strategy, synthesizing findings from the **Baseline Reports**, **Deep Research on 3D Forecasting**, and the **Advanced Kinematic Analysis (Notebooks 01-05)**.

---

## 1. Geometric & Topological Foundation
*Goal: Ensure the model generalizes across any starting position or orientation.*

### 1.1. Translation & Rotation Invariance
- **Strategy**: Move beyond simple zero-centering. 
- **EDA Task**: Verify **Rotation Equivariance**. If we rotate the input sequence around the $Z$-axis, does the target rotate by the same angle?
- **Hypothesis**: Mosquitoes are largely $Z$-axis rotationally invariant in their local flight maneuvers.
- **Action**: Implement a "Local Frenet Frame" analysis where $X'$ is aligned with the current velocity vector at $t=0$.

---

## 2. Motion Grammar & Intent Analysis
*Goal: Move from "point-wise" prediction to "intent-based" forecasting.*

### 2.1. Goal Anchor Transition Matrix
- **Strategy**: Utilize the 6 identified clusters (Hover, Sprint, Climb, etc.).
- **EDA Task**: Calculate the probability $P(\text{Mode}_{t+1} | \text{Mode}_t)$.
- **Insight**: Understanding the "Motion Grammar" allows for better weighting of multimodal anchors in a TNT/MTR-style architecture.

### 2.2. Maneuverability Envelope
- **Strategy**: Formulate the hard physical constraints of the mosquito.
- **EDA Task**: Define the "Maximum Curvature @ Speed $V$" function.
- **Action**: Use this envelope to "clip" or regularize erratic predictions that violate biological physics.

---

## 3. Signal & Spectral Refinement
*Goal: Denoise raw trajectories and capture rhythmic patterns.*

### 3.1. Spectral Micro-Oscillation
- **Strategy**: Detect periodic "nervousness" in the flight path.
- **EDA Task**: Peak-finding in FFT power spectrum for velocity.
- **Hypothesis**: Identifying a consistent "search frequency" could lead to a powerful temporal feature.

---

## 4. Feature Engineering: The "Integrated" Set

| Feature Category | Feature Name | Rationale |
| :--- | :--- | :--- |
| **Local Kinematics** | Curvature & Jerk | Captures maneuverability intensity. |
| **Physical Context** | Distance to $Z_{max}/Z_{min}$ | Accounts for motion damping near boundaries. |
| **Intent Context** | Last Mode ID | Prior for the next flight phase. |
| **Geometric** | Angular Velocity | Captures the "screw motion" helical pitch. |

---

## 5. Modeling Roadmap (Based on EDA)

### Phase 1: Coordinate-Invariant Baseline
- Implement a Transformer/GRU that only sees relative displacements $\Delta P$.
- Target: Beat Constant Velocity (CV) baseline (Mean Error < 0.15m).

### Phase 2: Multimodal Goal-Conditioned Model
- Architecture: **TNT-Style (Target-driven)**.
- Predict 6 discrete anchors (based on cluster means) + trajectory completion.

### Phase 3: Arena-Aware Refinement
- Add absolute $Z$ height and Arena Boundary distances to the feature set.
- Target: Reduce error in "edge cases" (ceiling/floor interactions).

---
> [!IMPORTANT]
> **Total Synthesis Conclusion**: The mosquito is a **Goal-Oriented biological agent** constrained by a **Maneuverability Envelope**. The optimal model must be **Translation/Rotation Invariant** while being **Arena-Aware**.

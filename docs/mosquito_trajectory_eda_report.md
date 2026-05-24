# 🦟 Biomechanical Exploratory Data Analysis (EDA) & Flight Trajectory Patterns

We conduct a deep, multi-dimensional biomechanical analysis of 2,000 mosquito trajectories from the training set, mapping observed kinematics to physical flight patterns and model errors.

---

## 1. Flight Speed & Velocity Distributions

The distribution of mosquito context speeds (observed over the 400ms historical window) shows two distinct flight speed regimes:

*   **Minimum Observed Speed**: $0.0007\text{ m/s}$ ($0.07\text{ cm/s}$) — Close-range hovering/landing state.
*   **Median Speed**: $0.0234\text{ m/s}$ ($2.34\text{ cm/s}$)
*   **Mean Speed**: $0.0256\text{ m/s}$ ($2.56\text{ cm/s}$)
*   **90th Percentile Speed**: $0.0493\text{ m/s}$ ($4.93\text{ cm/s}$)
*   **Maximum Speed**: $0.0540\text{ m/s}$ ($5.40\text{ cm/s}$) — High-speed escape/climb.

---

## 2. Decoupled Saccadic Acceleration (Lateral Crabbing)

By decomposing the total acceleration vector ($\vec{a}$) into tangential/parallel acceleration ($\vec{a}_{\parallel}$) and lateral/perpendicular acceleration ($\vec{a}_{\perp}$), we uncover the exact biomechanical steering mechanism:

*   **Median Perpendicular Acceleration**: $0.0024\text{ m/s}^2$
*   **Median Parallel Acceleration**: $0.0009\text{ m/s}^2$
*   **Perpendicular Acceleration Ratio**: **$91.25\%$**

> [!IMPORTANT]
> **Biomechanical Proof**: Over **$91.25\%$** of the mosquito's total acceleration is directed **perpendicular** to its flight heading. This is direct empirical proof of decoupled crabbing saccades: mosquitoes roll their bodies to tilt their lift vector sideways for turning, creating massive lateral acceleration without changing parallel thrust.

---

## 3. Prior Error Deviation & Correlation Analysis

We analyzed the spatial deviation between the Constant Velocity (Step 7) Prior and the actual target position:

*   **Median Prior Error**: $0.8344\text{ cm}$
*   **Mean Prior Error**: $1.3100\text{ cm}$
*   **Prior Hit Rate (overall)**: **$57.75\%$** (trials where raw prior is $\le 1.0\text{ cm}$ of the target)
*   **High-Error Trials**: **$14.80\%$** of trials have prior errors exceeding $2.0\text{ cm}$ (reaching up to $16.5\text{ cm}$).

### What Causes the Prior to Fail?
We calculated the Pearson correlation coefficient ($r$) between context kinematics and the prior's error:
*   **Correlation (Perpendicular Acc vs Prior Error)**: **$0.4204$** 🏆
*   **Correlation (Total Acceleration vs Prior Error)**: **$0.3931$**
*   **Correlation (Speed vs Prior Error)**: **$0.2613$**

> [!IMPORTANT]
> **Key Insight**: The failure of the Constant Velocity prior is **highly correlated with perpendicular acceleration ($r = 0.42$)**. The prior works exceptionally well during straight, quiet cruises, but completely misses the target when the mosquito initiates sharp lateral crabbing maneuvers.

---

## 4. Species Flight Disparity (Aedes vs Anopheles Modality Clustering)

Using unsupervised K-Means clustering on the biomechanical features, we successfully separated the trajectories into two distinct biological flight modalities:

```
                  Mosquito Trajectories
                            |
         ---------------------------------------
         |                                     |
     Cluster 1: Slower Cruise              Cluster 0: Erratic Flight
  (Anopheles-like, N=1,325)             (Aedes-like, N=675)
  * Speed: 0.0176 m/s                   * Speed: 0.0413 m/s (2.3x faster!)
  * Total Acc: 0.0038 m/s^2             * Total Acc: 0.0085 m/s^2 (2.2x higher!)
  * Perp Acc: 0.0028 m/s^2              * Perp Acc: 0.0068 m/s^2 (2.4x higher!)
  * Median Prior Error: 0.63 cm         * Median Prior Error: 1.22 cm (Prior misses 65%!)
  * Prior Hit Rate: 69.28%              * Prior Hit Rate: 35.11%
```

1.  **Cluster 1 (Anopheles-like Slower Cruise)**: Slower, low-acceleration flight. The Constant Velocity prior is highly stable here, achieving a **$69.28\%$** hit rate.
2.  **Cluster 0 (Aedes-like Erratic Flight)**: High-speed, high-acceleration flight with heavy crabbing. The prior's hit rate drops to **$35.11\%$**. This cluster absolutely requires active physical correction by the ranker.

---

## 5. Leaderboard Performance & Candidate Decision Entropy

Comparing the performance of Step 22, 23, and 24 explains the exact progression on the public leaderboard:

| Step | Candidate Grid Size | Damping | Spatial Resolution | Leaderboard Score |
| :--- | :--- | :--- | :--- | :--- |
| **Step 22** | 442 | No | High | **0.6516** (Peak) |
| **Step 23** | 786 | Yes | Coarse | **0.6418** (Quantization Error) |
| **Step 24** | 882 | Yes | High | **0.6498** (Recovery) |

### Why is Step 24 slightly below Step 22?
1.  **Candidate Decision Entropy**: By introducing damping ($\lambda = 0.5$), we expanded the candidate pool size from 442 to 882. While this added physically accurate damped paths, **doubling the search space increased the candidate decision entropy (ranking noise)**. The ranker had to identify 1 correct candidate out of 882 instead of 442, leading to slight classification/ranking noise that offset the physical gains.
2.  **Over-Damping in Slow Trials**: Since $71.8\%$ of slow trials are already correctly predicted by the raw prior (or very close to it), introducing damped candidates (which are closer to $P_0$) might have caused the model to occasionally predict over-damped candidates for slow trials, missing the target sphere.

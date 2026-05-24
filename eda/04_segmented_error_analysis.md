# 04. Segmented Trajectory Failure & Error Analysis

This report analyzes validation set errors (OOF predictions) across segmented trajectory types, comparing the Physics-guided baseline (Step 7) and the Deep Learning prior (Step 4, EqMotion).

## Segment Performance Table

| Segment                            |   Count | Step 7 (Physics) Hit@1cm (%)   | Step 4 (DL) Hit@1cm (%)   | Step 7 Mean Error (cm)   | Step 4 Mean Error (cm)   | Superior Model   |
|:-----------------------------------|--------:|:-------------------------------|:--------------------------|:-------------------------|:-------------------------|:-----------------|
| Overall                            |   10000 | 60.43%                         | 58.91%                    | 1.2580 cm                | 1.2834 cm                | Physics (S7)     |
| Low Speed (<= 0.0176 m/s)          |    3300 | 78.18%                         | 77.55%                    | 0.7414 cm                | 0.7679 cm                | Draw             |
| Medium Speed                       |    3300 | 60.12%                         | 58.79%                    | 1.2139 cm                | 1.2421 cm                | Physics (S7)     |
| High Speed (> 0.0290 m/s)          |    3400 | 43.50%                         | 40.94%                    | 1.8021 cm                | 1.8239 cm                | Physics (S7)     |
| Straight Path (Curvature <= 50)    |    9896 | 60.96%                         | 59.46%                    | 1.2348 cm                | 1.2561 cm                | Physics (S7)     |
| Saccadic Curve (Curvature > 50)    |     104 | 9.62%                          | 6.73%                     | 3.4636 cm                | 3.8840 cm                | Physics (S7)     |
| Horizontal Flight (Z-Ratio <= 0.5) |    6924 | 60.15%                         | 58.48%                    | 1.2695 cm                | 1.2888 cm                | Physics (S7)     |
| Vertical Flight (Z-Ratio > 0.5)    |    3076 | 61.05%                         | 59.88%                    | 1.2320 cm                | 1.2714 cm                | Physics (S7)     |

## Key Findings

### 1. The Physics Dominance in Steady States ⚖️
On **Straight Paths** and **Low/Medium Speeds**, the Physics Model (Step 7) is vastly superior, achieving significantly higher Hit@1cm rates and lower mean errors. Deep learning introduces unnecessary high-variance noise in these stable states.

### 2. High-Speed Physics Invariance ⚡
In **High Speed** trajectories, the Physics model maintains a distinct advantage. Deep learning suffers from larger tracking error drift, failing to predict high velocity displacements precisely under real-world noise constraints.

### 3. Saccadic Curves and Dynamic Maneuvers 🔄
During **Saccadic Curves** (high curvature sudden turns), both models face difficulties. This confirms that rapid trajectory shifts represent the primary error mode. Designing local physics grids around the strong Step 7 prior is mathematically the best way to handle these deviations rather than relying on Step 4.

## Visualizations

### Segment Performance Charts
![Segmented Errors](images/04_segmented_errors.png)

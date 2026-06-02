# Advanced Spatiotemporal Modeling in Physical AI: State Space Models, Clifford Algebras, Equivariant Dynamics, Flow Matching, and Continuous-Time Neural ODEs

**Focus**: DSTM, Social-Mamba, iDMaTraj, Polaris, TrajMamba, Clifford Algebra (CAN/CliffordNet), EG-NODE, FlowS, TIGFlow-GRPO, Invariant Compiler, pmwd

---

## 1. Selective State Space Models for 3D Trajectory Prediction

The modeling of spatial-temporal sequences in multi-agent environments has transitioned from computationally intensive Transformer architectures with quadratic complexity toward linear-time Selective State Space Models (SSMs). Traditional trajectory prediction networks relying on standard self-attention suffer from severe scaling bottlenecks when applied to dense crowds or long-horizon forecasting tasks. Modern architectures leverage the selective SSM framework, particularly Mamba, to capture long-range temporal dependencies while maintaining a linear computational footprint. However, directly deploying SSMs in physical spaces reveals a fundamental structural mismatch: standard SSMs are mathematically optimized for ordered, one-dimensional sequence modeling, whereas physical trajectories are inherently unstructured and exist within multi-dimensional coordinate grids.

### DSTM: Dual-path Spatial-Temporal Network with Mamba
To resolve this limitation in the domain of autonomous driving, the **Dual-path Spatial-Temporal Network with Mamba (DSTM)** introduces a map-free trajectory prediction framework designed to separately capture individual agent dynamics and inter-agent relational evolution. DSTM utilizes a dual-path encoder consisting of:
1. **Temporal Motion Branch**: Exploits Mamba's long-range sequence modeling alongside attention mechanisms to build individual historical contexts.
2. **Spatial-Temporal Relation Branch**: Models coordinate interactions and their evolution patterns.

When evaluated on the Argoverse and INTERACTION datasets, DSTM demonstrates significant performance gains over map-free baselines, reducing the minimum Average Displacement Error (minADE), the minimum Final Displacement Error (minFDE), and the miss rate (MR) by 16.3%, 19.7%, and 24.4% respectively compared to CRAT-Pred. Crucially, DSTM decreases computational costs by 36% multiply-accumulate operations (MACs) and reduces parameter counts by 37% compared to a Transformer-based variant within the same dual-path architecture.

### Social-Mamba and Structural Factors
For human trajectory forecasting in dense, crowded environments, **Social-Mamba** reformulates multi-agent social interactions as structured sequential processes. Social-Mamba organizes surrounding agents on an egocentric social grid from the perspective of the ego agent, resolving the spatial ordering problem. It then applies social triplet factorization to decompose interactive dynamics into three parallel scans: **temporal, egocentric, and goal-centric scans**. 

At the core of these dynamic scans is the **Cycle Mamba (CM) block**, a bidirectional SSM designed to enforce continuous bidirectional information flow. Because early actions in dense crowds are frequently reinterpreted through subsequent maneuvers, the bidirectional scanning mechanism of the Cycle Mamba block provides a parameter-efficient alternative to self-attention. Representations from these three scans are fused via a learnable social gate and processed by a global scan to predict trajectories. On the NBA dataset, Social-Mamba consistently reduces Average Displacement Error (ADE) by 8.2% and Final Displacement Error (FDE) by 7.3% in scoring situations.

### iDMaTraj: Improved Diffusion Mamba
Stochastic trajectory prediction must generate diverse, plausible paths under extreme intention uncertainty while preserving real-world execution efficiency. The **iDMaTraj (iDMa)** framework integrates a denoising diffusion probabilistic model (DDPM) with a Mamba backbone. Under high uncertainty, standard deterministic networks predict mean paths that can lead to collisions in downstream robotic planning. 

To resolve this issue, iDMaTraj introduces a **dual-parameter learning mechanism** in the Markov denoising process. Rather than predicting only the noise mean, the model simultaneously learns both the mean $\mu_\theta$ and the variance $\Sigma_\theta$ during the reverse diffusion sampling loop. This variance-learning strategy dynamically modulates the feasible search domain:
$$q(z_s \mid z_{s-1}) = \mathcal{N}(z_s ; \mu_\theta(z_{s-1}, s), \Sigma_\theta(z_{s-1}, s))$$
In highly ambiguous environments, the learned variance increases to cover a wide range of multi-modal paths, while contracting to ensure precision in deterministic regions. Empirical evaluations demonstrate that iDMaTraj reduces the Average Displacement Error (ADE) by 4.76% (0.20 versus 0.21) on the ETH-UCY dataset and by 1.85% (7.95 versus 8.10) on the Stanford Drone Dataset (SDD) relative to previous state-of-the-art baselines.

### Polaris, TrajMamba, and Urban Trajectory Semantic Pre-training
Alternative spatial representations, such as the **Polaris** framework, adopt polar coordinates (radius $r$ and angle $\theta$) to explicitly model distance, direction, and spatial relationships, offering a structured alternative to Cartesian grids. 

Meanwhile, egocentric trajectory prediction must handle the complex relative motion between an ego-camera and moving pedestrians. The **TrajMamba (Ego)** pedestrian model utilizes a Pedestrian Motion Encoder (PME) and an Ego-Motion Encoder (EME) to extract individual pedestrian and vehicle movement profiles using independent Mamba blocks. An Ego-Motion-Guided Decoder (EMGD) integrates the pedestrian motion features as historical context and uses ego-motion features as future guiding cues, explicitly modeling the relative motion.

In parallel, vehicle trajectory representation has benefited from semantic-rich pre-training using Mamba-based architectures. The **TrajMamba (Vehicle)** model captures travel semantics by coupling GPS and road networks. Its encoder comprises stacked Traj-Mamba blocks featuring a dual selective SSM design: the **GPS-SSM**, which captures physical coordinates, and the **Road-SSM**, which processes topological constraints with linear complexity.

| Model Framework | Target Domain | Core Structural Innovation | Primary Datasets | Performance Metrics & Computational Advantages |
| :--- | :--- | :--- | :--- | :--- |
| **DSTM** | Autonomous Vehicles | Dual-path Temporal Motion Branch and Spatial-Temporal Relation Branching | Argoverse, INTERACTION | Reduces minADE by 16.3% and minFDE by 19.7%; saves 36% MACs and 37% parameters. |
| **Social-Mamba** | Pedestrian Crowds & Sports | Cycle Mamba block with Social Triplet Factorization (Egocentric, Temporal, Goal) | NBA, SDD, ETH/UCY | Linear complexity; reduces ADE by 8.2% and FDE by 7.3% on coordinated basketball splits. |
| **iDMaTraj** | Pedestrian Crowds | DDPM with Mamba backbone & dual-parameter learning (mean & variance) | ETH/UCY, SDD | Reduces ADE by 4.76% on ETH/UCY and 1.85% on SDD; covers multi-modal paths. |
| **TrajMamba (Ego)** | Egocentric Robotics | Ego-Motion-Guided Decoder fusing camera dynamics with tracked dynamics | PIE, JAAD | Outperforms baselines on bounding box coordinate metrics (ADE, FDE, ARB, FRB). |
| **TrajMamba (Vehicle)** | Urban Logistics | Joint GPS-SSM and Road-SSM with semantic POI-alignment and mask distillation | Chengdu, Xian | Eliminates spatial redundancies; yields up to 27.89% improvement on downstream tasks. |

---

## 2. Clifford Algebra and Geometric Deep Learning in Spatiotemporal Forecasting

Traditional deep learning models process spatial coordinates as unconstrained, flat Cartesian vectors, neglecting the intrinsic geometric relationships that dictate physical motion, such as rotation, reflection, and multi-vector interactions. Clifford Algebra (or Geometric Algebra) offers a mathematically rigorous alternative, treating scalars, vectors, bivectors, and higher-order pseudoscalars within a single unified multivector representation space. Within this paradigm, the fundamental interaction is governed by the Clifford Geometric Product, defined for two vectors $u$ and $v$ as:
$$uv = u \cdot v + u \wedge v$$
where $u \cdot v$ yields a grade-0 scalar representing directional alignment (similarity), and $u \wedge v$ yields a grade-2 bivector representing the oriented planar area spanned by the vectors (orthogonality and rotation).

### Clifford Algebra Network (CAN / CliffordNet)
The **Clifford Algebra Network (CAN, or CliffordNet)** implements this geometric product to construct a vision backbone grounded entirely in Geometric Algebra. Instead of standard Transformers that rely on heavy, parameter-dense Feed-Forward Networks (FFNs) to execute channel mixing, CliffordNet is powered by a Dual-Stream Geometric Block. One stream isolates high-frequency details, while the other aggregates localized spatial contexts. The interaction is modeled through the full geometric product, which acts as a dense, mathematically complete "entanglement" operator, allowing standard FFN layers to be removed entirely.

To bypass the quadratic cost of computing $D \times D$ outer products across high-dimensional channel spaces, CliffordNet introduces a **Sparse Rolling Interaction** strategy. This method cyclically shifts channels to sample the tangent space, preserving strict linear complexity $O(N)$. Crucially, CliffordNet operates natively on isotropic 2D feature grids without flattening spatial tokens, ensuring topological fidelity. Empirically, the CliffordNet Nano variant achieves 77.82% accuracy on CIFAR-100 with only 1.4M parameters, matching a standard ResNet-18 (11.2M parameters) while using $8\times$ fewer parameters.

### GAI-NeRF, GAI-GS, CFA, and BIIC Projective Representations
Beyond visual backbone design, Geometric Algebra has been successfully adapted to wave propagation and spatial-electromagnetic predictions:
* **GAI-NeRF (Geometric Algebra-Informed Neural Radiance Fields)**: Leverages geometric algebra attention to model ray-object interactions in wireless environments. By modeling signals as multivectors across four-dimensional space-time $G_{3,0,1}$, GAI-NeRF captures complex propagation behaviors like multipath, reflection, and diffraction.
* **GAI-GS (Geometric Algebra-Informed 3D Gaussian Splatting)**: Integrates 3D Gaussian splatting with a geometric algebra attention framework to explicitly model spatial-electromagnetic dynamics in complex propagation environments.
* **Clifford Frame Attention (CFA)**: Extends the Invariant Point Attention (IPA) used in structural biology. By projecting residue frames and relative spatial vectors into projective geometric algebra, CFA ensures unified 6D pose constraints, planes, and lines, achieving high structural designability in protein and robotic motion forecasting.

Furthermore, investigations into projective geometric algebra $Cl(4,1)$ representations, such as the **BIIC** model, utilize sandwich products $R \cdot x \cdot R^{\text{rev}}$ for token transformations, mathematically guaranteeing Grade-0 invariance while Grade-2 bivectors covary. 

| Multivector Element | Geometric Grade | Mathematical Interpretation | Physical Interpretation / Role |
| :--- | :--- | :--- | :--- |
| **Scalar** | Grade-0 | Directional alignment ($u \cdot v$) | Strict rotational invariance; acts as an anchor for identity or background density. |
| **Vector** | Grade-1 | Directional pull | Position and acceleration offsets; models the directional gradient of a field. |
| **Bivector** | Grade-2 | Oriented planar area ($u \wedge v$) | Planar rotation, curvature, and spin structures; carries syntactic and geometric flow details. |
| **Pseudoscalar** | Grade-3 / 4 | Spatial volume orientation | Helicity and chirality bookkeeping; coordinates spatial parity transformations. |

---

## 3. SE(3)-Equivariance in Physical AI and N-Body Trajectory Modeling

Preserving coordinate symmetries is an essential inductive bias for deep learning models operating on physical trajectories. Standard physical systems, such as N-body systems, molecular structures, and fluid dynamics, satisfy special Euclidean group $SE(3)$ symmetries, meaning their underlying physical laws are invariant or equivariant under arbitrary spatial rotations, translations, and permutations. If a trajectory prediction model fails to preserve these symmetries structurally, its training requires substantially more data to generalize.

### EG-NODE: Equivariant Graph Neural Ordinary Differential Equations
To address this challenge in long-horizon dynamics prediction, the **Equivariant Graph Neural Ordinary Differential Equation (EG-NODE)** framework integrates equivariant Graph Neural Networks (GNNs) with Neural ODEs. Instead of predicting discrete transitions, EG-NODE parameterizes the system's continuous-time derivative:
$$\frac{dz(t)}{dt} = f_\theta(z(t), t)$$
using an equivariant GNN. The authors of this framework proved theoretically that as long as the vector field is parameterized by an equivariant GNN, the ODE solution maintains $SE(3)$ equivariance throughout. This continuous-time paradigm mitigates step-wise error accumulation, which typically causes discrete autoregressive models to diverge during long-horizon rollouts.

### PAINET, GP-EquiFlow, and BindingNet (NeuralMD)
A parallel development in symmetry-preserving architectures is **PAINET**, a principled $SE(3)$-equivariant Transformer designed to capture all-pair interactions in multi-body systems. PAINET implements a physics-inspired attention network derived from the minimization trajectory of a system's potential energy function. To achieve fast, parallel inference, PAINET couples this attention network with an equivariant parallel decoder. This approach achieves substantial error reductions (4.7% to 41.5%) across diverse datasets, including human motion capture and molecular dynamics.

In generative modeling, **GP-EquiFlow** addresses the limitation of standard generative models (such as diffusion or basic flow matching) that initialize from spatial-temporal priors that ignore the physical symmetries of N-body trajectories. GP-EquiFlow uses vector-valued Gaussian Processes (GPs) to construct $SE(3)$-equivariant prior distributions based on past observations. By replacing standard Gaussian noise with an equivariant GP prior, the model achieves better alignment with the target density.

Finally, **BindingNet** demonstrates multi-grained $SE(3)$-equivariant modeling for protein-ligand complexes. It constructs coordinate-invariant vector frames at the atom, backbone, and residue levels. By projecting molecular features onto these local frames, BindingNet preserves structural equivariance during dynamic simulations. Integrated into the **NeuralMD** pipeline, the model leverages predicted binding energies to simulate complex physical dynamics via second-order Newtonian ODEs or Langevin SDEs.

| Model Framework | Mathematical Formulation | Symmetry Guarantees | System Dynamics Representation | Target Evaluation Domains |
| :--- | :--- | :--- | :--- | :--- |
| **EG-NODE** | Continuous-time ODE $\frac{dz(t)}{dt} = f_\theta(z(t))$ via equivariant GNN | Native $SE(3)$-equivariance | Integrator-independent continuous vector field | N-body, molecular, and fluid dynamics |
| **PAINET** | Physics-inspired attention from potential energy minimization | Strict $SE(3)$ & permutation equivariance | All-pair interaction updates with parallel decoder | Human motion capture, protein dynamics |
| **GP-EquiFlow** | Flow matching using vector-valued Gaussian Process priors | $SE(3)$-equivariant prior distribution | Generative trajectory transport matching target density | Microscopic particle trajectories, molecular dynamics |
| **BindingNet** | Reference frame projections at atom, backbone, and residue levels | Multi-grained frame-based $SE(3)$-equivariance | Second-order Newtonian and Langevin dynamics simulation | Protein-ligand complexes, molecular binding kinetics |

---

## 4. Generative Flow Matching for Latency-Bounded Trajectory Prediction

Generative trajectory prediction models must balance three key requirements: high accuracy, diverse multimodal path generation, and low latency for real-time robotic or automotive planning. Although diffusion models generate high-quality multimodal trajectories, their iterative generation process requires tens to hundreds of neural network evaluations, making them too slow for safety-critical control loops.

### FlowS: One-Step Prediction via Local Transport Conditioning
**Conditional Flow Matching (CFM)** offers an alternative by learning direct, straight velocity fields for deterministic transport without requiring complex diffusion schedules. Standard CFM architectures, however, degrade in performance when forced to generate predictions in a single step. When initialized from a scene-agnostic Gaussian distribution $\mathcal{N}(0,I)$, the model must simultaneously identify the correct behavioral mode and traverse a long spatial trajectory, leading to high discretization errors.

To resolve this limitation, the **FlowS** framework introduces local transport conditioning. This strategy relies on two key mechanisms:
1. **Scene-Conditioned Learned Prior**: Rather than sampling from standard Gaussian noise, an online scene encoder evaluates historical agent trajectories and HD-map polylines to output $K$ calibrated anchor trajectories per agent. These anchors are already positioned near highly plausible future paths, simplifying the generation task from global mode-discovery to local path refinement.
2. **Step-Consistent Displacement Field**: The model trains a semigroup-consistent transport field that mathematically enforces the step-consistency constraint:
$$T_{\Delta t} \circ T_{\Delta t} \approx T_{2\Delta t}$$
This constraint ensures that a single Euler integration step faithfully approximates multi-step dynamics. FlowS achieves state-of-the-art performance on the Waymo Open Motion Dataset, reaching a Soft mAP of 0.4804 and executing inference at 75 FPS using a single integration step.

### TIGFlow-GRPO: Interaction-Aware Flow Matching and Policy Optimization
While FlowS optimizes inference speed through physical anchoring, other frameworks focus on aligning generative flows with non-differentiable behavioral constraints. Standard CFM frameworks rely primarily on supervised training, which can lead to violations of social norms (e.g., collisions in dense crowds) or physical scene boundaries. **TIGFlow-GRPO** addresses this with a two-stage training framework:
1. **Stage 1 (Representation & Context)**: The model constructs a CFM-based predictor coupled with a Trajectory-Interaction-Graph (**TIG-GAT**) encoder, selecting neighbors based on the visual field of the target agent.
2. **Stage 2 (Behavioral Alignment)**: To enforce compliance with physical and social rules, the model implements **Flow-GRPO** post-training. This step reformulates the deterministic ODE rollout as stochastic ODE-to-SDE sampling. The Group Relative Policy Optimization (GRPO) framework optimizes the model using view-aware social compliance (collision avoidance) and map-aware physical feasibility rewards, steering predictions toward physically and socially plausible trajectories.

| Generative Framework | Base Architecture | Sampling & Integration Scheme | Behavioral Alignment / Guidance | Target Validation Benchmarks |
| :--- | :--- | :--- | :--- | :--- |
| **FlowS** | Symmetric scene encoder with anchor-conditioned flow generator | Single-step Euler integration with semigroup consistency | Scene-conditioned learned prior; converts mode discovery to local refinement | Waymo Open Motion Dataset (75 FPS inference) |
| **TIGFlow-GRPO** | Trajectory-Interaction-Graph (TIG-GAT) encoder | Stochastic SDE rollout | Post-training GRPO; composite rewards for social and physical constraints | ETH/UCY, Stanford Drone Dataset (SDD) |
| **TrajFlow** | CFM backbone with self-conditioning loops | Straight optimal transport path integration | Self-conditioning training using noisy predictions | Waymo Open Motion Dataset |
| **Aircraft CFM** | CFM conditioned on ADS-B histories | Probabilistic trajectory ensemble integration | Risk-aware separation boundaries; loss-of-separation modeling | OpenSky Network historical dataset |

---

## 5. Differentiable Physics, Neural ODEs, and Manifold Constraints

Neural Ordinary Differential Equations (NODEs) offer a powerful framework for modeling physical dynamics as continuous-time processes. However, unconstrained NODEs are prone to integration drift, which can cause predicted trajectories to violate fundamental domain invariants.

### Invariant Compiler for Hard Constraint Enforcement
To address this limitation, the **Invariant Compiler** treats physical invariants as first-class types, compiling them directly into the mathematical structure of the Neural ODE. This approach separates the preserved physical structure from the unconstrained learned dynamics.

For example, a simplex constraint $\sum_i x_i = 1$ is enforced by mapping the coordinates to a unit sphere $S^{n-1}$ via a square-root transformation $u_i = \sqrt{x_i}$. The dynamics are then evolved on this sphere using a skew-symmetric vector field:
$$\frac{du}{dt} = A(u)u, \quad \text{where } A(u) = \frac{1}{2}\left(f_\theta(u) - f_\theta(u)^T\right)$$
Because $A(u)$ is skew-symmetric ($A^T = -A$), the integration preserves the unit norm $\|u\|_2^2 = 1$ over time. The physical coordinates are then recovered by squaring the states ($x_i = u_i^2$).

Similarly, stoichiometric invariants of the form $M c(t) = \text{const}$ are preserved by restricting vector field updates to the null-space of the molecular matrix $M$. For systems with unknown conservation laws, the compiler learns the invariants $V(u) \in \mathbb{R}^m$ as neural networks, projecting the unconstrained vector field $\hat{f}(u)$ directly onto the tangent space of the learned manifold:
$$\frac{du}{dt} = (I - P(u))\hat{f}(u), \quad \text{where } P = \nabla V^T(\nabla V \nabla V^T)^{-1}\nabla V$$

| Invariant Type | Physical Constraint | Geometric Representation | Enforcement Mechanism | Applications |
| :--- | :--- | :--- | :--- | :--- |
| **Simplex / Norm** | $\sum_i x_i = 1$, $x_i \ge 0$ | Unit sphere $S^{n-1}$ via square-root map | Skew-symmetric dynamics: $\frac{du}{dt} = A(u)u, A^T = -A$ | Epidemiology (SIR), Population Dynamics |
| **Center of Mass** | $\sum_i m_i r_i = 0, \sum_i m_i v_i = 0$ | Null-space of mass-weighted coordinate sum | Mean subtraction: $r_i(t) - \bar{r}$, $v_i(t) - \bar{v}$ | N-body trajectories, Molecular Dynamics |
| **Stoichiometric** | $Mc(t) = \text{const}$ | Null-space of molecular matrix $M$ | Projection: $\frac{dc}{dt} = B \cdot r_\theta(c,t)$, where $MB=0$ | Chemical reaction networks, NOx kinetics |
| **Lorentz Cone** | $t \ge \|x\|_2$ | Lorentz cone manifold $\mathcal{L}^{n+1}$ | Tangent-cone projection: $\frac{dz}{dt} = \pi_{T_K}(z)(\hat{f}_\theta(z))$ | Relativistic dynamics, robust control |
| **Port-Hamiltonian** | $\frac{dH}{dt} \le 0$ | Poisson manifold + PSD dissipation | Dissipative vector field: $\frac{dz}{dt} = \nabla_z K(z), R \succeq 0$ | Damped harmonic oscillators, electrical circuits |
| **GENERIC** | $\frac{dH}{dt}=0, \frac{dS}{dt} \ge 0$ | Poisson + projection-based dissipation | Casimir-dependent entropy projection: $M z$ | Non-equilibrium thermodynamics |

### Quantum Domain and Soft Robotic Friction Constraints
In the quantum domain, **Quantum Neural Ordinary and Partial Differential Equations (QNODEs and QNPDEs)** extend the continuous-time formulation of classical NODEs to quantum control and Hamiltonian Learning (HL) by parameterizing the system's dynamics using generalized Schrodinger-type Hamiltonian equations.

For soft robotic contact dynamics, interfaces are modeled using first-order ODEs based on the LuGre friction model describing normal displacement $\delta(t)$, damping $c_n$, and bristle deflection $z(t)$:
$$\frac{dz}{dt} = v - \frac{\sigma_0 |v|}{g(v)}z, \quad \text{where } g(v) = \mu |F_n|$$
To prevent physically implausible solutions, the network enforces the Coulomb friction inequality $|F_t| \le \mu |F_n|$ as a hard constraint.

---

## 6. Synthesis of Trajectory Paradigms

* **Symmetry vs. Efficiency**: Selective State Space Models (such as DSTM and Social-Mamba) prioritize scaling efficiency, achieving linear computational complexity $O(N)$ by factorizing temporal and relational scans. In contrast, Clifford networks and equivariant models prioritize geometric inductive biases, structurally guaranteeing coordinate consistency.
* **Discrete Transition Step-by-Step vs. Continuous Vector Fields**: Discrete-time trajectory models suffer from step-wise error accumulation during long-horizon rollouts. Continuous-time formulations (such as EG-NODE and the Invariant Compiler) resolve this by parameterizing the underlying vector field directly.
* **Generative Sampling Trade-offs**: Standard diffusion models generate high-quality multimodal trajectories but are limited by slow inference speeds. Flow Matching architectures (like FlowS) resolve this latency bottleneck by initializing from scene-conditioned anchor priors, reducing the generation task to local path refinement.

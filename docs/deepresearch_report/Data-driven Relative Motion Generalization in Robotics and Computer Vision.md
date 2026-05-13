# Data-driven Relative Motion Generalization in Robotics and Computer Vision

> [!TIP]
> **Executive Summary: The Paradigm Shift to Physical AI**
> - **Core Challenge**: Transitioning from rigid, task-specific automation to adaptive, general-purpose Physical AI capable of generalizing across unstructured environments.
> - **Key Innovation**: **Relative Motion Generalization**—learning movements defined by dynamic relationships rather than global coordinate frames.
> - **Technical Enablers**: Coordinate-invariant representations (DUTIR), SE(3)-equivariant architectures, and generative diffusion-based motion planning.
> - **Outcome**: Significant reductions in training data requirements (e.g., 5-10 demonstrations) and improved robustness to environmental perturbations.

The field of robotics is currently undergoing a paradigm shift, transitioning from rigid, task-specific automation to adaptive, general-purpose Physical Artificial Intelligence (AI). This transformation is driven by the realization that traditional analytical models, while mathematically robust, often fail to generalize across the vast, unstructured environments of the real world. At the heart of this evolution is the concept of relative motion generalization: the ability of a robotic system to learn, represent, and execute movements that are not anchored to a single global coordinate frame but are instead defined by the dynamic relationships between the robot, its environment, and the objects it manipulates.[1, 2] Unlike classical motion planning, which relies on precise geometric maps and predefined kinematic chains, data-driven approaches leverage large-scale datasets, multimodal foundation models, and geometric inductive biases to achieve robust performance in open-world scenarios.[3, 4]

---

## 1. Mathematical Foundations and Reference Frames

The conceptual foundation of relative motion begins with the definition of reference frames. In classical mechanics, a reference frame is a coordinate system for which the velocity is specified, often set to zero relative to a known object, such as the ground or a moving vehicle.[5] The principle of **Galilean invariance** asserts that the laws of physics remain consistent across all inertial frames, a concept that is foundational for robotic systems attempting to operate in moving environments—such as a drone landing on a ship or a mobile manipulator working within a crowded factory.[5, 6]

In complex multi-body systems, such as spacecraft formation flying or proximity operations near periodic orbits, relative motion becomes significantly more complex due to the chaotic nature of underlying gravitational solutions.[7] Researchers have introduced **local toroidal coordinate systems** to characterize relative motion near periodic orbits, derived from first-order approximations of invariant tori. These coordinate systems provide a geometric interpretation that is consistent across distinct orbits, allowing for the rapid generation of quasi-periodic relative motion approximations without the computational burden of high-fidelity ephemeris models.[7] This geometric insight is critical for developing proximity operation guidelines and formation flying control schemes that are robust to nonlinear perturbations.

For terrestrial robotics, the challenge lies in the **"action grounding gap."** Vision systems provide high-dimensional data regarding where objects are in the world, but proprioceptive signals—internal state awareness of joint positions, velocities, and torques—are essential for precise manipulation.[1] Relative motion generalization requires fusing these disparate signals into a unified representation. The ability of an agent to perceive its own arm position relative to its body, and subsequently its body relative to a target object, is what enables the transition from simple image-based reasoning to embodied physical action.[1, 8]

---

## 2. Coordinate-Invariant Representations

To achieve consistency when a trajectory is expressed in different coordinate systems, it is necessary to utilize models that are inherently coordinate-invariant. Classical representations often suffer from sensitivity to measurement noise or singularities, where the representation becomes uniquely undefined.[9, 10] Recent advancements have introduced the **Dual-Upper-Triangular Invariant Representation (DUTIR)**, which utilizes linear algebra and differential geometry to transform demonstrated trajectories into a representation that remains unchanged despite changes in the coordinate frame.[9]

> [!IMPORTANT]
> **Mathematical Core: The Screw**
> The mathematical entity at the core of this invariance is the **screw**, which abstracts both motion (twists) and forces (wrenches) into a six-dimensional vector $\xi=(\alpha^\top, \beta^\top)^\top$.[9]
> - **Twist**: First-order kinematics where $\alpha$ denotes rotational velocity and $\beta$ denotes translational velocity.
> - **Wrench**: Force systems where $\alpha$ is the resultant force and $\beta$ is the resultant torque.
> - **Screw Axis**: Defined by direction $e=\alpha/\|\alpha\|$ and position $p_\perp = (\alpha \times \beta)/\|\alpha\|^2$.

The DUTIR approach employs an **"SU-decomposition,"** where the local trajectory matrix is decomposed into a screw-transformation matrix $S$ and an invariant representation $U$, such that $\Xi=SU$.[9] This decomposition effectively "cancels out" the choice of coordinate system, as any change in the reference frame is absorbed by $S$, leaving $U$—composed of two 3×3 upper-triangular matrices—constant across transformations.[9] This mathematical rigor allows for the identification and generalization of trajectories in robotics and biomechanics, even in the presence of unexpected variations in the observer's viewpoint or the user's initial pose.[10, 11]

### Comparison of Motion Representations

| Representation Type | Mathematical Basis | Invariance Properties | Application Focus |
| :--- | :--- | :--- | :--- |
| **Cartesian** | Vectors (x,y,z) | None (Origin-dependent) | Basic position sensing |
| **Screw Theory** | Twists and Wrenches | Relative frame aware | Rigid-body kinematics |
| **DUTIR** | SU-Decomposition | Full Coordinate Invariance | Trajectory recognition |
| **Dual Quaternions** | Clifford Algebra | Translation & Rotation | Compact motion encoding |
| **Toroidal Sets** | Hamiltonian Dynamics | Invariant Tori | Multi-body space systems |

---

## 3. Geometric Deep Learning and Symmetry Inductive Biases

The efficacy of data-driven models is often limited by their sample efficiency and ability to generalize to novel poses. Conventional neural networks require massive amounts of data augmentation—rotating and translating input data manually—to learn spatial relationships. In contrast, **geometric deep learning** explicitly integrates physical symmetries into the network architecture.[12, 13]

### SE(3)-Equivariant Architectures
**Equivariance** is a formal property where a transformation of the input leads to a predictable, consistent transformation of the output. A mapping $f$ is **SE(3)-equivariant** if for any group element $g$ (representing 3D rotation and translation) and data $x$, the condition $f(g \cdot x) = g \cdot f(x)$ holds.[12, 14] This provides a strong inductive bias for tasks involving 3D geometric data, such as molecular modeling, point cloud processing, and robotic manipulation.[12]

The **SE(3)-Transformer** is a landmark architecture in this domain, maintaining precise equivariance through structured self-attention and message passing.[12] By using spherical harmonics and Clebsch–Gordan decompositions, it constructs kernels that ensure robust geometric feature extraction. In robotics, this allows for the processing of scene point clouds such that the resulting features remain coherent with the rigid transformations of the input.[12, 15] Empirical evidence suggests that SE(3)-equivariant models generalize far better to novel object poses and instances than their non-equivariant counterparts, particularly in low-data regimes.[12, 14]

### The RiEMann Framework
One of the most effective implementations of these principles is **RiEMann**, a near real-time imitation learning framework for 6-DOF robot manipulation from scene point clouds.[14] RiEMann achieves SE(3)-equivariance without requiring object segmentation or time-consuming field-matching processes. It addresses the computational complexity of equivariant backbones through a two-stage process:
1.  **Saliency Mapping**: Learns an SE(3)-invariant saliency map to extract a region of interest (ROI) from the input point cloud.
2.  **Targeted Policy**: Applies the SE(3)-equivariant policy only to the extracted subset.[14]

The action space in RiEMann is carefully parameterized to maintain symmetry:
- **Translational Actions**: Utilizes an SE(3)-invariant vector field as a target point affordance map.
- **Rotational Actions**: Employs three SE(3)-equivariant vector fields as bases for a target rotation matrix, followed by a Gram-Schmidt orthogonalization to ensure a valid SO(3) rotation.[14]

Experimental results indicate that RiEMann can be trained from scratch with as few as **5 to 10 demonstrations**, reducing SE(3) geodesic distance errors by **68.6%** compared to non-equivariant baselines.[14]

---

## 4. Generative Paradigms in Motion Planning

The complexity of motion planning increases exponentially with the number of robot joints and the density of environmental obstacles. Traditional sampling-based motion planners (SBMPs), while probabilistically complete, struggle in "narrow passage" scenarios and often produce non-smooth trajectories.[3, 16] Data-driven approaches are increasingly being used to accelerate these processes by learning sample distributions conditioned on environmental information.[3]

### Diffusion Models for Trajectory Synthesis
Diffusion models have emerged as a powerful generative tool for learning complex, multi-modal data distributions. In robotics, these models are applied to both imitation learning and direct policy optimization, treating motion planning as an inference problem.[17, 18] By learning a trajectory distribution prior, diffusion models can generate high-quality, dynamically feasible paths that are robust to high-dimensional input and output spaces.[16, 18]

The **Context-Aware Motion Planning Diffusion (CAMPD)** framework integrates sensor-agnostic contextual information, such as obstacle locations, directly into a denoising probabilistic model.[17] This allows the planner to adapt to unseen environments without the need for retraining. Furthermore, by utilizing cost function gradients during the denoising process (**cost guidance**), these models can ensure collision-free paths even in dynamic settings.[16, 17] For multi-robot systems, the **Multi-robot Multi-model planning Diffusion (MMD)** algorithm scales this approach by combining single-robot diffusion models with classical search-based techniques, effectively breaking the "curse of dimensionality" associated with joint multi-agent state spaces.[19]

### Comparison of Planning Paradigms

| Planning Paradigm | Mechanism | Scaling Behavior | Multi-Modality |
| :--- | :--- | :--- | :--- |
| **SBMP (RRT/PRM)** | Random Sampling | PSPACE-hard | Limited (Single path) |
| **Optimization-based** | Cost Function Min | Gradient-dependent | Low (Local minima) |
| **Diffusion (CAMPD)** | Denoising Inference | High-dimensional | High (Learned modes) |
| **Hybrid (MPD)** | Prior + Guidance | Task-dependent | High (Guided samples) |

---

## 5. Transporter Networks and Visual Rearrangement

Another influential data-driven approach is the **Transporter Network**, which formulates manipulation as a sequence of spatial displacements.[20, 21] Instead of directly regressing to joint angles, the Transporter Network rearranges deep features to infer where a chunk of 3D space (an object or part of an object) should be moved. This is achieved by attending to a local "pick" region and then searching for the target "place" displacement via deep feature template matching.[21, 22]

By operating on a spatially-consistent 3D reconstruction, Transporter Networks naturally exploit equivariance for inductive biases. They are capable of learning pick-and-place, pushing, and even deformable object manipulation from very few demonstrations.[21, 23] This formulation allows them to generalize to unseen object configurations and orientations far more effectively than traditional end-to-end models.[22] However, they remain sensitive to camera-robot calibration and noisy point cloud data, highlighting the need for robust perceptual backbones.[23]

---

## 6. Spatial Awareness and Contextual Relationship Learning

Generalization in cluttered environments requires more than just coordinate invariance; it requires an understanding of spatial dependencies between objects. An improper manipulation order in a dense bin can lead to collisions or blocked access to target items.[24, 25]

### Spatial Graph Neural Networks
Graph Neural Networks (GNNs) have become the standard for modeling these complex inter-object relationships. Frameworks like **OrderMind** construct spatial graphs where object center points are vertices and edges encode physical proximity and alignment.[24, 25] By using k-Nearest Neighbors to aggregate geometric information, the model learns object manipulation priorities based on the local layout.[24] This allows for real-time inference of manipulation orders that are physically and semantically plausible.

Similarly, in multi-robot social navigation, **Dynamic Graph Neural Networks (DGNNs)** model robots and pedestrians as nodes in a time-varying graph.[26] By using message passing and temporal modules like Gated Recurrent Units (GRUs), the DGNN can predict interaction patterns and generate navigation commands that reduce robot-human conflict rates by **30%** compared to traditional baselines.[26]

### Language-Conditioned Spatial Grounding
The rise of Large Language Models (LLMs) and Vision-Language Models (VLMs) has enabled a new level of task-level generalization. Systems like **Language Instruction grounding for Motion Planning (LIMP)** leverage foundation models and temporal logic to generate instruction-conditioned semantic maps.[27] This allows robots to follow complex, long-horizon instructions with open-vocabulary referents (e.g., "place the toy in front of the whiteboard").

The LIMP architecture translates natural language into temporal logic specifications, which are then grounded in a 3D environment representation using VLMs.[27] This ensures that the robot's behaviors are "correct-by-construction" and logically aligned with the user's intent. The ability to resolve ambiguous spatial descriptors is a key component of relative motion generalization in human-robot interaction.[27, 28]

---

## 7. Data-Centric AI and Foundation Models for Robotics

The performance of modern Vision-Language-Action (VLA) models is inextricably linked to the quality and diversity of the data used for training.[1] Unlike the internet-scale text datasets that power LLMs, robotics data is fundamentally constrained by the "action grounding gap" and the scarcity of real-world trajectories.[1, 29]

### Open X-Embodiment and RLDS
To address data scarcity, the **Open X-Embodiment** initiative has unified over **1 million real robot trajectories** from 22 different embodiments across 60 datasets.[1] Models trained on this diverse multi-embodiment data significantly outperform those trained on single-robot datasets, even when tested on the original robot. This proves that diversity provides complementary information that improves generalization.[1]

The storage and consumption of this data require standardized formats like **RLDS (Reinforcement Learning Datasets)**, which preserve the temporal structure of robotic episodes.[1] In robotics, data points cannot be treated as independent and identically distributed (i.e., they cannot be shuffled randomly) because actions at time $T$ depend on states at $T-1$.[1] Maintaining this causality is essential for models to learn state transitions and long-range dependencies.

### Video as a Planning Modality
A recent breakthrough in achieving zero-shot generalization is the use of video as the primary modality for foundation models. The **Large Video Planner (LVP)** treats video as a visual planner, leveraging the rich physical dynamics and semantics encoded in massive human-centric datasets like Ego4D.[29] Given a single image and a task instruction, the LVP generates a video depicting how the task should be completed. This generated motion is then retargeted to a robot hand for execution.[29]

This paradigm shift from "predicting actions" to **"predicting world evolution"** allows robots to internalize a "physics engine" that understands how objects should move.[29, 30] By pre-training on diverse video data, robots can synthesize fragments of knowledge to solve tasks they have never explicitly seen before, such as operating an air fryer or cleaning an unfamiliar room.[30]

---

## 8. Dexterity and In-Hand Manipulation

While global motion planning focuses on reaching a target, dexterous manipulation focuses on the "in-hand" reorientation of objects. This is particularly challenging when the object's shape is unknown.[31]

### Shape Priors and Tactile Feedback
Researchers have developed systems like **FreeTacMan**, a robot-free, human-centric data collection system that captures synchronized visual and tactile data.[32] By using a modular finger-gripper interface, human demonstrators can provide high-resolution feedback on grip force and object orientation. This data is critical for learning tasks like handling fragile cups or identifying textures, where vision alone is insufficient.[32]

For unknown objects, the **Dexterous Manipulation Graph (DMG)** method uses deep generative models to infer full object shapes from partial visual sensing.[31] By accounting for estimation uncertainty, the system can plan a sequence of manipulation actions to achieve a desired grasp without a pre-existing object model. This emphasizes the role of "imagining" invisible parts of an object by exploiting knowledge about object classes (e.g., hammers vs. bottles).[31]

### Non-Prehensile Sliding Manipulation
Generalization also extends to non-prehensile tasks, such as sliding an object on a surface.[33] This requires controlling the acceleration of a robotic arm to manipulate an object's relative displacement via frictional forces. Reinforcement learning frameworks like **DDPG** have been employed to estimate these frictional parameters dynamically, enabling precise positioning of objects that cannot be easily grasped.[33, 34] This highlights the necessity of **"adjustable deformable object manipulation,"** where dual arms are decoupled into "leader" and "follower" roles to coordinate complex tasks like stir-frying.[34]

---

## 9. Benchmarking Generalization and Repeatability

The ability to measure generalization is as important as the ability to achieve it. Traditional benchmarks often lack the visual realism or environmental diversity needed to evaluate models for the real world.[35, 36]

### The Colosseum and ManiSkill2
The **Colosseum** benchmark systematically evaluates manipulation models across 14 dimensions of environmental perturbations, including changes in object color, texture, lighting, and physical properties.[37, 38] Results show that the success rates of state-of-the-art models often degrade by **30-50%** when exposed to even single perturbation factors. When multiple perturbations are applied, performance can drop by over **75%**, indicating a critical "fragility" in current imitation learning models.[37, 38]

**ManiSkill2** addresses these pain points by providing over 2000 object models and 4 million demonstration frames.[36] It supports a unified interface for reinforcement learning, imitation learning, and classic sense-plan-act algorithms. Importantly, it covers rigid, soft-body, and articulated object manipulation, providing the scale necessary to test general-purpose skills.[36]

### Robotic Generalization Benchmarks

| Benchmark | Tasks | Variation Factors | Data Scale |
| :--- | :--- | :--- | :--- |
| **RLBench** | 100+ tasks | Geometry/Viewpoint | High (Demonstration-based) |
| **ManiSkill2** | 20 families | 2000+ objects | 4M+ frames |
| **The Colosseum** | 20 tasks | 14 perturbations | Perturbation-focused |
| **VISER** | Realistic | Visual Realism | High-fidelity simulation |
| **VLABench** | LCM tasks | Mesh/Texture/Semantic | 163k+ samples |

### Industrial Repeatability and DDMC
In industrial settings, repeatability is the primary concern. **Data-Driven Modeling and Control (DDMC)** aims to improve autonomy by using real-time sensor data rather than fully specified analytical models.[39] For redundant robots, such as the 8-DOF KUKA YouBot, increasing degrees of freedom also increases the complexity of control and potential for cumulative errors.[39] DDMC strategies facilitate effective task execution by online estimation of the Jacobian matrix, minimizing hardware requirements like multiple encoders and reducing the risk of failure from measurement discrepancies.[39]

---

## 10. Trends and Future Directions in 2026

As we approach mid-2026, several key trends are defining the future of robotic relative motion generalization. Humanoid robots are transitioning from research pilots to commercial platforms, with production allocations being committed to major manufacturing and logistics firms.[30, 40]

### Foundations for Generalist Agents
The architectural breakthrough enabling this transition is the **Vision-Language-Action (VLA)** foundation model.[4, 30] Rather than hardcoding routines, VLAs combine perception, reasoning, and motor commands into a single framework. NVIDIA's **GR00T** and Physical Intelligence's **pi-series** models exemplify this, demonstrating the ability to perform tasks in entirely unfamiliar environments.[30]

Key findings from the **"EgoScale"** research confirm that robotics foundation models follow scaling laws similar to LLMs: policy performance improves predictably with the size of pre-training data.[30] This has sparked a race for high-fidelity data collection, utilizing tactile AI, human video data, and synthetic data from hyper-realistic simulations like NVIDIA Omniverse.[40]

### Closing the Sim-to-Real Gap
Technological advancements in **"sim-to-real"** and **"real-to-sim"** transfer are cutting engineering times for new products.[40] Frameworks that combine machine learning with high-end physics simulations allow robots to learn tasks faster than real-time.[41] If the simulation realistically captures forces and elasticity, it provides a high-quality data source that minimizes human intervention and hardware wear-and-tear.[35, 41]

---

## 11. Summary of Core Mechanisms and Implications

The synthesis of the research suggests that relative motion generalization is not a single problem but a convergence of mathematical, architectural, and data-centric solutions.

1.  **Mathematical Invariance**: The move from Cartesian coordinates to screw-based and coordinate-invariant representations like **DUTIR** provides the geometric stability needed for trajectory recognition across different frames.[9, 10]
2.  **Structural Inductive Biases**: Architectures like **SE(3)-Transformers** and **RiEMann** prove that baking physical symmetries into neural networks is the most efficient way to learn 3D actions from minimal data.[12, 14]
3.  **Generative Planning**: **Diffusion models** are replacing traditional planners by learning the multi-modal distributions of valid motion, enabling robots to adapt to dynamic clutter without retraining.[16, 17]
4.  **Temporal and Proprioceptive Fusion**: **VLA models** are successfully bridging the gap between high-level reasoning and low-level motor control, provided the temporal structure of robotic data is preserved.[1, 30]
5.  **Scaling and Benchmarking**: The industry is embracing **scaling laws**, but benchmarks like **The Colosseum** remind researchers that robustness to environmental perturbations is still a significant hurdle to real-world deployment.[30, 37]

The future of robotics lies in the refinement of these general-purpose models. As hardware costs decrease—exemplified by the **$15,400 Unitree humanoid platforms**—the "brain" of the robot becomes the primary differentiator.[4, 30] The ability of that brain to generalize relative motion will determine whether robots remain confined to assembly lines or truly integrate into the fabric of daily life.

---

## References

1. **VLA Models: Why Data-Centric AI Unlocks Next-Gen Robotics** - Voxel51, [Link](https://voxel51.com/blog/vla-models-data-centric-ai-robotics)
2. **Foundation Models in Robotics: A Comprehensive Review** - arXiv, [Link](https://arxiv.org/html/2604.15395v1)
3. **Data-Driven Motion Planning: A Survey** - IEEE Xplore, [Link](https://ieeexplore.ieee.org/iel8/6287639/10820123/10930422.pdf)
4. **The real breakthrough in robotics is foundation models — not hardware** - The New Stack, [Link](https://thenewstack.io/physical-ai-models-frontier/)
5. **Relative Motion** - Front Matter, [Link](https://lipa.physics.oregonstate.edu/sec_relative-motion.html)
6. **Dynamic Aerial Coverage of Stationary and Moving Structures** - AIAA, [Link](https://arc.aiaa.org/doi/10.2514/1.G008236)
7. **Describing relative motion near periodic orbits** - University of Colorado Boulder, [Link](https://www.colorado.edu/faculty/bosanac/sites/default/files/attached-files/2022_ellbos_cmda_accepted.pdf)
8. **Dynamic thumb localization and its adaptation** - PMC, [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12791052/)
9. **A Coordinate-Invariant Local Representation of Motion and Force** - arXiv, [Link](https://arxiv.org/pdf/2604.10241)
10. **[2604.10241] A Coordinate-Invariant Local Representation of Motion** - arXiv, [Link](https://arxiv.org/abs/2604.10241)
11. **A Coordinate-Invariant Local Representation of Motion and Force Trajectories** - ResearchGate, [Link](https://www.researchgate.net/publication/403790538_A_Coordinate-Invariant_Local_Representation_of_Motion_and_Force_Trajectories_for_Identification_and_Generalization_Across_Coordinate_Systems)
12. **SE(3)-Transformer Overview - Equivariance** - Emergent Mind, [Link](https://www.emergentmind.com/topics/se-3-transformer)
13. **[2503.09829] SE(3)-Equivariant Robot Learning and Control** - arXiv, [Link](https://arxiv.org/abs/2503.09829)
14. **RiEMann: Near Real-Time SE(3)-Equivariant Robot** - GitHub, [Link](https://raw.githubusercontent.com/mlresearch/v270/main/assets/gao25a/gao25a.pdf)
15. **Multi-body SE(3) Equivariance for Unsupervised Rigid Segmentation** - NeurIPS, [Link](https://neurips.cc/virtual/2023/poster/72579)
16. **Learning and Adapting Robot Motion Planning with Diffusion Models** - arXiv, [Link](https://arxiv.org/html/2412.19948v2)
17. **Accelerated Multi-Modal Motion Planning Using Context-Conditioned Diffusion** - arXiv, [Link](https://arxiv.org/html/2510.14615v2)
18. **Diffusion Models for Robotic Manipulation: A Survey** - arXiv, [Link](https://arxiv.org/html/2504.08438v3)
19. **Multi-Robot Motion Planning with Diffusion Models** - arXiv, [Link](https://arxiv.org/html/2410.03072v1)
20. **Transporter Networks: Rearranging the Visual World** - Semanticscholar, [Link](https://www.semanticscholar.org/paper/Transporter-Networks%3A-Rearranging-the-Visual-World-Zeng-Florence/eb6b9bc4ff3e4e2cf1724324d79ce7de43131478)
21. **[2010.14406] Transporter Networks: Rearranging the Visual World** - ar5iv, [Link](https://ar5iv.labs.arxiv.org/html/2010.14406)
22. **Rearranging the Visual World** - Google Research, [Link](https://research.google/blog/rearranging-the-visual-world/)
23. **Transporter Networks: Visual World Rearrangement** - YouTube, [Link](https://www.youtube.com/watch?v=496UVuAdOP4)
24. **NeurIPS Poster Learning Spatial-Aware Manipulation Ordering** - NeurIPS, [Link](https://neurips.cc/virtual/2025/poster/116565)
25. **Learning Spatial-Aware Manipulation Ordering** - arXiv, [Link](https://arxiv.org/html/2510.25138v1)
26. **Dynamic Graph Neural Networks for Socially Aware Multi-Robot Navigation** - OpenReview, [Link](https://openreview.net/forum?id=h1smCvJcxI)
27. **Verifiably Following Complex Robot Instructions with Foundation Models** - arXiv, [Link](https://arxiv.org/html/2402.11498v1)
28. **Spatial relation learning in complementary scenarios** - Frontiers, [Link](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.844753/full)
29. **Large Video Planner: A New Foundation Model for General-Purpose Robots** - Harvard, [Link](https://kempnerinstitute.harvard.edu/research/deeper-learning/large-video-planner-a-new-foundation-model-for-general-purpose-robots/)
30. **Humanoid Robotics In 2026: The Race From Pilot To Platform** - KraneShares, [Link](https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/)
31. **In-Hand Manipulation of Objects with Unknown Shapes** - Diva-Portal, [Link](https://www.diva-portal.org/smash/get/diva2:1366113/FULLTEXT01.pdf)
32. **FreeTacMan: Robot-free Visuo-Tactile Data Collection System** - arXiv, [Link](https://arxiv.org/html/2506.01941v4)
33. **(PDF) A Reinforcement Learning Approach to Non-prehensile Manipulation** - ResearchGate, [Link](https://www.researchgate.net/publication/389316438_A_Reinforcement_Learning_Approach_to_Non-prehensile_Manipulation_through_Sliding)
34. **Robot Cooking with Stir-fry: Bimanual Non-prehensile Manipulation** - Idiap, [Link](https://www.idiap.ch/~scalinon/papers/Liu-RAL2022.pdf)
35. **Toward Visually Realistic Simulation: A Benchmark for Evaluating Robot Manipulation** - arXiv, [Link](https://arxiv.org/html/2605.06311v1)
36. **MANISKILL2: A UNIFIED BENCHMARK FOR GENERALIZABLE MANIPULATION SKILLS** - OpenReview, [Link](https://openreview.net/pdf?id=b_CQDy9vrD1)
37. **THE COLOSSEUM: A Benchmark for Evaluating Generalization** - Roboticsproceedings, [Link](https://www.roboticsproceedings.org/rss20/p133.pdf)
38. **The Colosseum: A Benchmark for Evaluating Generalization** - arXiv, [Link](https://arxiv.org/html/2402.08191v1)
39. **Data-Driven Model for Cyclic Tasks of Robotic Systems** - MDPI, [Link](https://www.mdpi.com/2227-9717/13/4/953)
40. **3 robotics trends from NVIDIA GTC 2026** - The Robot Report, [Link](https://www.therobotreport.com/3-robotics-trends-from-nvidia-gtc-2026/)
41. **Simulation-driven machine learning for robotics and automation** - Fraunhofer, [Link](https://publica.fraunhofer.de/bitstreams/e8b4e683-6852-411c-b6c2-551917176706/download)
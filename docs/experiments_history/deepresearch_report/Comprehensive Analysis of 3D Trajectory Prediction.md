# Comprehensive Analysis of 3D Trajectory Prediction: Deep Learning Frameworks, Spatiotemporal Reasoning, and Multi-Modal Scene Understanding

The anticipation of dynamic agent behavior represents the most critical hurdle in the realization of fully autonomous systems, encompassing the domains of intelligent transportation, service robotics, and aerial surveillance.[1, 2] As autonomous vehicles and robots transition from controlled environments to complex, high-density urban settings, the requirement for precision has moved beyond two-dimensional planar estimation toward high-fidelity three-dimensional (3D) trajectory prediction.[3, 4] Modern trajectory prediction is defined as the task of forecasting the future positions and states of surrounding agents—including vehicles, pedestrians, and cyclists—based on their historical movement patterns and the multifaceted semantic context of their environment.[1, 5] This functionality is an essential component of the perception-planning-control pipeline, directly influencing the vehicle's driving behavior and safety protocols.[2, 6]

## Spatiotemporal Foundations and Problem Formalization

Trajectory prediction is fundamentally structured as a sequence-to-sequence learning problem where the objective is to map a sequence of past observations to a distribution of future paths.[1, 7] For a given target agent $i$, the observed historical state $X_i$ typically spans a window of $T_{obs}$ timesteps, frequently set to 2 seconds at a sampling rate of 10 Hz.[8, 9] These states comprise spatial coordinates $(x,y,z)$, velocity, acceleration, and orientation (heading) $\theta$.[3, 5, 10] In the context of 3D trajectory prediction, the inclusion of the vertical $z$-axis is paramount for navigating uneven terrains, multi-floor structures, or aerial environments where altitude variations significantly impact collision risk.[4, 11, 12]

The environment is characterized by an auxiliary input $C$, which represents the scene context.[5, 13] This context includes high-definition (HD) maps containing lane centerlines, boundaries, crosswalks, stop lines, and traffic rules.[1, 14, 15] Furthermore, the social context—the states and intentions of all neighboring dynamic agents—is integrated to capture the inter-dependency of decisions.[1, 6, 16] The ultimate goal is to predict a future state sequence $Y_i$ over a prediction horizon $T_{pred}$, which standard benchmarks like Waymo and nuScenes define as 3 to 8 seconds.[8, 17]

Due to the inherent stochasticity of human behavior, a single observed history does not correspond to a unique future.[1, 18] Human drivers and pedestrians possess latent intentions that are not explicitly observable, such as the decision to turn, change lanes, or yield.[7, 19] Consequently, modern deep learning architectures are designed for multi-future trajectory prediction (MTP), generating a set of $K$ discrete, plausible trajectory hypotheses $\{Y_i^k\}_{k=1}^K$, each with an associated probability score $P_k$.[1, 8, 20]

> [!TIP]
> **Summary Table: Spatiotemporal Features**
>
> | Feature | Description and Formalization |
> | :--- | :--- |
> | Input Sequence ($X$) | $\{x_{t-T_{obs}}, \dots, x_t\}$ where $x \in \mathbb{R}^3$.[1, 5] |
> | Output Modes ($K$) | Multiple plausible future paths, typically $K=6$ or $K=25$.[8, 20] |
> | Semantic Context ($C$) | Vectorized lane graphs, traffic signs, and terrain types.[1, 15] |
> | Social Interaction | Graph-based or Attention-based modeling of neighboring agent states.[1, 5, 16] |
> | Evaluation Metrics | Distance-based (ADE, FDE) and safety-based (Miss Rate, Collision Score).[8, 21] |

## Evolution of Modeling Paradigms: From Physics to Deep Learning

The trajectory prediction domain has evolved through three distinct phases: physics-based modeling, traditional machine learning, and modern deep learning.[7, 22] Traditional physics-based methods, such as the Social Force model, utilized attractive and repulsive forces to simulate behaviors like collision avoidance and group clustering.[1] While mathematically elegant and interpretable, these models struggle with the complex, non-linear interactions found in urban traffic and lack the capacity to learn from large-scale data.[1, 23]

Machine learning approaches introduced probabilistic frameworks such as Gaussian Processes (GP), Hidden Markov Models (HMM), and Dynamic Bayesian Networks.[3, 7] Gaussian Processes are popular for pedestrian prediction due to their ability to provide uncertainty estimates; however, their computational complexity scales cubically with the number of training examples, limiting their applicability to real-time, large-scale autonomous systems.[3]

The current state-of-the-art is dominated by deep learning architectures, which excel at extracting high-level abstractions from raw spatiotemporal data.[7, 24] These models are broadly categorized by their core components:
- **Recurrent Architectures**: RNNs and LSTMs were the first to demonstrate superior performance in temporal modeling, utilizing gate mechanisms to filter information and manage the vanishing gradient problem in long sequences.[3, 7, 25]
- **Convolutional Architectures**: CNNs are often employed to extract features from rasterized bird's-eye-view (BEV) images of the environment, though they often require large receptive fields to capture long-range dependencies.[7, 15, 26]
- **Graph-based Architectures**: GNNs treat the traffic scene as a graph, where agents and map elements are nodes, and interactions are represented by edges. This enables the modeling of complex, unstructured relationships in vectorized space.[7, 13, 15]
- **Attention and Transformers**: Transformers have recently emerged as the leading paradigm, utilizing self-attention to capture global dependencies across both spatial and temporal dimensions without the sequential constraints of RNNs.[7, 27, 28]

## Vectorized Scene Representation and Map Encoding

A pivotal shift in the field occurred with the move from rasterized (image-based) map representations to vectorized representations.[13, 26] Rasterization, while compatible with mature CNN architectures, is computationally inefficient and can erase critical information due to pixelation and occlusion.[15] Vectorized approaches represent map elements (lanes, crosswalks) and trajectories as sets of polylines, preserving high-resolution geometric and topological features.[15, 16]

### VectorNet and LaneGCN

VectorNet was one of the first models to successfully incorporate vectorized HD maps and agent dynamics.[13, 15] It uses a hierarchical graph network where individual polylines are first encoded locally to extract sub-graph features, which are then aggregated globally via a fully-connected graph network.[13, 15, 29] This hierarchical approach allows the model to learn the spatial relationships between diverse entities, such as a vehicle's proximity to a lane boundary or a stop sign.[13, 15]

LaneGCN advanced this concept by constructing a dedicated lane graph from the HD map.[13, 15] It utilizes a dilated variant of graph convolution to aggregate context along the lanes, capturing long-range dependencies and complex topology.[13, 15] LaneGCN effectively models the "game-like" interactions at intersections, where the future path of one vehicle is heavily constrained by the presence of another and the underlying road structure.[6, 15]

## Target-Driven and Goal-Conditioned Prediction

The TNT (Target-driveN Trajectory) and DenseTNT models introduced a goal-conditioned paradigm to improve the scene-compliance of multi-modal predictions.[15, 26] TNT identifies that the endpoint (goal) carries the most uncertainty in a trajectory.[26] It first samples a set of "anchors" from lane centerlines and predicts the probability of the agent reaching each anchor.[15, 26] After selecting the most likely goals, it completes the full trajectory for each. DenseTNT further refined this by becoming anchor-free, generating dense goal candidates directly from the scene context and employing a goal set predictor to produce final trajectories, thereby eliminating the need for post-processing like non-maximum suppression (NMS).[26]

> [!TIP]
> **Comparison of Vectorized/Goal-Driven Models**
>
> | Architecture | Map Encoding Method | Interaction Mechanism | Core Innovation |
> | :--- | :--- | :--- | :--- |
> | VectorNet | Vectorized polylines.[13] | Hierarchical GNN.[15] | First to directly use HD vector data.[15] |
> | LaneGCN | Lane graph.[15] | Dilated graph convolution.[15] | Captures complex road topology.[15] |
> | TNT | Anchors on centerlines.[26] | Goal-conditioned.[15] | Intentions as discrete spatial targets.[15] |
> | DenseTNT | Dense goal sets.[26] | Anchor-free.[26] | End-to-end goal set prediction.[26] |

## Transformer-Based Frameworks and Hierarchical Attention

The adoption of the Transformer architecture has significantly enhanced the ability of models to process multi-agent scenes.[27, 28] Transformers rely on attention mechanisms to dynamically weight the importance of different agents and map elements, regardless of their spatial or temporal distance.[13, 27]

### Hierarchical Vector Transformer (HiVT)

HiVT is a representative baseline that addresses the trade-off between prediction accuracy and inference efficiency.[16, 30] It decomposes the multi-agent prediction problem into two stages: local context extraction and global interaction modeling.[16, 31] The local encoder extracts rotation-invariant spatiotemporal features for each agent within its local neighborhood, while the global interactor models interactions across the entire scene.[16, 30] This hierarchical decomposition allows HiVT to scale efficiently to scenes with a large number of agents while maintaining state-of-the-art accuracy on benchmarks like Argoverse.[16, 31] Furthermore, HiVT's use of pairwise relative pose encoding ensures that its representations are robust to geometric transformations, such as rotation and translation, which is critical for generalizing across different urban layouts.[16, 31]

### Motion Transformer (MTR) and MTR++

The Motion Transformer (MTR) framework formulates trajectory prediction as a motion query refinement task.[22, 28] It scores first on the Waymo Motion Prediction Challenge by combining global intention localization with local movement refinement.[28, 32] MTR uses dynamic intention queries to aggregate scene context iteratively during decoding.[33] MTR++ further improves upon this by incorporating symmetric scene modeling and mutually-guided intention querying modules.[32] These enhancements facilitate more complex future behavior interactions among multiple agents, leading to trajectories that are not only accurate in terms of displacement but also compliant with the global scene structure.[32]

### Wayformer and Latent Query Encoding

Wayformer introduces a compute-efficient latent query encoding mechanism to process massive amounts of scene context.[31, 34] It uses a transformer-based early-fusion encoder to integrate diverse input modalities—including agent history and map data—before passing them to a decoder.[34] The use of latent queries allows Wayformer to maintain a high level of representational power while significantly reducing the number of parameters and computational cost, making it more suitable for on-board deployment in autonomous vehicles.[31, 34]

## Generative Models and Multi-Modal Uncertainty

The fundamental challenge of trajectory prediction is modeling the non-deterministic nature of future movement.[1, 35] Generative models provide a framework for learning the underlying distribution of trajectories, allowing for the sampling of diverse, socially acceptable futures.[36, 37]

### VAEs and GANs

Variational Autoencoders (VAEs) map historical trajectories into a latent probabilistic space, which is then sampled to generate future paths.[36, 37] While VAEs are stable to train and provide controlled sampling, they often produce over-smoothed or "blurred" trajectories because the latent regularization tends to penalize outlier modes.[38, 39] Generative Adversarial Networks (GANs), such as Social-GAN or Goal-GAN, employ a minimax game between a generator and a discriminator.[38, 40] GANs can produce high-quality, sharp trajectories that better match the ground truth's statistical properties.[38, 40] However, GANs are notoriously difficult to train and frequently suffer from "mode collapse," where the generator only learns to output a single frequent maneuver (e.g., going straight) regardless of the context.[38, 39, 40]

### The Emergence of Diffusion Models

Diffusion models have recently surpassed GANs and VAEs in their ability to model multi-modal distributions for robotics and trajectory planning.[35, 36] Diffusion models work by gradually adding noise to real trajectory data (forward process) and then learning to reverse this process (denoising) to recover the clean trajectory.[35, 36, 41] Unlike GANs, diffusion models are stable to train and offer superior mode coverage, meaning they can effectively capture rare but critical maneuvers like sudden lane changes or emergency stops.[35, 36]

Recent research has focused on accelerating the diffusion process, as the iterative denoising required for sampling is computationally expensive and potentially too slow for real-time applications.[36, 42] The TrajDD-GAN architecture addresses this "generative model trilemma" (quality vs. diversity vs. efficiency) by combining a spatiotemporal diffusion process with a multimodal conditional GAN.[36] This hybrid approach significantly reduces the number of denoising steps required, allowing for faster generation of high-quality synthetic trajectories that preserve spatiotemporal characteristics.[36]

> [!TIP]
> **Summary of Generative Approaches**
>
> | Generative Approach | Core Strategy | Advantage | Major Drawback |
> | :--- | :--- | :--- | :--- |
> | VAE | Latent space distribution.[38] | Stable training, diverse sampling.[38] | Over-smoothed/blurred output.[38] |
> | GAN | Adversarial minimax game.[38] | Sharp, realistic samples.[38] | Training instability, mode collapse.[38] |
> | Diffusion (DDPM) | Iterative denoising.[35] | State-of-the-art mode coverage.[35] | High computational latency.[19, 36] |
> | Diffusion (DiT) | Transformer-based denoising.[43] | Scalable to complex conditions.[43] | Requires massive training data.[43] |

## Foundational Models and Semantic Reasoning

The integration of Large Foundation Models (LFMs), including Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs), represents a new paradigm shift in trajectory prediction.[5] While traditional deep learning models excel at low-level pattern recognition, they often lack the "common sense" or semantic reasoning required for complex urban scenarios.[5]

### LLM-Based Reasoning and Explainability

LFMs transform trajectory prediction from a purely numerical regression task into one grounded in cognitive reasoning and semantic understanding.[5] By mapping trajectories to linguistic tokens, models can utilize pre-trained knowledge from LLMs to infer intent in "long-tail" or rare scenarios.[5] For example, an LLM-based predictor can reason that a pedestrian carrying a large umbrella in a storm may have a restricted field of view and thus might cross the road unexpectedly.[5, 6] This reasoning capability significantly enhances safety and provides human-like explainability for why a certain prediction was made—a critical requirement for the social acceptance of autonomous vehicles.[5]

### Multimodal Fusion and World-Aware Models

Advanced models are moving toward a modular paradigm where specialized LLMs for spatiotemporal reasoning are integrated with visual perception modules.[5] These frameworks, such as OmniDrive, use counterfactual reasoning to anticipate "what if" scenarios, allowing the vehicle to plan defensively against potential but unobserved risks.[44] By integrating scene semantics, traffic rules, and general-world knowledge, LFMs facilitate more generalizable models that perform better in open-world environments where training data may be sparse.[5, 22]

## Real-Time Efficiency and Streaming Architectures

The practical deployment of trajectory prediction models on autonomous platforms is constrained by the need for ultra-low latency.[19, 23] High-precision models, particularly those involving iterative refinement or complex graph convolutions, often struggle to meet the millisecond-level requirements of on-board hardware.[19, 23, 45]

### Query-Centric Paradigms: QCNet and QCNeXt

The QCNet (Query-Centric Trajectory Prediction) framework addresses a major inefficiency in existing agent-centric models: the need to re-normalize and re-encode the input whenever the observation window slides forward.[19, 20] QCNet introduces a query-centric encoding paradigm that learns representations independent of the global spacetime coordinate system.[19, 20] This allows the model to cache and reuse past computations, spreading the processing cost across observation windows and reducing online inference latency from 8ms to 1ms.[19, 45]

QCNet employs anchor-free queries to generate initial trajectory proposals and anchor-based queries for subsequent refinement, combining the flexibility of data-driven approaches with the stability of anchor-based methods.[19, 45] Its successor, QCNeXt, extends this framework to joint multi-agent prediction, accurately estimating the joint future distribution of multiple agents while maintaining roto-translation invariance in space and translation invariance in time.[46]

### SEAM: Streaming Endpoint-Aware Modeling

The SEAM architecture proposes a lightweight yet accurate approach for real-time trajectory forecasting in a continuous environment.[33] Most existing models treat each frame as a standalone "snapshot," ignoring the global temporal context across evolving scenes.[33, 47] SEAM integrates information from previous predictions using a novel endpoint-aware modeling scheme.[33] It uses the trajectory endpoints from previous forecasts as anchors to extract targeted scenario context encodings.[33] This guides the scene encoder to focus on relevant context—such as a pedestrian near the end of a turning path—without needing compute-intensive refinement iterations.[33]

## 3D Trajectory Estimation: The Vertical Dimension and Physics

For 3D trajectory prediction, particularly in the context of UAVs, robotics, and multi-floor stairs, the $z$-axis (elevation) introduces significant complexity due to the unstructured nature of 3D point clouds and the lack of height-annotated data.[4, 11, 48]

### Physics-Informed 3D Tracking

Accurate 3D trajectory estimation for fast-moving small objects, such as UAVs, requires integrating kinematics motion equations to handle outliers and missed detections caused by occlusion or noise.[4, 10] Frameworks for anti-UAV systems use physics-informed refinement to impose temporal smoothness and kinematic consistency on estimated trajectories.[4] When depth data is invalid or unavailable from sensors like LiDAR or depth cameras, these systems can estimate the $z$-position using kinematic models like $z(t)=z_0 + V_{z0}t - \frac{1}{2}gt^2$ to ensure the path remains physically plausible.[10]

### Gait Recognition and Vertical Displacement

In human-centric robotics and pedestrian navigation, 3D path estimation often relies on inertial sensors (IMU) and factor graph optimization (FGO).[11] These systems detect stationary states and update paths using a "stride length + orientation vector" approach.[11] For vertical changes, a $z$-axis step-based elevation model is used, which computes height change based on stride length and walking posture angle (inclination relative to the ground).[11] This ensures geometric adaptability to different staircase slopes and multi-floor environments, achieving vertical displacement errors within 2%.[11]

> [!TIP]
> **Summary Table: 3D Estimation Techniques**
>
> | 3D Estimation Technique | Application | Core Mechanism | Key Benefit |
> | :--- | :--- | :--- | :--- |
> | Kinematic Equations | UAVs/Small Objects.[10] | $z(t)$ based on gravity/vel.[10] | Handles missing depth/occlusion.[10] |
> | FGO + IMU | Human Navigation.[11] | Posture-angle based elevation.[11] | High accuracy without GNSS.[11] |
> | Physics-Informed DiT | General Video/Motion.[43] | Spacetime motion patches.[43] | Simulates physical world movement.[43] |
> | 3D Gaussian Splatting | Scene Reconstruction.[49] | Ray tracing in splats.[49] | Realistic rendering of specular/3D.[49] |

## Uncertainty, Occlusion, and Certified Robustness

Trajectory prediction models must operate reliably under conditions of noise, sensor imperfection, and occlusion—where agents are partially or fully hidden from view.[23, 50, 51]

### Long-Term Occlusion Handling

Long-term occlusion remains a fundamental challenge for joint-detection-and-tracking systems.[51] When a trajectory is inactivated due to occlusion, appearance features become unreliable for reconnection.[51] The Long-term Spatio-Temporal Prediction (LSTP) module addresses this by using a combination of spatial and temporal transformers to estimate object states over time.[51] The spatial transformer models crowd interaction, while the temporal transformer models the continuity of historical movement.[51] By predicting the "visibilities" of motion prediction boxes, the system can prioritize data association based on occlusion attributes, significantly reducing identity switching even after multiple seconds of occlusion.[40, 51]

### Certified Robustness and Adversarial Defense

Data-driven models often lack robustness to noisy inputs or adversarial examples.[50] To address this, certified trajectory prediction frameworks have been proposed to provide guaranteed robustness.[50] These methods use a certification approach tailored for multi-modal and unbounded outputs.[50] A key innovation is the use of a diffusion-based trajectory denoiser integrated into the prediction method to mitigate the inherent performance drop associated with certification.[50] This ensures that the predictor remains accurate and robust against perturbations that might otherwise cause catastrophic failures in downstream planning algorithms.[50]

### MoE for Long-Tail Uncertainty

Mixture of Experts (MoE) architectures, such as Tra-MoE, are increasingly used to handle the diverse data distributions found in real-world scenarios.[22, 52] By using a sparsely-gated MoE architecture with a Top-1 gating strategy, models can maintain constant computational cost (FLOPs) while effectively benefiting from large-scale out-of-domain data.[52] A hybrid architecture combining statistical ranking and semantic reasoning (tri-expert gate) has demonstrated the ability to outperform single-expert models by 9.5% on the nuPlan-mini dataset.[22] This is particularly relevant for "long-tail" scenarios like sudden cut-ins or dense intersections, where specialized experts can be more effective than a monolithic "one-size-fits-all" model.[22]

## Benchmark Datasets and Performance Evaluation

The development of trajectory prediction models is inextricably linked to the availability of large-scale datasets that capture the complexity of real-world traffic.[17, 53, 54]

### Dataset Comparative Analysis

Modern datasets like Argoverse 2 and the Waymo Open Motion Dataset provide a magnitude more data and complexity than earlier benchmarks.[14, 17, 53] Waymo's dataset is particularly notable for its 8-second forecasting horizon and its collection of 100,000 scenes mined specifically for "interesting" interactions between agents.[17] Argoverse 2 provides 250,000 multi-actor scenarios and detailed HD vector maps with lane connectivity and semantic metadata.[14, 53] nuPlan, on the other hand, focuses on long-term ego-vehicle planning across 1,200 hours of human driving data, auto-labeled using world-class offline perception systems to ensure track quality that exceeds typical on-car systems.[55]

> [!TIP]
> **Summary Table: Comparison of Major Datasets**
>
> | Dataset | Total Hours/Scenes | Time Horizon | Key Characteristic |
> | :--- | :--- | :--- | :--- |
> | nuScenes | 1,000 20s scenes.[8, 53] | 6s.[8] | Multimodal (Lidar, Radar, Camera).[53] |
> | Argoverse 2 | 250,000 scenarios.[14, 53] | 11s.[14] | Focus on interaction-critical events.[53] |
> | Waymo (WOMD) | 570 hours / 100k scenes.[17] | 8s.[17] | Mined for interactive behaviors.[17] |
> | nuPlan | 1,200 hours human driving.[55] | Long-term.[55] | Large-scale offline auto-labeling.[55] |
> | Interaction | 16.5 hours.[54] | 3s.[17, 54] | Highly interactive roundabout/merging.[54] |
> | DeepUrban | N/A (Urban Intersections).[54] | 3D focus.[54] | Captured from 100m altitude.[54] |

### Advanced Evaluation Metrics

While Average Displacement Error (ADE) and Final Displacement Error (FDE) remain the standard for measuring geometric accuracy, they are increasingly seen as insufficient for safety-critical analysis.[21, 56]
- **ADE/FDE**: Measures the mean square error (MSE) or $L_2$ distance over all estimated points or at the final timestep.[21, 57]
- **Miss Rate (MR)**: The percentage of cases where no predicted trajectory is within a given threshold (e.g., 2m) of the ground truth.[8, 29]
- **Collision Score**: Quantifies how many predicted trajectories would result in a collision with the environment or other agents.[54]
- **Off-Road Rate**: The frequency with which a model predicts a path that leaves the driveable surface.[8, 32]
- **Certified Performance Metrics**: Reliability measures that account for guaranteed robustness in the presence of input uncertainty.[50]

Recent studies indicate that higher open-loop accuracy (lower ADE/FDE) does not always translate to better closed-loop driving behavior.[34] Factors like temporal consistency—how much predictions change between successive frames—and compatibility with downstream planning algorithms play a critical role in the overall system performance.[33, 34]

## Industry Implementations and Practical Deployment

Major autonomous vehicle manufacturers have transitioned toward deep learning frameworks that balance predictive power with operational constraints.[7]

### Tesla and Apollo

Tesla's prediction algorithm is an end-to-end vision-based model that forecasts trajectories directly from visual sequences and aggregated contextual information.[6] It continuously learns from massive amounts of data collected across its global fleet, emphasizing a data-driven approach over rule-based constraints.[6] In contrast, Baidu's Apollo system (from release 5.0) incorporates an integrated prediction module supporting multiple specialized predictors: the "Free Move Predictor" for non-lane-constrained obstacles, the "Single Lane Predictor" for standard highway driving, and the "Lane Sequence Predictor" for complex multi-lane maneuvers.[6, 7]

### Aurora and Aurora's System

Aurora utilizes a joint object detection and trajectory forecasting system that combines deep learning with rule-based constraints.[7] This ensures that while the model can learn complex human motion patterns, it remains constrained by the fundamental rules of the road (e.g., not driving on sidewalks).[7] Aurora extensively tests its system across virtual and real-world urban logistics scenarios to ensure safety in heterogeneous traffic.[7]

## Future Research Directions and Conclusions

The future of 3D trajectory prediction lies in bridging the gap between perception and cognitive reasoning.[5] While deep learning models have achieved impressive mean metrics, the "long-tail" problem remains the most significant barrier to full autonomy.[5, 22]

The emergence of Foundation Models and LLMs provides a promising path toward world-aware, causally-grounded models that can explain their predictions and generalize to extreme scenes.[5] Simultaneously, the development of ultra-low latency architectures like QCNet and SEAM ensures that these sophisticated models can be deployed on real-time hardware without sacrificing safety.[19, 33]

Key areas for ongoing research include:
1. **Interactive Game Theory**: Moving beyond data-driven implicit interaction to explicit collaborative models where agents are treated as strategic, goal-oriented players.[22, 23]
2. **Physics-informed Diffusion**: Enhancing generative models with explicit kinematic constraints to ensure that every sampled trajectory is physically achievable.[10, 42, 43]
3. **Occlusion-Robust Sensing**: Integrating 3D Gaussian Splatting and NeRFs for better scene reconstruction and hallucination mitigation in occluded environments.[49, 58, 59]
4. **Certification and Stability**: Developing standard protocols for the safety certification of deep learning predictors, ensuring they remain robust to environmental shifts and adversarial noise.[22, 50]

Ultimately, the successful integration of 3D trajectory prediction into autonomous systems will require a holistic approach that combines spatiotemporal precision, semantic understanding, and real-time computational efficiency.[1, 2, 19] By moving from low-level pattern recognition to high-level intention reasoning, autonomous systems will achieve the level of safety and social compliance required for seamless integration into human environments.[1, 5, 6]

--------------------------------------------------------------------------------

## References

1. Vision-based Multi-future Trajectory Prediction: A Survey - arXiv, https://arxiv.org/html/2302.10463v2
2. A Survey on Vehicle Trajectory Prediction Procedures for Intelligent Driving - MDPI, https://www.mdpi.com/1424-8220/25/16/5129
3. 3DOF Pedestrian Trajectory Prediction Learned from ... - ILIAD Project, http://iliad-project.eu/wp-content/uploads/2018/03/Kevin_UoL_ICRA18.pdf
4. 3D UAV Trajectory Estimation and Classification from Internet Videos via Language Model, https://arxiv.org/html/2603.09070v1
5. Large Foundation Models for Trajectory Prediction in Autonomous Driving: A Comprehensive Survey - arXiv, https://arxiv.org/html/2509.10570v1
6. Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions, https://arxiv.org/html/2503.03262v1
7. Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions, https://arxiv.org/html/2503.03262v3
8. nuScenes prediction task, https://www.nuscenes.org/prediction
9. Evaluation - EvalAI, https://eval.ai/web/challenges/challenge-page/1194/evaluation
10. Physics-Guided Fusion for Robust 3D Tracking of Fast Moving Small Objects - arXiv, https://arxiv.org/html/2510.20126v1
11. A Study on a High-Precision 3D Position Estimation Technique Using Only an IMU in a GNSS Shadow Zone - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12694286/
12. 3D Position Estimation using Deep Learning - Diva-Portal.org, https://www.diva-portal.org/smash/get/diva2:1335815/FULLTEXT01.pdf
13. Interactive trajectory prediction for autonomous driving based on Transformer - Recent, https://ms.copernicus.org/articles/16/87/2025/ms-16-87-2025.html
14. Beyond Features: How Dataset Design Influences Multi-Agent Trajectory Prediction Performance - arXiv, https://arxiv.org/html/2507.05098v1
15. Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals - Proceedings of Machine Learning Research, https://proceedings.mlr.press/v164/deo22a/deo22a.pdf
16. HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction - CVF Open Access, https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf
17. Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset - CVF Open Access, https://openaccess.thecvf.com/content/ICCV2021/papers/Ettinger_Large_Scale_Interactive_Motion_Forecasting_for_Autonomous_Driving_The_Waymo_ICCV_2021_paper.pdf
18. A Survey of Deep Learning-Based Pedestrian Trajectory Prediction ..., https://www.mdpi.com/1424-8220/25/3/957
19. Query-Centric Trajectory Prediction - IEEE Xplore, https://ieeexplore.ieee.org/iel7/10203037/10203050/10203873.pdf
20. Query-Centric Trajectory Prediction - CVPR 2023 Open Access Repository, https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.html
21. Error Metrics for Trajectory Prediction Accuracy - Jaime B. Fernández R., https://jaimefernandezdcu.wordpress.com/2019/02/07/error-metrics-for-trajectory-prediction-accuracy/
22. Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features, https://arxiv.org/html/2511.00126v1
23. A Survey of Autonomous Driving Trajectory Prediction ... - MDPI, https://www.mdpi.com/2075-1702/13/9/818
24. Deep Learning - Robotics and Perception Group, https://rpg.ifi.uzh.ch/research_learning.html
25. Transformer-Based Trajectory Prediction Using LiDAR Data for Situational Awareness in Complex Urban Environments - IEEE Xplore, https://ieeexplore.ieee.org/iel8/8784355/11300375/11277286.pdf
26. DenseTNT: End-to-End Trajectory Prediction From Dense Goal Sets - CVF Open Access, https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_DenseTNT_End-to-End_Trajectory_Prediction_From_Dense_Goal_Sets_ICCV_2021_paper.pdf
27. Attention-Linear Trajectory Prediction - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11511033/
28. Transfer Learning Study of Motion Transformer-based Trajectory Predictions* - arXiv, https://arxiv.org/html/2404.08271v3
29. Henry1iu/TNT-Trajectory-Prediction: A Unofficial Pytorch Implementation of TNT - GitHub, https://github.com/Henry1iu/TNT-Trajectory-Prediction
30. PFR-HiVT: Enhancing Multi-Agent Trajectory Prediction with Progressive Feature Refinement - MDPI, https://www.mdpi.com/2073-8994/18/2/310
31. Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding | OpenReview, https://openreview.net/forum?id=YcmGuwdLoU
32. MTR++: Multi-Agent Motion Prediction With Symmetric Scene ..., https://www.researchgate.net/publication/377361870_MTR_Multi-Agent_Motion_Prediction_with_Symmetric_Scene_Modeling_and_Guided_Intention_Querying
33. Streaming Real-Time Trajectory Prediction ... - CVF Open Access, https://openaccess.thecvf.com/content/WACV2026/papers/Prutsch_Streaming_Real-Time_Trajectory_Prediction_Using_Endpoint-Aware_Modeling_WACV_2026_paper.pdf
34. Closing the Loop: Motion Prediction Models beyond Open-Loop Benchmarks - arXiv, https://arxiv.org/html/2505.05638v1
35. Diffusion models for robotic manipulation: a survey - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12454101/
36. TrajDD-GAN: A Synthetic Mobility Trajectory Generation Solution Based on Diffusion Models - IEEE Xplore, https://ieeexplore.ieee.org/iel8/6287639/10820123/11135496.pdf
37. Trajectory generative models: a survey from unconditional and conditional perspectives, https://d-nb.info/1382757859/34
38. Research and Analysis of VAE, GAN, and Diffusion Generation Models - ResearchGate, https://www.researchgate.net/publication/398877064_Research_and_Analysis_of_VAE_GAN_and_Diffusion_Generation_Models
39. Research and Analysis of VAE, GAN, and Diffusion Generation Models, https://lseee.net/index.php/te/article/view/2057
40. Deep Learning for Human Motion: Advancing Trajectory Prediction and Multi-Object Tracking - mediaTUM, https://mediatum.ub.tum.de/doc/1690798/ziyp4efjmznw3uwqryx4qb19s.2023_Dendorfer_PhD_published.pdf
41. CVPR Poster World-consistent Video Diffusion with Explicit 3D Modeling, https://cvpr.thecvf.com/virtual/2025/poster/32498
42. CVPR Poster Acc3D: Accelerating Single Image to 3D Diffusion Models via Edge Consistency Guided Score Distillation, https://cvpr.thecvf.com/virtual/2025/poster/32628
43. CVPR Poster Tora: Trajectory-oriented Diffusion Transformer for Video Generation, https://cvpr.thecvf.com/virtual/2025/poster/34398
44. CVPR 2025 Papers, https://cvpr.thecvf.com/virtual/2025/papers.html
45. QCNet Aims to Better Guess Where Other Drivers Are Aiming to Boost the Safety of Autonomous Vehicles - Hackster.io, https://www.hackster.io/news/qcnet-aims-to-better-guess-where-other-drivers-are-aiming-to-boost-the-safety-of-autonomous-vehicles-79e9e55ae805
46. QCNeXt: A Next-Generation Framework For Joint Multi-Agent Trajectory Prediction - arXiv, https://arxiv.org/html/2306.10508v1
47. Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling - arXiv, https://arxiv.org/html/2603.01864v1
48. Tree height-growth trajectory estimation using uni-temporal UAV laser scanning data and deep learning | Forestry: An International Journal of Forest Research | Oxford Academic, https://academic.oup.com/forestry/article/96/1/37/6628789
49. CVPR 2025 Awards, https://cvpr.thecvf.com/virtual/2025/awards_detail
50. CVPR Poster Certified Human Trajectory Prediction, https://cvpr.thecvf.com/virtual/2025/poster/34812
51. ORT: Occlusion-robust for multi-object tracking - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12167898/
52. CVPR Poster Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning, https://cvpr.thecvf.com/virtual/2025/poster/34256
53. NuScenes vs Argoverse Datasets - Emergent Mind, https://www.emergentmind.com/topics/nuscenes-and-argoverse-datasets
54. DeepUrban: Interaction-aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery - arXiv, https://arxiv.org/html/2601.10554v2
55. nuPlan - nuScenes, https://www.nuscenes.org/nuplan
56. Beyond ADE and FDE: A Comprehensive Evaluation Framework for Safety-Critical Prediction in Multi-Agent Autonomous Driving Scenarios - arXiv, https://arxiv.org/html/2510.10086v1
57. Summary of Metrics for Trajectory Prediction | Download Scientific Diagram - ResearchGate, https://www.researchgate.net/figure/Summary-of-Metrics-for-Trajectory-Prediction_tbl2_378322685
58. CVPR 2024 Awards, https://cvpr.thecvf.com/virtual/2024/awards_detail
59. Quantification of Occlusion Handling Capability of 3D Human Pose Estimation Framework | Request PDF - ResearchGate, https://www.researchgate.net/publication/359127088_Quantification_of_Occlusion_Handling_Capability_of_3D_Human_Pose_Estimation_Framework
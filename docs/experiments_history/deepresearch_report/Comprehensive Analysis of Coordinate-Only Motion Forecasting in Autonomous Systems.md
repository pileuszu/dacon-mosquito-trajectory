# Comprehensive Analysis of Coordinate-Only Motion Forecasting in Autonomous Systems: Architectures, Social Dynamics, and Robustness Paradigms

The trajectory of autonomous vehicle development has fundamentally shifted from reactive obstacle avoidance to proactive motion forecasting, where the system must anticipate the future states of dynamic agents to ensure safe and efficient navigation.[1, 2] Coordinate-only motion forecasting represents a specialized yet increasingly critical domain within this field, characterized by the prediction of future spatial paths based primarily on historical coordinate sequences and agent dynamics, often operating independently of high-definition maps or raw sensor inputs.[3, 4] This methodology addresses the core challenges of scalability and robustness, particularly in environments where map data is unavailable, outdated, or subject to adversarial manipulation.[5, 6] By reducing the scene to structured, vectorized elements, these models focus on the fundamental kinematic and social principles that govern traffic behavior.[7, 8]

## 1. Theoretical Foundations and the Forecasting Pipeline

Motion forecasting serves as the critical bridge in the autonomous driving stack, receiving outputs from upstream perception and tracking modules and providing essential data for downstream motion planning and control.[2, 9] The primary objective is to estimate the future motion states of traffic participants—including vehicles, pedestrians, and cyclists—over a horizon typically spanning three to eight seconds.[2, 10] Accurate predictions allow the ego vehicle to plan safer, time-saving paths, preventing energy loss and facilitating the implementation of active braking systems that can calculate collision probabilities with higher precision.[2, 11]

The formal problem formulation for coordinate-only forecasting defines a traffic scenario through an ego vehicle and a set of surrounding agents, each represented by a state vector. This state vector, at its most reduced form, consists of two-dimensional position coordinates $(x,y)$ over an observation window.[4] Depending on the sophistication of the tracking module, this vector may be expanded to include orientation, velocity, and acceleration.[4, 12] The task of the trajectory predictor is to map these historical sequences into a set of future trajectory sequences while accounting for the inherent randomness in human actions.[4, 13]

### 1.1. Physics-Based Kinematic Models

Traditional approaches to motion forecasting rely on physics-based models that utilize deterministic physical principles to estimate future movement.[7, 14] These models are highly efficient and require minimal computing resources, making them suitable for short-term predictions in uncomplicated environments.[3, 9]

> [!TIP]
> **Summary Table: Physics-Based Kinematic Models**
>
> | Model Type | Dynamic Assumption | Mathematical Basis | Typical Use Case |
> | :--- | :--- | :--- | :--- |
> | Constant Velocity (CV) | Zero acceleration; constant heading | $p_{t+1} = p_t + v\Delta t$ | Highway cruising; simple straight-line motion [7, 12] |
> | Constant Acceleration (CA) | Fixed rate of speed change | $p_{t+1} = p_t + v\Delta t + 0.5a\Delta t^2$ | Braking or initial acceleration from standstill [9, 12] |
> | Constant Turn Rate (CTRV) | Fixed velocity and angular velocity | Integration of yaw rate and speed | Negotiating roundabouts or curves [7] |
> | Decaying Acceleration | Acceleration gradually returns to zero | Exponential decay of current $a$ | Longer-term predictions where constant $a$ is unrealistic [12] |

While physics-based models provide a robust baseline, they overlook environmental factors and interactions with other agents, limiting their applicability in complex urban scenes where agents frequently change lanes or respond to traffic signals.[3, 7] To mitigate these drawbacks, practitioners often employ 4th or 5th order splines (quintic polynomials) to refine paths, ensuring minimum jerk and accounting for basic vehicle dynamics.[12]

### 1.2. Curvilinear Coordinate Systems and Frenet Frames

A significant challenge in coordinate-based forecasting is the choice of reference frame. Curvilinear coordinate systems, commonly referred to as Frenet frames, are often used to align the representation with the road geometry.[15] In these frames, motion is decomposed into longitudinal (along the road) and lateral (perpendicular to the road) components, which simplifies the formulation of collision avoidance constraints.[15] However, these transformations are prone to singularities in areas with high curvature, such as intersections, where points cannot be uniquely represented if they exceed the radius of the osculating circle.[15] Modern adaptation schemes involve iterative procedures to ensure that the reference path remains $C^2$ continuous, thereby preventing discontinuities in the transformed trajectories.[15]

## 2. Architectures for Coordinate-Only Forecasting

The evolution of neural network architectures has transitioned from simple recurrent structures to complex graph-based and attention-driven models that can implicitly learn the social and kinematic laws of the road.[1, 8, 16]

### 2.1. Recurrent and Sequential Modeling

Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks were among the first deep learning models applied to coordinate forecasting due to their ability to handle variable-length sequential data.[17, 18] In these architectures, the historical coordinates are passed through an MLP to form an internal representation vector, which is then used by an LSTM to generate future waypoints sequentially.[1]

A milestone in this domain was the introduction of the **Social LSTM**, which addressed the independence assumption of standard RNNs.[14, 19] Social LSTM incorporates a social pooling layer that aggregates the hidden states of neighboring agents within a local spatial grid, typically an 8x8 grid.[19, 20] This allows the model to adjust an agent's path based on the proximity and velocity of others, effectively modeling collective behaviors without explicit map information.[20, 21]

### 2.2. Graph Neural Networks and Vectorized Interaction

Because traffic participants do not exist on a rigid grid, Graph Neural Networks (GNNs) have emerged as a superior method for modeling the non-Euclidean relationships between agents.[9, 17] In a GNN framework, agents are represented as nodes, and their interactions are captured through edges that can be adaptively learned based on semantic and motion contexts.[4, 22] VectorNet, for example, uses a vectorized graph-based approach to reduce model parameters significantly compared to rasterized ConvNet models while maintaining high accuracy.[9] These models encode agent histories into polylines and use global graph mechanisms to capture inter-agent dependencies.[1, 23]

### 2.3. Transformer-Based Prediction Paradigms

Transformer architectures have set the current state-of-the-art by leveraging self-attention and cross-attention to capture long-range dependencies and intricate relationships between scene elements.[24, 25, 26] Unlike RNNs, Transformers process entire sequences in parallel, making them more efficient for capturing complex temporal dynamics.[23, 27]

A critical consideration in Transformer models is the coordinate frame used for encoding. Agent-centric models transform all inputs into the local frame of each agent, providing inherent translation and rotation invariance.[28, 29] While these models perform best on major leaderboards, they scale quadratically with the number of agents.[29] Scene-centric models, conversely, use a fixed coordinate system for all agents, offering better scalability but requiring more sophisticated training to achieve pose-invariance.[29]

### 2.4. Generative Modeling and Multi-Modality

Motion forecasting is inherently multi-modal, as a single past trajectory can lead to multiple plausible futures.[24, 30] To capture this uncertainty, coordinate-only models employ several generative strategies.
- **Conditional Variational Autoencoders (CVAE)**: These models learn a latent space of possible futures and sample from it to generate diverse trajectories.[5, 14]
- **Generative Adversarial Networks (GANs)**: Models like Social GAN use a discriminator to ensure that generated trajectories are "socially acceptable" and realistic, while a variety loss encourages the generator to cover the full distribution of possible paths.[14, 21, 22]
- **Diffusion and Flow Matching**: Recent SOTA models utilize diffusion processes to gradually refine noise into a coherent trajectory.[31, 32] TrajFlow, a novel flow-matching framework, predicts multiple plausible trajectories in a single pass, significantly reducing the computational overhead compared to iterative sampling in standard diffusion models.[32]

### 2.5. Discretization and Classification vs. Regression

While many models treat trajectory prediction as a regression task (directly outputting coordinates), some high-performance models decompose the task into classification.[24] In these architectures, continuous values for velocity and yaw are discretized into bins.[24]

> [!TIP]
> **Discretization Parameters**
>
> | Parameter | Binning Method | Classes | Representative Value |
> | :--- | :--- | :--- | :--- |
> | Velocity | 1 m/s intervals | 25 classes | Median of the bin (e.g., 0.5 m/s for 0-1 bin) [24] |
> | Yaw | 5-degree intervals | 71 classes | Median of the bin (e.g., 0 deg for -2.5 to 2.5 bin) [24] |

By converting coordinates into these discrete labels, the models can leverage classification losses, which are often more stable during training and can more easily represent multi-modal "hydra-head" distributions of future paths.[24]

## 3. Social Interaction and Behavioral Modeling

The ability to model how agents influence one another is the hallmark of advanced forecasting systems.[33, 34] This goes beyond simple collision avoidance and enters the realm of "Theory of Mind," where the system predicts how others will react to the ego vehicle's presence.[22, 35]

### 3.1. Social Forces and Collision Avoidance

Social Force Models (SFM) simulate pedestrian and vehicle dynamics as a series of attractive and repulsive forces.[21, 33] A person is modeled as moving toward a goal with an attractive force, while being repelled by obstacles and other pedestrians to maintain a comfortable social distance.[21] In deep learning, these principles are often integrated through custom loss functions, such as the Dynamic Occupied Space (DOS) loss, which penalizes predictions that result in collisions between agent bodies, typically modeled as circular disks with a radius of 0.2 meters.[20]

### 3.2. Interaction-Aware Transformers

Modern models like SocialFormer leverage semantic relationships and road topology to encode agent interactions.[34] These systems use edge-enhanced heterogeneous graph transformers (EHGT) to aggregate spatial and temporal information about maneuvers like lane changing, car following, and yielding.[34] Crucially, they can model intricate questions such as whether an agent behind will slow down to allow a lane change, which is essential for proactive planning.[34]

### 3.3. Courtesy and Game Theory

A sophisticated area of research focuses on the "selfishness" versus "courtesy" of autonomous agents.[35] Purely selfish robots optimize only for their own safety and driving quality, often leading to aggressive behaviors like cutting people off.[35] Courteous planning involves a mathematical "courtesy term" that minimizes the inconvenience brought to other drivers, measured as the increase in another driver's cost due to the robot's planned behavior.[35] Inverse Reinforcement Learning (IRL) is frequently used to study whether such terms accurately reflect real-world human driving patterns.[35]

## 4. Robustness, Generalization, and Mapless Navigation

A primary driver for coordinate-only methods is the need for robustness against sensory noise and environmental shifts.[5, 26, 36]

### 4.1. Vulnerability to Map-Based Attacks

Research into trajectory prediction robustness has revealed that models incorporating context maps are vulnerable to map-based adversarial attacks.[5] Attackers can interfere with these models by adding imperceptible perturbations to context maps—such as slightly altering lane boundaries or road signs—which can increase trajectory prediction errors by 29% to 110%.[5] Coordinate-only models, by focusing on agent trajectories rather than semantic maps, are inherently more resilient to such attacks.[5, 37]

### 4.2. Handling Noisy and Out-of-Sight Observations

In real-world deployment, perception systems are rarely perfect. They suffer from occlusions, ID switches, and tracking drift.[36] Ego-centric (first-person view) observations are particularly prone to perspective distortion.[36] SOTA models like OOSTraj (Out-of-Sight Trajectory Prediction) use vision-positioning denoising to map sensor-based trajectories of out-of-sight objects into visual trajectories, allowing for the prediction of agents even when they are temporarily hidden by physical obstructions.[38]

### 4.3. Generalization Across Geographies

A significant challenge for AI-based forecasting is the domain discrepancy between Western road environments (where datasets like Waymo and Argoverse are collected) and other regions like South Korea or Singapore.[39, 40] Models trained on Western data often experience performance degradation when deployed elsewhere due to different driving behaviors and infrastructure.[39] To address this, techniques such as encoder freezing and full fine-tuning are used to transfer pretrained knowledge to new environments, often boosting accuracy significantly.[39, 41]

## 5. Data Augmentation for Trajectory Forecasting

To train robust models, practitioners use various data augmentation techniques to simulate rare or dangerous scenarios that are under-represented in real-world datasets.[42, 43]

> [!TIP]
> **Summary Table: Data Augmentation Techniques**
>
> | Technique | Implementation | Goal |
> | :--- | :--- | :--- |
> | Position Shifts | Applying random offsets to GPS coordinates | Mimics positioning system noise [42] |
> | Temporal Noise | Introducing 1–3 second deviations in timestamps | Simulates transmission delays [42] |
> | Map Reshaping | Reshaping linear lanes into curved lanes | Diversifies the geometry of the driving scene [43] |
> | Poisson Modeling | Assigning departure times based on Poisson distribution | Captures realistic traffic flow rhythms [42] |
> | Synthetic Synthesis | Rule-based planners on augmented maps | Generates feasible, non-existing trajectories [43] |

Advanced frameworks categorize real-world data by hour and day to extract spatiotemporal features, ensuring that augmented data reflects peak travel periods and typical origin-destination probabilities.[42]

## 6. Benchmark Datasets and Evaluation Metrics

The progress of the field is benchmarked on several large-scale datasets, each offering unique challenges for coordinate-only and multi-agent forecasting.[39, 44, 45]

### 6.1. Principal Forecasting Datasets

> [!TIP]
> **Comparison of Datasets**
>
> | Dataset | City/Context | Characteristics | Evaluation Focus |
> | :--- | :--- | :--- | :--- |
> | Argoverse 1 | Pittsburgh, Miami | 320 hours; 10 Hz sampling | High-resolution trajectory prediction [9, 39] |
> | Argoverse 2 | 6 Diverse US Cities | 763 hours; 250,000 scenarios | Challenging interactions; mapless tracks [39, 45] |
> | nuScenes | Boston, Singapore | 1,000 scenes; 2 Hz sampling | 360-degree sensor fusion; 11 semantic layers [9, 40] |
> | Waymo Open | Multiple US Cities | 103,354 segments; 20s each | Large-scale multi-agent forecasting [9, 39] |
> | DeepUrban | Munich Tal | Dense urban setting | Heavy road user interaction [44] |

Argoverse 2 has recently introduced a Scene Flow track which evaluates models on longer-range predictions (up to 70m) and encourages generalization across multiple datasets.[46]

### 6.2. Performance Metrics

Metrics are categorized into displacement-based, ability-oriented, and stability-oriented categories.[9]
- **Average Displacement Error (ADE)**: The mean Euclidean distance between the ground truth and predicted waypoints. It measures overall path accuracy.[1, 9]
- **Final Displacement Error (FDE)**: The Euclidean distance between the predicted final position and the ground truth final position. It highlights long-term destination accuracy.[9]
- **Miss Rate (MR)**: The ratio of sequences where the minFDE exceeds a threshold (e.g., 2 meters).[9, 47]
- **Drivable Area Compliance (DAC)**: The percentage of predicted trajectories that remain within the legally drivable boundaries.[9]
- **Trajectory Prediction Consistency (TPC)**: A novel metric introduced to evaluate planning stability. It measures the discrepancy between trajectories predicted at consecutive time steps to ensure smooth vehicle control.[48, 49]

## 7. The Rise of Foundation Models and LLMs in Forecasting

The current research frontier (2024–2026) is dominated by the integration of Large Foundation Models (LFMs), particularly Large Language Models (LLMs), into the prediction pipeline.[8, 50]

### 7.1. Trajectory-Language Mapping

This paradigm involves converting continuous multi-agent trajectories into discrete sequences of tokens that an LLM can process.[27, 50] Researchers have developed compact token vocabularies that preserve centimeter-level accuracy while allowing the model to use next-token prediction strategies.[50] This approach enables the model to leverage the vast commonsense reasoning capabilities of LLMs to understand causal relationships, such as "the car will stop because there is a construction zone sign".[8, 50]

### 7.2. Momentum-Aware Driving (MomAD)

To address the limitations of "one-shot" predictions that lead to unstable control, the MomAD framework introduces trajectory and perception momentum.[48, 49] This system uses Topological Trajectory Matching (TTM) with Hausdorff Distance to select optimal planning queries that align with historical paths.[48] By ensuring coherence over a 6-second horizon, MomAD has been shown to reduce collision rates by 26% and improve prediction consistency by 33%.[48, 49]

### 7.3. Zero-Shot and Low-Data Regimes

Pretrained LLMs are increasingly used in zero-shot or low-data regimes, where they can predict trajectories without extensive fine-tuning on a specific dataset.[8, 27] These models excel in long-tail scenarios where traditional data-driven models struggle due to a lack of training examples.[8]

## 8. Conclusions and Future Outlook

The field of coordinate-only motion forecasting is moving toward a highly integrated, foundation-model-driven future where agents are no longer just point masses but participants in a complex social world.[8, 51, 52] The reliance on static HD maps is giving way to "online scene understanding," where geometry and intent are inferred simultaneously from observed tracks.[6, 53]

Key future directions include:
1. **Ultra-Low Latency Inference**: Adapting complex Transformers and LLMs for real-time edge deployment on vehicle hardware.[8, 9]
2. **Explainable End-to-End Models**: Moving toward systems that not only predict where an agent will go but also "explain" the causal reasoning behind the prediction in natural language.[8, 52]
3. **World-Aware Foundations**: Developing pre-trained world models that can predict future states of entire traffic scenes across all modalities—LiDAR, camera, and coordinates—to provide a holistic understanding of reality.[8, 52]

By mastering the coordinate-only paradigm, autonomous systems will achieve a level of generalization and resilience that allows them to navigate unvisited environments and complex social scenarios with the same fluidity as human drivers.[8, 39, 52]

--------------------------------------------------------------------------------

## References

1. How to Design ML Systems for Motion Prediction in Autonomous Vehicles - Medium, https://medium.com/@alexnaydenov/how-to-design-ml-systems-for-motion-prediction-in-autonomous-vehicles-c59e725c0acc
2. A Review of Deep Learning-Based Vehicle Motion Prediction for Autonomous Driving, https://www.mdpi.com/2071-1050/15/20/14716
3. Motion Forecasting for Autonomous Vehicles: A Survey - ResearchGate, https://www.researchgate.net/publication/388963778_Motion_Forecasting_for_Autonomous_Vehicles_A_Survey
4. Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions, https://arxiv.org/html/2503.03262v1
5. Robustness of Trajectory Prediction Models Under Map-Based Attacks - CVF Open Access, https://openaccess.thecvf.com/content/WACV2023/papers/Zheng_Robustness_of_Trajectory_Prediction_Models_Under_Map-Based_Attacks_WACV_2023_paper.pdf
6. Autonomous Navigation without HD Prior Maps - DSpace@MIT, https://dspace.mit.edu/handle/1721.1/147214
7. Motion Forecasting for Autonomous Vehicles: A Survey - arXiv, https://arxiv.org/html/2502.08664v1
8. Large Foundation Models for Trajectory Prediction in Autonomous Driving: A Comprehensive Survey - arXiv, https://arxiv.org/html/2509.10570v1
9. Single-Vehicle Trajectory Prediction: A Review and ... - IEEE Xplore, https://ieeexplore.ieee.org/iel8/8782711/11268961/11366921.pdf
10. CVPR 2024 Workshop on Autonomous Driving, https://cvpr2024.wad.vision/
11. Motion Planning under Uncertainty for On-Road Autonomous Driving - Carnegie Mellon University's Robotics Institute, http://www.ri.cmu.edu/pub_files/2014/6/ICRA14_0863_Final.pdf
12. map_based_prediction - Autoware Universe Documentation, https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_map_based_prediction/
13. Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions, https://arxiv.org/html/2503.03262v3
14. PHR-Net: Proposal-Level Historical Retrieval for Non-Stationary Temporal Consistency in Trajectory Prediction - MDPI, https://www.mdpi.com/2624-8921/8/5/109
15. Robust and Efficient Curvilinear Coordinate Transformation with Guaranteed Map Coverage for Motion Planning - mediaTUM, https://mediatum.ub.tum.de/doc/1740514/oo9dhtktnpk40v2rp7hc2xc0j.24_04_24_CCosy_Paper_final_submission_reupload_MEDIATUM_VERSION.pdf
16. A Survey of Deep Learning-Based Pedestrian Trajectory Prediction: Challenges and Solutions - Semantic Scholar, https://pdfs.semanticscholar.org/03f5/e54c9b3eaa83b0847c3fff3be6062e995274.pdf
17. Skeleton-based motion prediction: A survey - Frontiers, https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1051222/full
18. Skeleton-based motion prediction: A survey - Frontiers, https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1051222/pdf
19. Social LSTM for Trajectory Prediction | PDF | Prediction | Applied Mathematics - Scribd, https://www.scribd.com/document/892373227/Final-Poster
20. Social LSTM with Dynamic Occupancy Modeling for Realistic Pedestrian Trajectory Prediction - arXiv, https://arxiv.org/html/2511.09735v1
21. Human motion trajectory prediction using the Social Force Model for real-time and low computational cost applications - UPCommons, https://upcommons.upc.edu/bitstreams/846e9740-91a1-49da-82b7-73a58abbc1db/download
22. Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks | Request PDF - ResearchGate, https://www.researchgate.net/publication/329740296_Social_GAN_Socially_Acceptable_Trajectories_with_Generative_Adversarial_Networks
23. Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting - arXiv, https://arxiv.org/html/2603.28091v1
24. Uncertainty-Aware Multimodal Trajectory Prediction via a Single ..., https://www.mdpi.com/1424-8220/25/1/217
25. SimpliHuMoN: Simplifying Human Motion Prediction - arXiv, https://arxiv.org/html/2603.04399v1
26. Goal-based Trajectory Prediction for improved Cross-Dataset Generalization - arXiv, https://arxiv.org/html/2507.18196v1
27. Trajectory Prediction Meets Large Language Models: A Survey - ResearchGate, https://www.researchgate.net/publication/392405835_Trajectory_Prediction_Meets_Large_Language_Models_A_Survey
28. Dynamic Intent Queries for Motion Transformer-based Trajectory Prediction - arXiv, https://arxiv.org/html/2504.15766v1
29. Narrowing the Coordinate-frame Gap in Behavior Prediction Models ..., https://waymo.com/research/narrowing-the-coordinate-frame-gap-in-behavior-prediction-models-distillation-for-efficient-and-acc/
30. EqDrive: Efficient Equivariant Motion Forecasting with Multi-Modality for Autonomous Driving, https://arxiv.org/html/2310.17540v3
31. Motion Generation: A Survey of Generative Approaches and Benchmarks - arXiv, https://arxiv.org/html/2507.05419v1
32. TrajFlow: Multi-modal Motion Prediction via Flow Matching - arXiv, https://arxiv.org/html/2506.08541v2
33. Modeling local behavior for predicting social interactions towards human tracking, https://nij.ojp.gov/library/publications/modeling-local-behavior-predicting-social-interactions-towards-human-tracking
34. SocialFormer: Social Interaction Modeling with Edge-enhanced Heterogeneous Graph Transformers for Trajectory Prediction - arXiv, https://arxiv.org/pdf/2405.03809
35. Social Interaction-Aware Motion Planning for Autonomous Vehicles, https://msc.berkeley.edu/research/social_interaction_IRL.html
36. EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations, https://arxiv.org/html/2510.00405v2
37. Robustness of Trajectory Prediction Models Under Map-Based Attacks - YouTube, https://www.youtube.com/watch?v=Gh5tr0XVPeU
38. OOSTraj: Out-of-Sight Trajectory Prediction With Vision-Positioning Denoising - CVF 2024 Open Access Repository, https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_OOSTraj_Out-of-Sight_Trajectory_Prediction_With_Vision-Positioning_Denoising_CVPR_2024_paper.html
39. Argoverse: 3D Tracking and Forecasting With Rich Maps - ResearchGate, https://www.researchgate.net/publication/338507163_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps
40. nuPlan - nuScenes, https://www.nuscenes.org/nuplan
41. STraj: Self-training for Bridging the Cross-Geography Gap in Trajectory Prediction, https://ojs.aaai.org/index.php/AAAI/article/download/34432/36587
42. Vehicle Trajectory Data Augmentation Using Data Features and ..., https://www.mdpi.com/2079-9292/14/14/2755
43. UC Berkeley - eScholarship.org, https://escholarship.org/content/qt9nb7p9ns/qt9nb7p9ns.pdf
44. DeepUrban: Interaction-aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery - arXiv, https://arxiv.org/html/2601.10554v2
45. Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting, https://www.researchgate.net/publication/366821672_Argoverse_2_Next_Generation_Datasets_for_Self-Driving_Perception_and_Forecasting
46. AV2 2024 Scene Flow Challenge Announcement - Argoverse, https://www.argoverse.org/sceneflow.html
47. nuScenes prediction task, https://www.nuscenes.org/prediction
48. Don't Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving, https://cvpr.thecvf.com/virtual/2025/poster/34802
49. arXiv:2503.03125v1 [cs.RO] 5 Mar 2025, https://arxiv.org/pdf/2503.03125
50. Trajectory Prediction Meets Large Language Models: A Survey - arXiv, https://arxiv.org/html/2506.03408v1
51. SocialGen: Modeling Multi-Human Social Interaction with Language Models - arXiv, https://arxiv.org/html/2503.22906v1
52. Challenge 2024 | OpenDriveLab, https://opendrivelab.com/challenge2024/
53. [Literature Review] MapVision: CVPR 2024 Autonomous Grand Challenge Mapless Driving Tech Report - Moonlight | AI Colleague for Research Papers, https://www.themoonlight.io/en/review/mapvision-cvpr-2024-autonomous-grand-challenge-mapless-driving-tech-report
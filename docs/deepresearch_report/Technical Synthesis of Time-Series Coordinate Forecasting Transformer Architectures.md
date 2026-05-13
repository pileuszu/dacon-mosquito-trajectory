# Technical Synthesis of Time-Series Coordinate Forecasting Transformer Architectures

The evolution of time-series forecasting (TSF) has moved into a high-dimensional spatial regime, where the objective is no longer merely the prediction of a scalar value but the anticipation of precise spatial coordinates for moving agents. This transition is necessitated by the rapid expansion of autonomous driving, unmanned aerial systems, and maritime logistics, all of which require modeling the complex interplay between temporal dynamics and spatial constraints.[1, 2, 3] The Transformer architecture, originally devised for natural language processing, has emerged as the dominant paradigm for these tasks due to its inherent ability to model long-range dependencies and perform parallelized computations across large-scale datasets.[4, 5, 6]

## Theoretical Framework and the Shift from Recurrence to Attention

The historical reliance on Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) units for coordinate forecasting was rooted in their ability to maintain a hidden state that captures temporal dependencies.[3] However, these architectures suffer from several fundamental deficiencies in a spatial context. Primarily, the sequential processing of data prevents the model from attending to distant but relevant historical maneuvers without passing through all intermediate states, leading to the "vanishing gradient" problem.[7, 8, 9] In coordinate forecasting, where a maneuver performed ten seconds ago may be the primary indicator of a future turn, this limitation is critical.

Transformers address this by utilizing the self-attention mechanism, formulated as:
$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$
This mechanism allows for direct pairwise comparisons between every timestep in a sequence, effectively bypassing the sequential bottleneck of RNNs.[8, 10] For coordinate forecasting, this means the model can simultaneously evaluate an agent's current velocity, a social interaction that occurred several frames prior, and the distal environmental constraints provided by a map.[11, 12, 13]

> [!TIP]
> **Comparison of Sequential vs. Attention-Based Architectures**
>
> | Architecture | Temporal Processing | Dependency Modeling | Scalability | Complexity |
> | :--- | :--- | :--- | :--- | :--- |
> | RNN / LSTM | Sequential | Limited by hidden state decay.[7] | Poor parallelization.[5] | $O(L)$ |
> | Transformer | Parallel | Global via self-attention.[7, 8] | High training efficiency.[5, 10] | $O(L^2)$ |
> | Sparse Transformer | Parallel | Targeted/Selective.[9, 14] | Optimized for long sequences.[9] | $O(L \log L)$ |

While the theoretical advantages of Transformers are clear, empirical evidence indicates that the complexity of these models does not always translate to superior performance on simpler time-series tasks.[15, 16] Research into the "finite-sample gap" suggests that for certain linear stationary processes, simpler linear models can outperform complex Transformers.[15] This has led to a nuanced understanding where Transformers are viewed as high-capacity models that require significant data and sophisticated training strategies to realize their potential in coordinate forecasting.[7, 17, 18]

## Feature Engineering and Multi-Scale Dimensionality

The input to a coordinate forecasting Transformer is rarely a raw stream of $(x,y,z)$ points. The instability of sensor data and the presence of environmental noise require robust preprocessing methods.[1, 3]

### Frequency-Domain Augmentation

One innovative approach involves the synergy of Fast Fourier Transform (FFT) and Transformer architectures.[1] By converting temporal coordinate windows into frequency-domain features, the model can capture periodic motion patterns that are often obscured in the time domain. These features are represented as complex numbers, further decomposed into magnitude and phase [1]:
- **Magnitude**: Defines the amplitude of specific frequency components, indicating the intensity of a particular motion pattern.[1]
- **Phase**: Determines the relative timing of these components, essential for understanding the synchronization between different moving parts or agents.[1]

Training a model on these FFT-derived features allows the Transformer to focus on the structural regularities of the motion rather than the raw, noisy coordinate values, leading to significant accuracy improvements across diverse sensor datasets.[1]

### Dimension-Segment-Wise (DSW) Embedding

In maritime and high-latitude navigation, models like **Crossformer** utilize Dimension-Segment-Wise (DSW) embedding to handle the geographical coordinates (LAT/LON).[19] Unlike traditional Transformers that concatenate all variables at a single timestep into a vector, DSW embedding segments each variable's time series independently.[19] This captures the unique temporal patterns of latitude and longitude separately before allowing the Transformer to model their cross-dimensional correlations through a Two-Stage Attention (TSA) layer.[19]

> [!TIP]
> **Summary Table: Embedding and Processing Methods**
>
> | Embedding Method | Mechanism | Core Benefit |
> | :--- | :--- | :--- |
> | Standard Pointwise | Concatenates all coordinates at time $t$.[19] | Simple, low overhead. |
> | DSW Embedding | Segments time series per variable.[19] | Preserves variable-specific temporal logic. |
> | Patching | Groups contiguous time points into patches.[14, 20] | Reduces sequence length, improves local context. |

## Spatio-Temporal Interaction and Social Dynamics

The movement of a single agent is rarely independent of its surroundings. Coordinate forecasting must model "social dynamics"—the invisible forces of attraction, repulsion, and cooperation that govern multi-agent environments.[21, 22, 23]

### Spatio-Temporal Graph Transformer (STAR)

The Spatio-Temporal grAph tRansformer (STAR) framework treats a crowd as a dynamic graph where pedestrians are nodes and their potential interactions are edges.[24, 25] The core innovation of STAR is the **TGConv** mechanism, a Transformer-based graph convolution that improves upon standard Graph Attention Networks (GAT).[25] TGConv treats self-attention as a form of message passing on a fully connected graph, allowing the model to adaptively weight the importance of neighboring agents based on their relative coordinates and velocities.[25]

STAR captures interactions by interleaving:
1. **Spatial Transformers**: These compute intra-graph interactions at a specific timestep, identifying which agents are most relevant to each other's immediate future.[25]
2. **Temporal Transformers**: These model the inter-graph dependencies, focusing on the historical consistency of each individual agent's trajectory.[25, 26]

This interleaving strategy is complemented by an external read-writable graph memory module, which provides temporal smoothing and helps the model maintain the "social identity" of agents that might be momentarily occluded or stationary.[25, 27]

### Agent-Aware Attention in AgentFormer

While STAR interleaves space and time, the **AgentFormer** model argues that such a separation is suboptimal because it prevents an agent's state at one time from directly influencing another agent's state at a future time.[23, 28] AgentFormer flattens the multi-agent trajectory into a single sequence and introduces "agent-aware attention".[23]

This mechanism generates two distinct sets of keys and queries:
- **Intra-Agent Queries**: Targeted at the agent's own past positions to ensure kinematic smoothness.[23]
- **Inter-Agent Queries**: Targeted at other agents' past positions to model social interactions.[23]

By using masked operations to preserve agent identity within the sequence, AgentFormer allows for direct socio-temporal modeling, which is particularly effective in complex urban intersections where simultaneous attention to one's own path and the paths of others is critical for safety.[23, 28]

## Handling Multimodality and Latent Intentions

In coordinate forecasting, the future is inherently uncertain. An agent approaching a junction may turn left, turn right, or continue straight—a phenomenon known as multimodality.[12, 23, 29]

### The Motion Transformer (MTR) Framework

The Motion Transformer (MTR) framework represents a paradigm shift in addressing multimodality by modeling it as a joint optimization problem of global intention localization and local movement refinement.[11, 12]

MTR replaces dense goal-candidate grids with a small set of learnable motion query pairs.[12] The process begins with the generation of "intention points" through k-means clustering of the endpoints of ground-truth trajectories in the training set.[12] These points represent common destinations or motion modes (e.g., a "slow right turn" or a "high-speed straight").

- **Global Intention Localization**: Static queries are initialized as positional embeddings of these intention points, narrowing down the uncertainty of the future trajectory by assigning each query to a specific motion mode.[12]
- **Local Movement Refinement**: Dynamic searching queries are then used to iteratively refine the predicted coordinates. In each decoder layer, these queries gather fine-grained features from the agent's history and the surrounding map context.[12]
- **GMM Output Head**: The model predicts multiple trajectories, each associated with a probability score from a Gaussian Mixture Model (GMM). This ensures that the output is not a single, averaged "mean trajectory" (which would be physically impossible) but a diverse set of plausible paths.[11, 12]

> [!TIP]
> **Summary Table: MTR Components**
>
> | Component | Function | Implementation |
> | :--- | :--- | :--- |
> | Intention Points | Define potential destinations.[12] | K-means clustering on training data. |
> | Static Queries | Anchor each mode in space.[12] | PE of clustered intention points. |
> | Dynamic Queries | Refine local movement.[12] | Updated embeddings per decoder layer. |
> | Loss Function | Optimize accuracy and diversity.[11] | GMM Negative Log-Likelihood + L1 loss. |

## Scene Context and Map-Integrated Architectures

The physical environment—represented by HD maps, lane boundaries, and traffic rules—is a primary constraint on coordinate forecasting.[2, 30, 31]

### Vectorized Map Encoding

Early models used rasterized (image-based) maps, which are computationally heavy and lack semantic precision.[31] Modern Transformers favor vectorized representations, where road elements are treated as polylines.[11, 13] VectorNet and its derivatives encode these polylines using local self-attention, creating "context tokens" that the trajectory Transformer can attend to during decoding.[11, 13]

The **Lane Interaction Transformer (LITransformer)** builds upon this by introducing a lane topology encoder.[2] This module uses direction-sensitive, multi-scale dilated graph convolutions to fuse geometric and semantic lane attributes.[2] By converting lane centerlines into topology-aware representations, the model can more effectively predict how a vehicle will adhere to lane geometry, particularly in complex scenarios like merging lanes or roundabouts.[2]

### Scene-Aware Lightweight Models (ASTRA)

A critical challenge for real-world deployment is the computational burden of processing HD maps alongside multi-agent trajectories.[18, 30] **ASTRA** (A Scene-aware TRAnsformer-based model) addresses this by utilizing a U-Net-based key-point extractor to capture essential scene features without the need for explicit segmentation maps.[30] ASTRA processes spatial, temporal, and social dimensions concurrently by embedding the graph structure into the token sequence prior to the attention mechanism.[30] This "graph-aware" Transformer achieves state-of-the-art performance while utilizing seven times fewer parameters than contemporary models, making it highly suitable for in-vehicle processing units.[30, 32]

## Empirical Evaluation and Benchmarking

The validation of coordinate forecasting Transformers relies on large-scale datasets that provide high-resolution trajectories and rich environmental context.

### Metrics of Accuracy and Compliance

Evaluating a coordinate forecast requires measuring both the precision of the predicted path and its adherence to physical and social norms.[31, 33, 34]
- **Average Displacement Error (ADE)**: Calculated as the mean Euclidean distance over $T$ timesteps. This metric provides a general sense of the "drift" between the prediction and reality.[13, 19]
- **Final Displacement Error (FDE)**: Focuses solely on the error at the final prediction horizon, which is critical for destination-based tasks like robotic planning.[13, 35]
- **Minimum Metrics (mADE/mFDE)**: For multimodal models, these assess the error of the best prediction among the top $K$ generated trajectories, effectively evaluating the model's ability to cover the correct "mode".[11, 13]
- **Scene Compliance**: Recent studies emphasize safety-driven metrics, such as the cross-boundary rate (how often a predicted vehicle leaves the drivable area) and collision scores.[11, 36, 37]

### Comparative Dataset Performance

The choice of dataset significantly impacts the perceived performance of Transformer models.

> [!TIP]
> **Comparison of Transformer Performance Across Datasets**
>
> | Dataset | Scene Type | Interaction Level | Map Complexity | Typical Model Improvement |
> | :--- | :--- | :--- | :--- | :--- |
> | ETH / UCY | Pedestrian | High social | Low | Transformers outperform LSTM by ~10%.[32] |
> | Argoverse | Urban vehicle | High topological | High | Transformers show 12-31% error reduction.[13, 38] |
> | Waymo | Mixed driving | Critical/Joint | High | MTR achieves 1st rank on leaderboards.[11, 12] |
> | DeepUrban | Drone intersection | Dense interaction | High | Boosting accuracy by up to 44% when added to nuScenes.[37] |

Research on the Argoverse dataset reveals that Transformers are particularly adept at capturing long-range contextual cues from maps and other agents, with ADE increasing only marginally as the number of neighbors grows—a stark contrast to the degradation observed in LSTM and TCN models.[38]

## Computational Efficiency and Sparse Attention

The quadratic complexity of self-attention ($O(L^2)$) presents a scaling problem for long-sequence coordinate forecasting.[9] For instance, analyzing several minutes of historical aircraft trajectory data to predict a future conflict can quickly exhaust GPU memory.[4, 9]

### Informer and ProbSparse Attention

The Informer architecture introduced a key insight: attention weight distributions in time-series data are often highly sparse, with most queries only caring about a small subset of keys.[9, 14] By implementing **ProbSparse** attention, which uses a KL-divergence-based measure to select the most "active" queries, Informer reduces the complexity to $O(L \log L)$.[9] This allows Transformers to handle historical windows that are an order of magnitude longer than previously possible.[9]

### Deformable Attention (DeformableTST)

While patching (grouping time steps) can reduce complexity, it can also lead to an over-reliance on local chunks of data, which is suboptimal for forecasting tasks unsuitable for patching.[14] **DeformableTST** addresses this by using deformable attention, which helps the model focus on a small number of "important" time points throughout the entire sequence.[14] This improves the model's ability to adapt to varying input lengths and reduces the performance degradation often seen in advanced Transformers when patching is not feasible.[14]

## Domain-Specific Case Studies

The versatility of coordinate forecasting Transformers is evident in their application across diverse scientific and industrial domains.

### Aviation and 4D Trajectory Operations

In Air Traffic Management (ATM), the transition to Trajectory-based Operations (TBO) requires high-precision 4D trajectory prediction (latitude, longitude, altitude, and time).[4, 35] The **Bayesian Spatio-Temporal grAph tRansformer (B-STAR)** effectively models the interactions between multiple aircraft under uncertainty.[4, 26] By utilizing a deterministic encoder for complexity management and a Bayesian decoder approximated with Variational Inference, B-STAR can provide probability trajectories with confidence intervals—a requirement for safety-critical conflict detection.[26]

### Maritime Navigation and Ship Trajectories

Predicting ship trajectories involves handling irregular AIS data and geographical coordinates.[19] Models like **Crossformer** demonstrate a reduction in average error of over 60% compared to GRU and LSTM baselines.[19] The robustness of these models in predicting both latitude and longitude dimensions is essential for intelligent ship scheduling and maritime safety.[19]

### Epidemic and Infectious Disease Trends

Surprisingly, the Transformer architectures developed for spatial coordinate forecasting have found utility in epidemic modeling.[20, 39] While epidemic data is often univariate (infected counts), the "spatial" dimension is represented by different geographic regions (e.g., US states). The **Spatial–Temporal Synchronous Graph Transformer (STSGT)** uses multi-head self-attention on a graph where vertices are states and edges are physical distances, allowing for the simultaneous capture of complex non-linear spatial and temporal dependencies in disease spread.[40]

## Comparative Dissection: Transformers vs. Recurrent and Convolutional Models

A critical analysis of the "coordinate forecasting landscape" reveals that while Transformers are the current state-of-the-art, the choice of architecture must be informed by specific operational constraints.[7, 10, 38]

### Performance under Traffic Density

Transformers demonstrate the most robust performance as traffic density increases.[38] In scenarios with six or more neighboring agents, the ADE of LSTM and TCN models degrades significantly, whereas Transformer models maintain stable accuracy due to the global reach of the attention mechanism.[3, 38]

### Convergence and Training Stability

The "Pre-LN" Transformer variation (applying layer normalization before the attention layer) has been found to stabilize training without the need for extensive learning rate warmups, which was a common challenge in earlier Transformer-based forecasting.[5] When compared to LSTMs, Transformers typically exhibit a higher convergence rate and higher final accuracy in complex multi-agent urban environments.[3]

> [!TIP]
> **Summary Table: Comparison of Models**
>
> | Feature | LSTM | TCN | Transformer |
> | :--- | :--- | :--- | :--- |
> | Inference Speed | Moderate | Very Fast.[38] | Slower (Quadratic).[7] |
> | Long-Range | Limited.[7] | Fixed by receptive field. | Strong (Global).[7, 8] |
> | Resource Demand | Low | Low | High (Enterprise GPUs).[10] |
> | Data Requirement | Moderate | Moderate | High.[7] |

## Future Horizons: Foundation Models and Physics-Informed AI

The trajectory of coordinate forecasting research points toward two main frontiers: the generalization power of foundation models and the safety of physics-informed architectures.

### The Rise of Foundation Models

Recent evidence suggests that foundation models like Google’s **TimesFM** and Meta’s **Lag-Llama**, trained on billions of time points, can perform remarkably well on downstream forecasting tasks with zero-shot or few-shot learning.[20, 39] These models leverage the "scaling laws" of Transformers to generalize across different data distributions, potentially outperforming traditional statistical and mechanistic models in data-limited settings.[39]

### Physics-Informed Safe Reinforcement Learning

The integration of Transformers into reinforcement learning (RL) frameworks, such as the **Constraint-Guided Behavior Transformer (CoBT-SRL)**, represents the next step in autonomous decision-making.[41] By using Transformers as the policy network, these systems can capture long-range historical states and actions to predict future maneuvers while adhering to safety constraints.[41] This is particularly relevant for intersection management, where the model must navigate complex, stochastic interactions while maintaining a 100% safety guarantee.[41, 42]

## Summary of Findings

The research indicates that the "Time-series Coordinate Forecasting Transformer" is not a single model but a diverse family of architectures tailored to the nuances of spatial data. From the frequency-domain analysis of raw sensor streams to the intention-based decoding of multimodal futures, Transformers have addressed the fundamental limitations of recurrence. However, the transition has introduced new challenges in computational complexity and the need for scene compliance. The future of the field lies in the successful fusion of these high-capacity attention mechanisms with the rigid constraints of physical reality, ensuring that the "coordinates" of the future are not only predicted with precision but also with absolute safety.

--------------------------------------------------------------------------------

## References

1. Time Series Forecasting Model Based on the Adapted Transformer Neural Network and FFT-Based Features Extraction - MDPI, https://www.mdpi.com/1424-8220/25/3/652
2. LITransformer: Transformer-Based Vehicle Trajectory Prediction Integrating Spatio-Temporal Attention Networks with Lane Topology and Dynamic Interaction - MDPI, https://www.mdpi.com/2079-9292/14/24/4950
3. Transformer-Based Trajectory Prediction Using LiDAR Data for Situational Awareness in Complex Urban Environments - IEEE Xplore, https://ieeexplore.ieee.org/iel8/8784355/11300375/11277286.pdf
4. Research on flight trajectory prediction method based on transformer - SPIE Digital Library, https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13018/1301854/Research-on-flight-trajectory-prediction-method-based-on-transformer/10.1117/12.3024772.full
5. Transformer (deep learning) - Wikipedia, https://en.wikipedia.org/wiki/Transformer_(deep_learning)
6. Transformer-Based Vehicle-Trajectory Prediction at Urban Low-Speed T-Intersection - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12298809/
7. Transformers vs. LSTMs for time series forecasting - Kaggle, https://www.kaggle.com/discussions/general/562677
8. LSTM-to-Transformer Transition - Emergent Mind, https://www.emergentmind.com/topics/lstm-to-transformer-transition
9. How Informer Solved the Transformer Scaling Problem for Time-Series Forecasting | by William Tenhunen | Medium, https://medium.com/@williamtenhunen/how-informer-solved-the-transformer-scaling-problem-for-time-series-forecasting-32834ee02676
10. RNNs vs LSTM vs Transformers | SabrePC Blog, https://www.sabrepc.com/blog/deep-learning-and-ai/rnns-vs-lstm-vs-transformers
11. Motion Transformer Model - Emergent Mind, https://www.emergentmind.com/topics/motion-transformer-mtr-model
12. Motion Transformer with Global Intention Localization ... - MPG.PuRe, https://pure.mpg.de/rest/items/item_3482807_5/component/file_3501489/content
13. Interactive trajectory prediction for autonomous driving based on Transformer - Recent, https://ms.copernicus.org/articles/16/87/2025/ms-16-87-2025.html
14. DeformableTST: Transformer for Time Series Forecasting without Over-reliance on Patching - NIPS papers, https://proceedings.neurips.cc/paper_files/paper/2024/file/a0b1082fc7823c4c68abcab4fa850e9c-Paper-Conference.pdf
15. Why Do Transformers Fail to Forecast Time Series In-Context? - arXiv, https://arxiv.org/html/2510.09776v1
16. WHY DO TRANSFORMERS FAIL TO FORECAST TIME SERIES IN-CONTEXT? - OpenReview, https://openreview.net/pdf?id=eBCk0nXz17
17. A Review of Pedestrian Trajectory Prediction Methods Based on Deep Learning Technology, https://pmc.ncbi.nlm.nih.gov/articles/PMC12694338/
18. Transformers for Multivariate Time Series Forecasting: Comprehensive Analysis, Challenges, Research Opportunities, and Future Pr - IEEE Xplore, https://ieeexplore.ieee.org/iel8/6287639/11323511/11352790.pdf
19. Ship trajectory prediction via a transformer-based model by considering spatial-temporal dependency - OAE Publishing Inc., https://www.oaepublish.com/articles/ir.2025.29
20. Foundation time series models for forecasting and policy evaluation in infectious disease epidemics | medRxiv, https://www.medrxiv.org/content/10.1101/2025.02.24.25322795v1.full-text
21. Multi-Person 3D Motion Prediction with Multi-Range Transformers, https://proceedings.neurips.cc/paper_files/paper/2021/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
22. Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review, https://arxiv.org/html/2506.14831v3
23. AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting - CVF Open Access, https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.pdf
24. cunjunyu/STAR: [ECCV 2020] Code for "Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction" - GitHub, https://github.com/cunjunyu/STAR
25. Spatio-Temporal Graph Transformer Networks for Pedestrian ..., https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570494.pdf
26. B-STAR: Multi-Aircraft Trajectory Prediction Network Architecture. In ..., https://www.researchgate.net/figure/B-STAR-Multi-Aircraft-Trajectory-Prediction-Network-Architecture-In-B-STAR-trajectory_fig1_360535740
27. MTP-STG: Spatio-Temporal Graph Transformer Networks for Multiple Future Trajectory Prediction in Crowds - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12737231/
28. AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting, https://www.researchgate.net/publication/358994093_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting
29. MTP-STG: Spatio-Temporal Graph Transformer Networks for Multiple Future Trajectory Prediction in Crowds - MDPI, https://www.mdpi.com/1424-8220/25/24/7466
30. ASTRA: A Scene-aware TRAnsformer-based model for trajectory prediction - arXiv, https://arxiv.org/html/2501.09878v1
31. NuScenes vs Argoverse Datasets - Emergent Mind, https://www.emergentmind.com/topics/nuscenes-and-argoverse-datasets
32. ASTRA: A Scene-aware Transformer-based Model for Trajectory Prediction | OpenReview, https://openreview.net/forum?id=fqSVqPcaVi
33. Beyond ADE and FDE: A Comprehensive Evaluation Framework for Safety-Critical Prediction in Multi-Agent Autonomous Driving Scenarios - arXiv, https://arxiv.org/html/2510.10086v1
34. TrACT: A Training Dynamics Aware Contrastive Learning Framework for Long-tail Trajectory Prediction - arXiv, https://arxiv.org/html/2404.12538v1
35. (PDF) Long-Term Trajectory Prediction Model Based on Transformer - ResearchGate, https://www.researchgate.net/publication/376638146_Long-term_trajectory_prediction_model_based_on_Transformer
36. STGT-Gen: Spatio-Temporal Graph Transformer for Multi-Vehicle Traffic Scenario Generation - Technical Paper - SAE Mobilus, https://saemobilus.sae.org/papers/stgt-gen-spatio-temporal-graph-transformer-multi-vehicle-traffic-scenario-generation-2025-01-7316
37. DeepUrban: Interaction-aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery - arXiv, https://arxiv.org/html/2601.10554v2
38. (PDF) A Comparative Study of LSTM, Transformer, and Temporal Convolutional Networks for Multi-Step Vehicle Trajectory Forecasting in Dynamic Traffic Environments - ResearchGate, https://www.researchgate.net/publication/398054726_A_Comparative_Study_of_LSTM_Transformer_and_Temporal_Convolutional_Networks_for_Multi-Step_Vehicle_Trajectory_Forecasting_in_Dynamic_Traffic_Environments
39. Foundation time series models for forecasting and policy evaluation in infectious disease epidemics - ResearchGate, https://www.researchgate.net/publication/389346428_Foundation_time_series_models_for_forecasting_and_policy_evaluation_in_infectious_disease_epidemics
40. Spatial–Temporal Synchronous Graph Transformer network (STSGT) for COVID-19 forecasting - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC9577246/
41. Constraint-Guided Behavior Transformer for Centralized Coordination of Connected and Automated Vehicles at Intersections - MDPI, https://www.mdpi.com/1424-8220/24/16/5187
42. Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions, https://arxiv.org/html/2503.03262v1
# Theoretical and Architectural Foundations of Translation-Invariant Trajectory Prediction in Autonomous Driving

Trajectory prediction serves as the critical bridge between the perception of the current environment and the downstream planning and control modules in autonomous driving systems.[1, 2] As autonomous vehicles (AVs) navigate through complex, dynamic urban environments, the ability to accurately forecast the future movements of surrounding agents—including vehicles, cyclists, and pedestrians—is paramount for ensuring both safety and traffic efficiency.[3, 4] Over the past decade, the field has transitioned from simple physics-based models to sophisticated deep learning architectures that attempt to model complex social interactions and environmental constraints.[4, 5] A fundamental challenge in this domain is ensuring that the prediction model remains robust and consistent regardless of the global position of the scene, a property known as translation invariance. Achieving this requires a nuanced understanding of how spatial representations are constructed, how coordinate frames are normalized, and how geometric symmetries are exploited within neural network architectures.[6, 7]

## Fundamentals of Spatial Invariance and Equivariance in Motion Forecasting

The concepts of invariance and equivariance are central to the design of robust trajectory prediction systems. While these terms are occasionally used interchangeably in colloquial technical discourse, they represent distinct mathematical properties under the action of a transformation group, such as the translation group $T(2)$ or the special Euclidean group $SE(2)$.[8] The ability of a system to maintain these properties determines its capacity for generalization across diverse geographic regions and its robustness to the inherent noise and shifts in sensor data.[9, 10]

### Defining Translation Invariance and Equivariance

A function or model $f(x)$ is considered invariant to translation if its output remains unchanged when the input $x$ is shifted by a vector $v$.[10] Formally, let $g_v$ be a translation operator such that $g_v(x)=x+v$. A system is translation-invariant if $f(g_v(x))=f(x)$.[8] In the context of trajectory prediction, translation invariance is highly desirable for high-level scene understanding and classification tasks, such as intent recognition.[11, 12] For instance, a model should recognize a "left-turn maneuver" regardless of whether that maneuver occurs at a specific intersection in a training dataset or a previously unseen one in a new city.[13, 14] The exact global coordinates should not influence the categorical representation of the agent's intent or the underlying semantic features of the scene.[14, 15]

In contrast, translation equivariance implies that if the input is shifted, the output shifts in a predictable, corresponding manner.[10] Formally, a function $f$ is equivariant to a translation $g_v$ if there exists a transformation $g'_v$ (often the same translation) such that $f(g_v(x))=g'_v(f(x))$.[8] Trajectory prediction is inherently an equivariant task because the predicted future positions must be relative to the input positions.[6, 16] If an entire traffic scene is shifted five meters to the north, the predicted trajectories must also shift five meters to the north to remain physically valid within the coordinate frame.[7, 17] The distinction between these two properties is summarized in the following comparison.

> [!TIP]
> **Summary Table: Invariance vs. Equivariance**
>
> | Property | Mathematical Definition | Role in Autonomous Systems | Application in Prediction |
> | :--- | :--- | :--- | :--- |
> | Invariance | $f(g(x))=f(x)$ | Recognition and Classification | Intent classification, feature encoding, global context extraction |
> | Equivariance | $f(g(x))=g(f(x))$ | Localization and Regression | Coordinate regression, waypoint generation, spatial interaction modeling |
> | Symmetry Group | $SE(2)$ or $T(2)$ | Transformation handling | Managing shifts, rotations, and reflections in the BEV plane |

Achieving these properties through architecture rather than data augmentation alone is a major focus of modern research.[18, 19, 20] While convolutional neural networks (CNNs) are often cited as being "architecturally invariant" to translation, empirical evidence suggests they are primarily equivariant in their convolutional layers and only achieve approximate invariance through pooling operations or intensive data augmentation.[9, 11, 12, 14]

### The Impact of Spatial Symmetries on Robustness

The requirement for translation invariance stems from the need for models to handle the lack of a fixed, global coordinate system that is universally optimal for all traffic participants.[21, 22] Autonomous vehicles operate in a dynamic "ego-centric" world, yet they must predict the behavior of other agents who have their own "agent-centric" perspectives.[23, 24] If a model is not translation-invariant, it may learn spurious correlations between the absolute coordinates of a training site and the movement patterns observed there, leading to a catastrophic failure in generalization when the vehicle is deployed in a new environment.[2, 21, 25]

The underlying trends in the research suggest that while early models relied on the inherent (though limited) translation equivariance of CNNs applied to rasterized images, there is a distinct move toward "hard-coding" these symmetries into the architecture itself.[6, 7, 26] This shift is driven by the realization that data augmentation, while effective at increasing the volume of training data, is a "soft" constraint that the model can still violate in edge cases.[19] By contrast, architectural invariance provides a mathematical guarantee that the model's output will remain stable under transformation, which is critical for safety-critical components in the autonomous driving stack.[16, 17]

## Evolution of Scene Representations for Invariant Modeling

The method by which a traffic scene is represented significantly impacts the model's ability to maintain spatial robustness and achieve translation invariance.[3, 27] Research has identified three primary ways to represent agent paths: discrete point sequences, parametric curves (such as B-splines or polynomials), and grid occupancy maps.[3, 27] Each of these representations offers different advantages for achieving invariance.

### Rasterization-Based Approaches and CNNs

Early deep learning approaches for motion forecasting relied heavily on rasterized representations, where agent trajectories and high-definition (HD) map features (lanes, crosswalks, traffic lights) were rendered into multi-channel bird's-eye view (BEV) images.[6, 28] These images are then processed by 2D CNNs, such as ResNet-18, to extract context features.[29, 30, 31]

The rasterization process, however, introduces several critical limitations. First, it results in inevitable information loss due to discretization, especially for long-range dependencies where a single pixel may represent a large physical area.[28, 31] Second, the maps have a complex graph structure with topological connections that are inefficient for 2D convolutions to capture.[28] For example, lanes that are spatially close in the BEV image may be semantically disconnected (e.g., opposing lanes separated by a barrier), but a CNN would process them through the same local filter.[28, 31] Furthermore, CNNs are not naturally invariant to rotation or scale, meaning the model's interpretation of a scene can change if the vehicle's orientation changes unless the model is trained with exhaustive rotation-based data augmentation.[11, 14, 32]

### Vectorized and Graph-Based Representations

To overcome the inefficiencies of rasterization, contemporary SOTA models have largely shifted toward vectorized representations.[6, 26] These methods treat HD maps and agent trajectories as sets of polylines or vectors, preserving the precise geometry and topology of the road network without the overhead of rendering.[26, 28, 33]

In a vectorized framework, a polyline is represented as a sequence of vectors $v_i$, where each vector contains the coordinates of its start and end points, along with semantic attributes.[26, 34] This approach allows the use of Graph Neural Networks (GNNs) or Transformers to process the scene elements as nodes in a graph.[6, 26, 35] By operating on relative coordinates and connectivity rather than a fixed grid, these models are more naturally suited for achieving translation invariance.[6, 34]

> [!TIP]
> **Summary Table: Scene Representation Types**
>
> | Representation Type | Primary Mechanism | Advantages for Invariance | Disadvantages |
> | :--- | :--- | :--- | :--- |
> | Rasterized (Grid) | 2D CNNs, Pooling | Leverages vision techniques, easy fusion | Lossy discretization, high compute, no inherent rotation invariance |
> | Vectorized (Graph) | GNNs, Transformers | High precision, efficient, preserves topology | Complex graph construction, requires structured HD maps |
> | Parametric (Curves) | Polynomials, B-Splines | Smooth paths, low parameter count | Sensitive to noise, struggles with non-linear behaviors |
> | Grid Occupancy | Probabilistic mapping | Handles uncertainty, interpretable | Computationally intensive for long horizons |

The transition from rasterized to vectorized representations represents a second-order shift in the philosophy of motion forecasting: the move from "seeing" the scene as an image to "understanding" the scene as a relational graph of entities.[24, 26, 35] This evolution allows models to maintain a stable representation of the environment even as the vehicle's position and orientation shift, which is the foundational requirement for translation-invariant prediction.[6, 34]

## The Coordinate Transformation Problem: Paradigms and Trade-offs

A central problem in designing translation-invariant systems is the selection of the coordinate frame in which to perform modeling.[21, 24] Research has categorized these into three distinct paradigms: agent-centric, scene-centric, and the emerging query-centric paradigm.[21, 36, 37]

### The Agent-Centric Paradigm and Normalization

The agent-centric paradigm is currently the dominant approach in the literature.[23, 24] In this scheme, the environment is normalized for each target agent by transforming the past and surrounding trajectories relative to that specific agent's current position and orientation.[21, 23] Typically, the origin (0,0) is set at the agent's position at the last observed timestep, and the coordinate axes are rotated to align with the agent's heading.[6, 34]

This approach provides a powerful form of spatial invariance. Because every agent's movement is analyzed in its own local coordinate system, the model can more easily learn "primitive" motion patterns that are independent of global location.[21, 22, 24] For example, a lane change maneuver looks identical in the local frame whether it happens on a highway or a city street.[6, 38] However, the efficiency cost is high. In complex multi-agent scenarios, the system must re-normalize and re-encode the entire environment for every single agent being predicted.[21, 36] This leads to significant redundant computation, as the same lane segments and neighbor trajectories are processed multiple times in different local reference frames.[36, 37]

### The Scene-Centric Paradigm and Efficiency

The scene-centric paradigm attempts to resolve these redundancies by modeling all agents and static map elements within a single global or shared reference frame.[21, 37] While this is computationally efficient—encoding the scene once for all participants—it places a heavy burden on the model to "learn" translation invariance from the data.[21, 22]

Scene-centric models are often prone to overfitting on the specific static layouts of the training data.[21] Because they lack the explicit normalization of the agent-centric approach, their prediction accuracy can drop significantly when faced with subtle changes in lane layouts or traffic signs in unseen environments.[21] Some researchers argue that scene-centric methods are less "data-efficient" because they generate only one sample per scene, whereas agent-centric methods generate multiple training samples by viewing the same scene from the perspective of every agent.[37]

### The Query-Centric Breakthrough

The query-centric paradigm, introduced in frameworks such as QCNet, represents a hybrid solution aimed at achieving both the invariance of agent-centric models and the efficiency of scene-centric ones.[36] QCNet learns representations that are independent of the global spacetime coordinate system.[36] By using a query-based design, the model can reuse past computations even as the observation window slides forward.[36, 39] This "streaming scene encoding" avoids the redundant re-normalization steps required by traditional agent-centric models.[36] By sharing invariant scene features among all target agents, it enables parallel multi-agent trajectory decoding, which is essential for the real-time requirements of autonomous vehicle control stacks.[36]

> [!TIP]
> **Comparison of Coordinate Paradigms**
>
> | Paradigm | Spatial Reference | Efficiency | Invariance Level |
> | :--- | :--- | :--- | :--- |
> | Agent-Centric | Target agent's pose at t=0 | Low (Redundant encoding) | High (Explicit normalization) |
> | Scene-Centric | Global/Fixed frame | High (Encode once) | Low (Implicit/Learned) |
> | Query-Centric | Spacetime-independent query | High (Parallel decoding) | High (Invariant scene features) |
> | Polar-Centric | Radius and Angle from Ego | Moderate | High (Relative distance/direction) |

The causal relationship between coordinate normalization and prediction robustness is clear: models that explicitly anchor their representations to the agents' local poses are far more resilient to global shifts.[6, 22] However, the ripple effect of this choice is felt in system latency, where agent-centric models can struggle to scale as the number of traffic participants increases.[21, 36]

## Architectural Mechanisms for Invariance in Landmark Models

Several landmark models have defined the state of the art by implementing specific architectural mechanisms for translation invariance and spatial robustness.[6, 26, 28]

### VectorNet: Hierarchical Polylines and Coordinate Normalization

VectorNet was one of the first models to successfully implement a hierarchical GNN on vectorized map and agent data.[26, 34] To ensure that the learned representations are invariant to absolute locations, VectorNet employs a specific normalization strategy: all coordinates are normalized to be centered around the target agent's last observed position.[34, 40]

The architecture is split into two levels:
1.  **Local Polyline Subgraphs**: These GNNs process individual polylines (e.g., a single lane or an agent's history). Since each polyline is its own subgraph, the model can capture the spatial locality of individual road components.[26, 34]
2.  **Global Interaction Graph**: A fully-connected graph then models the higher-order interactions among all polyline-level features.[26, 34]

By vectorizing the inputs, VectorNet avoids lossy rendering and achieves an order of magnitude reduction in floating-point operations (FLOPs) compared to CNN-based approaches, while simultaneously improving accuracy on datasets like Argoverse.[26]

### LaneGCN: Topology-Aware Graph Convolutions

LaneGCN takes a different approach by focusing on the "lane graph," which explicitly preserves the connectivity and topology of the road network.[28] Instead of polyline-level nodes, LaneGCN uses polyline segments as map nodes to capture higher resolution.[28, 31]

To ensure translation invariance and effectively model long-range dependencies, LaneGCN extends graph convolutions with several specialized components:
- **Multiple Adjacency Matrices**: The model uses four types of connections (predecessor, successor, and left/right neighbors) to capture the legal and physical constraints of the road.[28]
- **Along-Lane Dilation**: Similar to dilated convolutions in 1D or 2D CNNs, this operator expands the receptive field along the lane direction, allowing the model to "see" further down a road segment without increasing the number of parameters.[28, 31]
- **Actor-Map Fusion**: LaneGCN employs a fusion network with four interaction blocks (actor-to-lane, lane-to-lane, lane-to-actor, and actor-to-actor), ensuring that all spatial relationships are modeled as relative displacements.[28, 31]

> [!TIP]
> **Summary of Landmark Model Architectures**
>
> | Model | Map Representation | Interaction Mechanism | Spatial Normalization |
> | :--- | :--- | :--- | :--- |
> | VectorNet | Polyline set | Global attention (Transformer) | Centered on target agent |
> | LaneGCN | Structured lane graph | Dilated GCN + Attention | 2D displacements ($\Delta p$) |
> | HiVT | Local regions + Global | Hierarchical Transformers | Translation-invariant features |
> | QCNet | Query-centric | Streaming attention | Global spacetime independent |

The analysis indicates that LaneGCN's reliance on topology (connectivity) rather than just geometry provides a more robust form of translation invariance.[28] Even if the entire map is shifted, the connectivity between nodes (the "predecessor" of a lane is still the same lane) remains unchanged, allowing the model to maintain consistent internal activations.[28, 31]

## Geometric Deep Learning and the Quest for SE(2) Equivariance

Recent advancements in Geometric Deep Learning have sought to go beyond simple translation invariance to achieve full $SE(2)$ or $SO(2)$ equivariance.[7, 17, 41] $SE(2)$ equivariance implies that the model's predictions are stable under both translations and rotations in the 2D plane—a critical requirement for autonomous vehicles that may approach the same intersection from different directions.[7, 16]

### EqMotion and Equivariant Particle Prediction

EqMotion is a pioneering model that theoretically guarantees sequence-to-sequence motion equivariance.[7] Most existing methods overlook the fundamental principle that if an input motion is transformed under a Euclidean geometric transformation (translation, rotation, or reflection), the output must be transformed in an equivalent way.[7]

EqMotion achieves this through several dedicated designs:
- **Equivariant Geometric Feature Learning**: This module learns features that are "Euclidean transformable," meaning they preserve the underlying geometric properties of the motion.[7]
- **Invariant Interaction Reasoning**: While the motion itself is equivariant, the interaction between agents (e.g., the "force" of a social interaction) is modeled as being invariant to the input's transformation.[7, 16]
- **Sample Efficiency**: A significant finding is that equivariant networks are highly sample-efficient. EqMotion achieved comparable performance using only 5% of training data compared to other models using the full dataset.[7]

### Rotary Phase Encoding and Geometric Bias

In the context of Transformer-based solvers for motion and routing problems, the introduction of Rotary Phase Encoding (RoPhE) provides a theoretical guarantee for strict $SO(2)$ equivariance within the attention layer.[41] RoPhE decouples asymmetric physical distances from rotation-stable geometric features, allowing the self-attention mechanism to capture translational invariance and symmetric geometric structures more effectively.[41]

This geometric perspective has implications beyond simple vehicle prediction. For instance, in indoor trajectory forecasting (SITUATE), leveraging equivariant and invariant geometric learning modules allows models to capture the intrinsic symmetries of human movement and the physical layouts of spaces like supermarkets or train stations.[42] The ability to decouple the "what" (invariant intent) from the "where" (equivariant position) is a central theme in these advanced architectures.[7, 42]

## The Role of Attention and Transformers in Spatial Robustness

Transformers have become the dominant architecture for motion forecasting due to their strong ability to model both temporal and spatial information.[4, 43] However, their inherent permutation invariance requires specific mechanisms to handle spatial relationships.[44, 45]

### Relative Positional Encodings (RPE)

A critical challenge in applying Transformers to geometric tasks is encoding the spatial relationships between tokens that originate from different coordinate systems.[44] While absolute positional encodings (PE) are common in sequence modeling, they lack the relative position bias necessary for spatial robustness.[15, 44]

The research highlights several key developments:
- **Rotary Position Embedding (RoPE)**: This method encodes relative positions through rotation matrices applied to query and key vectors, allowing for relative position bias without explicit bias terms.[44]
- **Unified RoPE (URoPE)**: An extension that unifies inter-camera and intra-image spatial positions into a single mechanism, which is particularly relevant for end-to-end vision-based prediction models.[44]
- **Spatial Similarity Distance Correlation (SSDC)**: A metric used to quantify how spatial structure is preserved in token representations. Studies show that Transformers trained with positional encodings shift toward an "index-anchored" spatial organization that persists even under content disruption, enhancing robustness to distribution shifts.[15]

### Factorized and Latent Query Attention

Models like Wayformer and SceneTransformer explore ways to trade off quality and efficiency in these large-scale attention networks.[46, 47]
- **Factorized Attention**: Applying self-attention over each dimension (temporal vs. spatial) sequentially or in an interleaved fashion.[46, 47]
- **Latent Query Attention**: Using a smaller set of latent queries to control the feature dimension, decoupling the computation needed from the input length or resolution.[46, 47] This is inspired by Perceiver architectures and is effective for fusing heterogeneous inputs like road geometry and agent history.[46, 48]

The choice between "early fusion" (processing all modalities simultaneously) and "late fusion" (dedicated encoders per modality) has a notable impact on how the model learns spatial relationships.[48, 49] Early fusion, despite its simplicity, is often modality-agnostic and achieves state-of-the-art results on benchmarks like Argoverse and WOMD.[48]

## Architectural Invariance versus Data Augmentation: A Comparative Analysis

A long-standing debate in the field is whether spatial invariance should be "hard-coded" into the architecture or "learned" through data augmentation.[18, 19, 20]

### Data Augmentation as Orbit Averaging

Data augmentation (DA) is the process of adding transformed versions of the data (e.g., rotated or translated) to the training set.[19] Theoretically, training with DA is equivalent to learning with an "orbit-averaged" loss function, where the "orbit" is the group of all possible transformations applied to a sample.[19] This leads to variance reduction in the model's predictions, but it may introduce a bias if the augmentation group is too large or if the data distribution is not perfectly invariant under that group.[19]

### Empirical Findings on Invariance Mechanisms

Empirical studies on CNNs have produced surprising results: architectural choices, such as the number of pooling layers or filter size, often have only a secondary effect on translation invariance.[9, 18] Instead, training data augmentation is identified as the most critical factor for obtaining translation-invariant representations.[18] However, standard CNNs systematically fail to recognize new objects at untrained locations unless they are specifically designed for it or trained on "naturalistic" datasets that encourage the learning of deep perceptual rules.[14]

> [!TIP]
> **Factors Influencing Translation Invariance**
>
> | Factor | Influence on Translation Invariance | Theoretical Basis |
> | :--- | :--- | :--- |
> | Data Augmentation | Primary Factor | Orbit-averaged loss [19] |
> | Pooling Layers | Secondary Factor | Approximate local invariance [11, 18] |
> | Receptive Field Size | Secondary Factor | Gradual sensitivity reduction [9] |
> | Vectorized Normalization | Primary Factor (Vector models) | Explicit coordinate anchoring [34] |
> | Equivariant Layers | High (Hard-coded) | Symmetry group theory [7] |

The analysis suggests a "ripple effect" where the lack of architectural invariance leads to a massive increase in the training data requirements.[7, 20] For instance, models that are architecturally equivariant (like EqMotion) can reach high performance with a fraction of the data required by non-equivariant models, which must "see" every maneuver at every possible angle to learn the same rules.[7]

## Evaluating Generalization and Spatial Robustness Across Benchmarks

The true measure of a translation-invariant model is its ability to generalize across different geographical regions and scenarios.[13, 25] Large-scale datasets like Argoverse 1 & 2, nuScenes, and the Waymo Open Motion Dataset (WOMD) have been instrumental in pushing the boundaries of this research.[13, 50]

### Cross-Dataset and Cross-Domain Generalization

Recent studies using the "UniTraj" framework have evaluated state-of-the-art models on their out-of-domain generalization capabilities, such as training on WOMD and testing on nuScenes.[25]
- **PerReg+ (Perceiver with Register Queries)**: This model achieved a 6.8% reduction in B-FDE on smaller datasets through pretraining, and a 11.8% reduction in cross-domain tests compared to its non-pretrained variant.[25]
- **Multi-dataset Pretraining**: Exposing models to a broad range of driving behaviors across different cities and datasets significantly enhances their performance and adaptability to unseen data distributions.[25]
- **DeepUrban Dataset**: Adding high-density environment data from different altitudes (e.g., Munich Tal intersections) to training sets like nuScenes has been shown to boost prediction accuracy by up to 44% in ADE and FDE, highlighting the importance of scene diversity for generalization.[13]

### Fragility to Scene Perturbations

Despite the advancements, models often exhibit unexpected fragility.[2] Research into "scene attacks" revealed that even high-performing models like LaneGCN can see a 60% increase in off-road predictions when faced with minor perturbations in road topologies.[2] This suggests that "learned" awareness of traffic rules and road boundaries is not always robust to spatial shifts.[2] The causal relationship here is that models lacking strict kinematic and road-map constraints are more likely to generate physically infeasible trajectories when their internal spatial representations are pushed outside the training distribution.[2, 51]

## Conclusion and Strategic Recommendations

The research into translation-invariant trajectory prediction has reached a critical inflection point, transitioning from "image-like" interpretations to "graph-relational" and "query-centric" paradigms.[6, 26, 36] The significance of this transition cannot be overstated; the move to vectorized, equivariant architectures is not just about improving accuracy metrics by "shaving off centimeters," but about building systems that are inherently stable, data-efficient, and capable of operating in the unpredictable diversity of the real world.[7, 50]

The analysis of the current landscape leads to several nuanced conclusions:
1. **Hard-coding Symmetry is Superior**: While data augmentation is a powerful tool, architectural designs that explicitly incorporate translation invariance and rotation equivariance (e.g., EqMotion, QCNet) provide safer, more stable foundations for autonomous driving control stacks.[7, 36]
2. **The Shift to Relative Representations**: The use of delta-coordinates, displacements, and polar coordinates is essential for decoupling a maneuver from its global location, allowing models to learn the "laws of physics" and "social norms" in a way that generalizes across any city.[28, 38, 51]
3. **Efficiency and Scaling**: The "agent-centric" paradigm, while robust, is being challenged by "query-centric" and "scene-centric" approaches that offer the parallelism required for high-speed, multi-agent environments without sacrificing spatial invariance.[21, 36]

For researchers and engineers, the path forward involves a deepening integration of geometric deep learning with domain-specific knowledge—such as kinematic feasibility and road-rule adherence—to ensure that the next generation of autonomous vehicles can navigate safely and intuitively, regardless of where in the world they are deployed.[2, 17, 51]

--------------------------------------------------------------------------------

## References

1. Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions - arXiv, https://arxiv.org/pdf/2503.03262
2. Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving - arXiv, https://arxiv.org/html/2505.06740v3
3. A Survey of Autonomous Driving Trajectory Prediction: Methodologies, Challenges, and Future Prospects - MDPI, https://www.mdpi.com/2075-1702/13/9/818
4. Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions, https://arxiv.org/html/2503.03262v1
5. A Survey on Vehicle Trajectory Prediction Procedures for Intelligent Driving - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12390385/
6. HiVT: Hierarchical Vector Transformer for Multi ... - GitHub Pages, https://wkui.github.io/HiVT.pdf
7. EqMotion: Equivariant Multi-Agent Motion Prediction With Invariant Interaction Reasoning, https://cvpr.thecvf.com/virtual/2023/poster/22899
8. What is the difference between "equivariant to translation" and "invariant to translation" - Data Science Stack Exchange, https://datascience.stackexchange.com/questions/16060/what-is-the-difference-between-equivariant-to-translation-and-invariant-to-tr
9. Quantifying Translation-Invariance in Convolutional Neural Networks - ResearchGate, https://www.researchgate.net/publication/322306292_Quantifying_Translation-Invariance_in_Convolutional_Neural_Networks
10. What is the difference between "equivariant to translation" and "invariant to translation" - GeeksforGeeks, https://www.geeksforgeeks.org/machine-learning/what-is-the-difference-between-equivariant-to-translation-and-invariant-to-translation/
11. Translational Invariance Vs Translational Equivariance - Towards Data Science, https://towardsdatascience.com/translational-invariance-vs-translational-equivariance-f9fbc8fca63a/
12. Translation Invariance & Equivariance in Convolutional Neural Networks - Paperspace Blog, https://blog.paperspace.com/pooling-and-translation-invariance-in-convolutional-neural-networks/
13. DeepUrban: Interaction-aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery - arXiv, https://arxiv.org/html/2601.10554v2
14. Learning Translation Invariance in CNNs - Generalisation in Mind & Machine - University of Bristol, https://mindandmachine.blogs.bristol.ac.uk/files/2020/11/Translation_Invariance_NeurIPS_SVRHM_2020.pdf
15. A Geometric Perspective on Robustness in Vision Transformers : r/deeplearning - Reddit, https://www.reddit.com/r/deeplearning/comments/1t9xsua/a_geometric_perspective_on_robustness_in_vision/
16. arXiv:2403.11304v1 [cs.RO] 17 Mar 2024, https://arxiv.org/pdf/2403.11304
17. [2403.11304] Pioneering SE(2)-Equivariant Trajectory Planning for Automated Driving - arXiv, https://arxiv.org/abs/2403.11304
18. Quantifying Translation-Invariance in Convolutional Neural Networks - Stanford Vision Lab, http://vision.stanford.edu/teaching/cs231n/reports/2016/pdfs/107_Report.pdf
19. A Group-Theoretic Framework for Data Augmentation - NIPS papers, https://papers.nips.cc/paper/2020/file/f4573fc71c731d5c362f0d7860945b88-Paper.pdf
20. Revisiting Data Augmentation for Rotational Invariance in Convolutional Neural Networks | Request PDF - ResearchGate, https://www.researchgate.net/publication/331838234_Revisiting_Data_Augmentation_for_Rotational_Invariance_in_Convolutional_Neural_Networks
21. A HYBRID AGENT-CENTRIC AND SCENE-CENTRIC APPROACH FOR MULTI-AGENT TRAJECTORY PREDICTION, https://libeldoc.bsuir.by/bitstream/123456789/58675/1/Tang_Yi_A_hybrid.pdf
22. arXiv:2307.14187v1 [cs.CV] 26 Jul 2023, https://arxiv.org/pdf/2307.14187
23. PHR-Net: Proposal-Level Historical Retrieval for Non-Stationary Temporal Consistency in Trajectory Prediction - MDPI, https://www.mdpi.com/2624-8921/8/5/109
24. Agent-Centric Paradigm in AI Systems - Emergent Mind, https://www.emergentmind.com/topics/agent-centric-paradigm
25. CVPR Poster Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting, https://cvpr.thecvf.com/virtual/2025/poster/33122
26. VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation, https://waymo.com/research/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation/
27. (PDF) A Survey of Autonomous Driving Trajectory Prediction: Methodologies, Challenges, and Future Prospects - ResearchGate, https://www.researchgate.net/publication/395513866_A_Survey_of_Autonomous_Driving_Trajectory_Prediction_Methodologies_Challenges_and_Future_Prospects
28. arXiv:2007.13732v1 [cs.CV] 27 Jul 2020, https://arxiv.org/pdf/2007.13732
29. Research on the improvement of the LaneGCN trajectory prediction algorithm - Oxford Academic, https://academic.oup.com/tse/article-pdf/4/4/tdac034/48351595/tdac034.pdf
30. Research on the improvement of the LaneGCN trajectory prediction algorithm | Transportation Safety and Environment | Oxford Academic, https://academic.oup.com/tse/article/4/4/tdac034/6956891
31. Learning Lane Graph Representations for Motion Forecasting - ECVA | European Computer Vision Association, https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470528.pdf
32. Deep Learning – Equivariance and Invariance, https://www.doc.ic.ac.uk/~bkainz/teaching/DL/notes/equivariance.pdf
33. Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals - Proceedings of Machine Learning Research, https://proceedings.mlr.press/v164/deo22a/deo22a.pdf
34. [2005.04259] VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation - ar5iv, https://ar5iv.labs.arxiv.org/html/2005.04259
35. Spatio-Temporal Context Graph Transformer Design for Map-Free Multi-Agent Trajectory Prediction, https://ieeexplore.ieee.org/iel7/7274857/7448921/10306305.pdf
36. Query-Centric Trajectory Prediction, https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.html
37. SceneMotion: From Agent-Centric Embeddings to Scene-Wide Forecasts - arXiv, https://arxiv.org/html/2408.01537v3
38. Digital-Twin Losses for Lane-Compliant Trajectory Prediction at Urban Intersections - arXiv, https://arxiv.org/html/2603.05546v1
39. CVPR Poster Query-Centric Trajectory Prediction, https://cvpr.thecvf.com/virtual/2023/poster/21235
40. VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized ..., https://arxiv.org/abs/2005.04259
41. Dubins-Aware NCO: Learning SE(2)-Equivariant Representations for Heading-Constrained UAV Routing - MDPI, https://www.mdpi.com/2504-446X/10/1/59
42. SITUATE: Indoor Human Trajectory Prediction through Geometric Features and Self-Supervised Vision Representation - arXiv, https://arxiv.org/html/2409.00774v1
43. MS 3 M: Multi-Stage State Space Model for Motion Forecasting | OpenReview, https://openreview.net/forum?id=MmOQY71YHw
44. URoPE: Universal Relative Position Embedding across Geometric Spaces - arXiv, https://arxiv.org/html/2604.18747v1
45. ICML Poster Learnable Spatial-Temporal Positional Encoding for Link Prediction, https://icml.cc/virtual/2025/poster/45924
46. Wayformer: Motion Forecasting via Simple & Efficient Attention Networks - GitHub Pages, https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/wayformer.html
47. Wayformer: Motion Forecasting via Simple & Efficient Attention Networks - IEEE Xplore, https://ieeexplore.ieee.org/iel7/10160211/10160212/10160609.pdf
48. Wayformer: Motion Forecasting via Simple & Efficient Attention Networks - Waymo, https://waymo.com/research/wayformer/
49. [2207.05844] Wayformer: Motion Forecasting via Simple & Efficient Attention Networks - arXiv, https://arxiv.org/abs/2207.05844
50. Closing the Loop: Motion Prediction Models beyond Open-Loop Benchmarks - arXiv, https://arxiv.org/html/2505.05638v1
51. Relative Position Matters: Trajectory Prediction and Planning with Polar Representation, https://arxiv.org/html/2508.11492v1
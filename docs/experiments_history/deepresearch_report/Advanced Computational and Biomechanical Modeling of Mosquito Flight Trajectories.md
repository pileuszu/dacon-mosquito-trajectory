Advanced Computational and Biomechanical Modeling of Mosquito Flight Trajectories: A Multi-Scale Integrative Review
The quantification and prediction of mosquito flight trajectories represent a critical frontier in both biological physics and global public health. As the primary vectors for devastating pathogens including malaria, dengue, Zika, and West Nile virus, mosquitoes are directly responsible for over 700,000 deaths annually.[1, 2] The ability to accurately model their movement across scales—ranging from the high-frequency biomechanics of a single wingbeat to the long-range, wind-borne migration of populations across continents—is essential for designing the next generation of vector control interventions.[3, 4] Traditional modeling approaches, which often relied on coarse population-level diffusion estimates, are being rapidly superseded by high-fidelity, data-driven frameworks that leverage three-dimensional (3D) tracking, machine learning, and computational fluid dynamics (CFD).[5, 6, 7] These advanced models reveal that mosquito flight is not a stochastic process but a highly regulated behavioral sequence governed by the non-linear integration of multi-sensory cues.[8]
Sensory Integration and the Hierarchical Navigation Paradigm
Mosquito navigation is characterized by a sophisticated, state-dependent sensory hierarchy that allows the insect to locate hosts within complex and often turbulent environments.[9, 10] This process involves the sequential activation of olfactory, visual, and thermal sensing systems, where each stage serves to narrow the search space and transition the mosquito into a more refined behavioral state.[11, 12, 13]
Long-Range Olfactory Activation and Plume Tracking
Host localization typically initiates with the detection of carbon dioxide (CO 
2
​
 ), which acts as a primary long-range "beacon".[12] Humans exhale CO 
2
​
  at concentrations of approximately 4%, roughly two orders of magnitude higher than ambient atmospheric levels.[11] Specialized sensory structures known as capitate peg sensilla, located on the maxillary palps, house the cpA olfactory receptor neurons (ORNs) specifically tuned to detect CO 
2
​
 .[9, 11] These neurons are remarkably sensitive, capable of detecting concentration shifts as small as 0.01%.[9]
Upon encountering a CO 
2
​
  plume, female mosquitoes exhibit optomotor anemotaxis, a navigation strategy where the insect uses visual feedback to orient and move upwind.[10, 13] Because atmospheric airflow is inherently turbulent, odor plumes do not exist as smooth gradients but rather as intermittent "packets" of high-concentration signals.[9, 10] Mosquitoes track these plumes by surging upwind when a packet is detected and casting crosswind to re-establish contact when the signal is lost.[13] Beyond its role as a directional cue, CO 
2
​
  serves as a behavioral primer; its detection lowers the threshold for sensing skin odors and activates a strong attraction to high-contrast visual features.[7, 11, 13]
Intermediate Visual Guidance and Odor-Gated Behavior
As a mosquito moves within 5 to 15 meters of a potential host, visual cues begin to dominate the navigation sequence.[13] Despite having relatively low-resolution compound eyes, mosquitoes are highly sensitive to visual contrast and movement.[9, 14] A critical discovery in trajectory modeling is the concept of "odor-gated" vision, where the attraction to a visual object is explicitly triggered by the prior or simultaneous detection of CO 
2
​
 .[13]
Experimental evidence using 3D tracking has shown that mosquitoes are 2.5 times more likely to approach a CO 
2
​
  source if it is associated with a high-contrast visual feature.[9] Interestingly, this attraction persists even if the mosquito loses the CO 
2
​
  signal; some individuals have been observed taking circuitous paths for more than 10 seconds after leaving a plume to eventually reach a visual target.[13] This indicates that the sensory integration process is not a simple additive mechanism but a complex state transition where chemical signals unlock new behavioral responses to the environment.[9, 13]
Sensory Stimulus
Effective Distance
Biological Mechanism
Navigational Response
Carbon Dioxide (CO 
2
​
 )
> 10–50 m
Maxillary palp (cpA neurons)
Upwind surge and crosswind casting
High-Contrast Visuals
5–15 m
Compound eyes (Optic flow)
Object orientation and "orbiting"
Volatile Skin Odors
1–5 m
Antennal trichoid sensilla
Close-range attraction and landing
Thermal Infrared
< 1 m
Specialized thermal bristles
Targeted landing on exposed skin
Convection/Humidity
< 0.1 m
Antennal and labellar sensors
Probing site selection and feeding
[9, 11, 12, 13]
Short-Range Thermal and Humidity Feedback
The final stage of the trajectory involves the transition from visual guidance to thermal and humidity-based sensing at ranges of less than one meter.[9, 11] Human skin emits infrared radiation and generates convection currents of heat and moisture.[11] Mosquitoes detect these cues to distinguish between an inanimate visual target and a living host.[12] Thermal cues are highly effective for pinpointing areas of exposed skin, such as the ankles or neck, where the insect can successfully probe for capillaries.[7, 12] The integration of these short-range cues ensures that the mosquito does not waste energy attempting to feed on non-biological surfaces.[11]
Mathematical Foundations of Trajectory Modeling
Quantifying the movement of mosquitoes requires a robust mathematical framework that can account for both the stochastic nature of flight and the deterministic influence of sensory cues.[15, 16]
Correlated Random Walks and Persistence
The most fundamental models for insect movement are based on random walk processes.[15] However, unlike a simple Brownian motion where each step is completely independent, mosquito flight exhibits "persistence"—a correlation between the directions of successive steps.[15, 16] This is formally described as a Correlated Random Walk (CRW).[15, 17] In a one-dimensional system, the population density n(x,t) of individuals exhibiting persistence can be modeled using the telegraph equation:
∂t 
2
 
∂ 
2
 n
​
 +2λ 
∂t
∂n
​
 =v 
2
  
∂x 
2
 
∂ 
2
 n
​
 
[15]
where v represents the constant flight speed and λ is the turning frequency.[15] In more realistic 2D or 3D scenarios, the persistence is quantified by the distribution of turning angles. The mean squared displacement (MSD) for a CRW after n moves is given by the formula:
E=nE[l 
2
 ]+2E[l] 
2
  
1−c
c
​
 (n− 
1−c
1−c 
n
 
​
 )
[17]
where l is the move length and c is the mean cosine of the turning angle.[17] As c approaches 1 (smaller turning angles), the net displacement increases dramatically, reflecting the efficient forward-searching behavior of the insect.[15, 17]
Lévy Flights and Search Optimization
When resources are sparsely distributed, such as in the case of host-seeking in vast landscapes, mosquito flight may shift from a CRW to a Lévy flight.[15, 18, 19] Lévy flights are random walks where the step lengths are drawn from a heavy-tailed distribution, meaning that the variance of the step lengths is infinite.[18, 20] This allows for occasional very long jumps that enable the insect to escape localized search areas and encounter new resource patches.[16, 18]
The probability density for a Lévy flight can be described using a fractional diffusion equation:
∂t
∂f
​
 =γ 
∂∣x∣ 
α
 
∂ 
α
 f
​
 
[18]
where α is the stability parameter (0<α<2) and γ is the anomalous diffusion constant.[18] Processes with 1<α<2 exhibit super-diffusion, where the MSD scales as t 
μ
  with 1<μ<2.[15] This type of movement is often considered an optimal foraging strategy under conditions of uncertainty.[16, 18]
Biased Diffusion and Drift-Advection
In the presence of a directional attractant like a CO 
2
​
  plume, the random walk becomes "biased".[15, 21] This is modeled as a Biased Random Walk (BRW), which at a population scale converges to the drift-diffusion equation:
∂t
∂p
​
 +u 
∂x
∂p
​
 =D 
∂x 
2
 
∂ 
2
 p
​
 
[15, 22]
where u is the drift velocity toward the attractant and D is the diffusion coefficient.[15, 21] For large time scales, the displacement of the population is dominated by the drift term, resulting in "ballistic" movement toward the source.[15]
High-Fidelity 3D Trajectory Reconstruction
Advancements in computer vision and high-speed imaging have made it possible to record and reconstruct mosquito flight paths in three dimensions with sub-millimeter precision.[5, 10, 23]
The Photonic Fence and Multi-Camera Systems
Modern experimental setups often utilize "photonic fence" monitoring devices (PFMD), which consist of dual infrared cameras surrounded by LED arrays.[5] These systems capture stereoscopic images of insects in space at temporal resolutions as high as 100 Hz (0.01-s time resolution).[5] In large-scale experiments, such as those conducted at Georgia Tech, mosquitoes were tracked within mesh enclosures of significant depth (up to 8 meters), generating tens of millions of data points.[5, 24]
The technical challenge of reconstructing these trajectories lies in the "sub-pixel" imaging characteristics of small insects like mosquitoes.[23] Traditional convolutional neural networks (CNNs) often struggle with feature extraction in complex, cluttered backgrounds.[23] Consequently, researchers employ hybrid approaches that combine background subtraction with Kalman filtering to predict target states and the Hungarian algorithm for multi-target data association.[23] These systems can achieve 3D reconstruction with a mean error of only 10±4 mm and detection accuracies exceeding 95%.[23]
Machine Learning and Automated Behavior Analysis
Tools like FlightTrackAI have been developed to automate the analysis of these massive datasets.[25] By utilizing CNNs for robust identity tracking even during occlusions (with success rates over 91%), these tools can automatically calculate flight parameters such as distance, speed, and volume coverage.[25] This level of automation is crucial for comparative studies—for example, comparing the flight fitness of wild-type mosquitoes versus those carrying genetic modifications for population suppression.[25]
Tracking Metric
Typical Value (Laboratory)
Significance
Temporal Resolution
60–100+ fps
Capturing rapid maneuvers and wingbeat patterns.
Spatial Precision
±10 mm
Localizing landing sites and probing behavior.
Data Density
53M+ points
Enabling Bayesian dynamical modeling.
Identification Accuracy
> 99%
Tracking individual behavior in swarms.
[5, 23, 25]
Data-Driven Modeling with Bayesian Dynamical Systems
A significant breakthrough in the field was the 2026 report of the first comprehensive 3D model of mosquito flight based on over 53 million data points.[5, 8, 24] Researchers at MIT and Georgia Tech used a Bayesian machine learning technique to derive a parsimonious mathematical model that accurately predicts flight paths in response to specific cues.[2, 5]
The Non-Additive Principle: 1+1

=2
Perhaps the most startling insight from this research is that the resulting flight path when both visual and chemical cues are present is not simply the sum of the paths taken for each cue individually.[2, 8] In other words, the mosquito's multi-sensory integration is non-additive.[8]
Visual Only ("Fly-by"): When mosquitoes only detect a visual silhouette (e.g., a black sphere), they exhibit a "fly-by" approach, quickly diving toward the target and then exiting if no other cues are detected.[8]
CO 
2
​
  Only ("Double-take"): In the presence of CO 
2
​
  without visual cues, mosquitoes exhibit a "double-take" pattern, slowing down and flitting back and forth to maintain proximity to the odor source.[8]
Combined Cues ("Orbiting"): When both cues are present, the behavior shifts to a distinct "orbiting" pattern.[8] The mosquito flies around the target at a steady speed, much like a predator circling its prey, as it prepares to land.[8]
This orbiting behavior is critical for the design of multisensory lures for mosquito traps. If a trap provides only a single cue, the mosquito is unlikely to remain engaged long enough to be captured by suction or adhesive mechanisms.[8, 26]
Bayesian Iteration and Model Parsimony
The model was developed by proposing a broad range of dynamical equations and then iteratively reducing their complexity against the tracking data.[8] This process ensured that the final model was the simplest possible representation that still agreed with the biological observations.[8] Such a model not only reproduces the stereotypical swarming seen around a human head and shoulders but also allows for "in silico" testing of how different trap geometries or cue release rates might impact capture efficiency.[24, 26]
Biomechanics and Aerodynamics of Flapping Flight
While trajectory modeling often focuses on the "macro" scale of movement, the "micro" scale of wing kinematics provides the physical basis for every maneuver.[27, 28]
High-Frequency Wingbeats and Lift Mechanisms
Mosquitoes are unique among insects of their size for their extremely high wingbeat frequencies.[27, 29] Their long, slender wings do not rely solely on the standard leading-edge vortex (LEV) found in larger insects like moths or bees.[6, 27] Instead, mosquitoes have evolved a suite of specialized aerodynamic mechanisms:
Rotational Drag: Forces generated by the rotation of the wing at the transition between strokes.[27]
Wake Capture: Reinforcing the trailing-edge vortex by interacting with the wake of the previous wingbeat.[27]
Added Mass Effect: The inertia of the air mass accelerated by the wing, which becomes highly significant at high frequencies.[27, 30]
Computational fluid dynamics (CFD) simulations using body-fitted meshes or immersed boundary methods (IBM) have shown that these mechanisms are essential for producing the lift necessary to stay aloft, especially when the mosquito is carrying a blood meal that can double its body weight.[30, 31, 32, 33]
Stability and the Role of Non-Wing Appendages
Flight stability is maintained through the coordinated movement of the wings, legs, and halteres.[27, 34] Halteres act as gyroscopic sensors, oscillating at high frequencies to detect angular rates across three axes.[27] Furthermore, the legs of the mosquito have been shown to play a surprisingly large role in roll stability.[34] When a mosquito is attacked or disturbed, it can perform incredibly rapid escape maneuvers, including upside-down or backwards flight, by subtly adjusting the kinematics of its wings and the orientation of its legs.[34]
Component
Function in Flight
Technical/Biological Mechanism
Wings
Lift and Thrust Generation
LEVs, added mass, and wake capture [27, 30]
Halteres
Gyroscopic Stabilization
High-frequency sensory oscillations [27]
Legs
Roll Stability and Damping
Inertial and aerodynamic stabilization [34]
Wing Hinge
Maneuverability Control
12-muscle "puppeteer" mechanism [35]
The Wing Hinge: A Masterpiece of Biological Engineering
The wing hinge of the mosquito (and other Dipterans like Drosophila) is considered one of the most complex structures in the history of life.[35] It contains 12 control muscles, each innervated by a single neuron.[35] Researchers at Caltech have used high-speed cameras and machine learning to map how these muscles act together to precisely regulate wing motion.[35] This "neural-biomechanical interface" allows the insect to perform agile aerodynamic maneuvers that engineered micro-aerial vehicles (MAVs) still struggle to replicate.[28, 30, 35]
Environmental and Microclimatic Modulation of Flight
Mosquito trajectories are not generated in a vacuum; they are constantly modified by the physical and chemical state of the surrounding atmosphere.[10, 36]
Turbulence and Performance
The impact of turbulence on host-seeking behavior is a burgeoning area of study.[37] Experiments in wind tunnels have compared mosquito performance in low (5%) versus high (20%) turbulence intensities.[37] High turbulence tends to disrupt the structure of odor plumes, making them harder to follow, but it may also provide the mosquito with additional "upwind" cues through the detection of fluctuating pressure gradients.[10, 37] Understanding how natural turbulence affects capture rates is essential for optimizing the placement of odor-baited traps in field conditions.[10, 37]
Temperature, Humidity, and Desiccation Risk
Mosquitoes are ectotherms, meaning their metabolic rates and flight performance are strictly bound by ambient temperature.[36, 38] Flight activity increases from a minimum critical temperature (CT 
min
​
 ) up to an optimum (T 
opt
​
 ), followed by a sharp decline toward the thermal maximum (CT 
max
​
 ).[38]
Humidity, however, is often the more critical factor for survival and trajectory choice.[36, 38] In environments with high vapor pressure deficits (VPD), mosquitoes are at risk of lethal desiccation.[39, 40] Consequently, they often adjust their vertical flight height to remain in more humid microclimates, typically staying closer to the ground or within dense vegetation.[39, 40] Interestingly, research has shown that strong visual cues (like a potential host) can "override" these homeostatic constraints, causing the mosquito to venture into unfavorable microclimates to seek a blood meal.[39, 40]
Variable
Influence on Trajectory
Thresholds/Observations
Temperature
Metabolic rate and wingbeat frequency
Wingbeat freq increases with temp until plateau [34]
Humidity (VPD)
Vertical distribution and desiccation risk
Mosquitoes lower flight height to minimize VPD [39, 40]
Wind Speed
Long-range dispersal vs. station-keeping
Wind can facilitate migration at altitude (120-290m) [4]
Light Intensity
Escape performance and circadian rhythm
Aedes escape faster in light; Anopheles in dark [41]
Multi-Scale Modeling: From Individual Bouts to Populations
The integration of fine-grained flight trajectories into population-level models is the key to predicting disease outbreaks and the impact of interventions.[1, 3, 42]
Agent-Based Models (ABM) and Microsimulation
Agent-based models (ABMs) represent individual mosquitoes as "agents" that interact with a heterogeneous landscape.[1, 43] These models explicitly track the behavioral state of each female—whether she is searching for a host, resting to digest blood, or seeking an aquatic habitat for oviposition.[19, 43] The landscape is modeled as a set of "resource point sets" (houses, habitats, and sugar sources), and movement among these points is governed by dispersal kernels.[42, 44]
Frameworks such as ramp.micro and Micro-MoB allow researchers to simulate how the spatial arrangement of these resources affects population dynamics.[42, 45] For example, simulations have shown that even with a uniform distribution of resources, mosquito populations tend to form highly structured communities.[19, 44] Areas that lack a specific resource (like water for eggs) become "sinks" as mosquitoes exit to find the missing resource, while areas with all required resources become "sources".[19, 44]
AnophelesModel and Vector Control Quantification
The AnophelesModel R package provides an interface between mosquito bionomics and vector control intervention analysis.[46, 47] It uses discrete-time state transition models to infer how interventions like insecticide-treated nets (ITNs) or indoor residual spraying (IRS) affect the mosquito feeding cycle.[46] By scaling the deterrency and killing effects of these tools based on setting-specific exposure coefficients (indoor vs. outdoor biting rhythms), the package can estimate the resulting reduction in vectorial capacity.[46, 47]
High-Altitude Migration and Long-Distance Trajectories
While much research focuses on the "near-field" behavior around humans, a substantial portion of the mosquito-borne disease burden may be driven by long-distance, high-altitude trajectories.[4]
The African Aerial Vector Network
Invasive and sylvatic pathogens are often spread across vast distances by mosquitoes caught in high-altitude wind currents.[4] Sampling at 120 to 290 meters above ground level has revealed that gravid, blood-fed mosquitoes regularly engage in migratory flights.[4] These individuals are frequently infected with arboviruses, protozoans (Plasmodium), and filarial worms.[4]
Prevalence: Studies in Mali and Ghana found infection rates of 7.2% for Plasmodium spp. in high-flying migrants.[4]
Infectivity: Approximately 4.4% of these high-flying mosquitoes had disseminated infections, meaning they could potentially transmit the pathogen immediately upon landing in a new territory.[4]
This suggests that trajectory modeling must account for "jump" events where wind-borne transport allows mosquitoes to bypass traditional terrestrial barriers and spread resistance alleles or pathogens hundreds of kilometers from their origin.[4]
Optimization of Surveillance and Control Strategies
The final application of trajectory and movement models is the mathematical optimization of mosquito control.[48, 49]
MGSurvE and Trap Network Design
The MGSurvE (Mosquito Gene SurveillancE) framework was developed to optimize trap placement for detecting specific genetic traits, such as gene-drive alleles or insecticide resistance.[49, 50] This framework uses biological features of the landscape and mosquito movement kernels to minimize the "time to detection".[49, 51] By placing traps at strategic nodes in the metapopulation—informed by high-fidelity flight models—researchers can ensure that emerging threats are identified while remediation is still viable.[49, 50]
Bayesian Optimization for Pest Monitoring
Similar techniques are being used in agricultural pest management.[48, 52] Bayesian optimization (BO) algorithms can identify the most informative trap locations under resource constraints.[48] By approximating the complex, non-linear relationship between trap placement and population density estimation with a surrogate model, BO identifies optimal distributions that improve monitoring precision and efficiency.[48, 52]
Conclusion
The modeling of mosquito flight trajectories has entered a new era of multi-scale integration. By combining the fundamental physics of flapping wings with the behavioral complexity of multi-sensory navigation and the spatial dynamics of population dispersal, researchers have developed a comprehensive framework for understanding one of nature's most successful—and deadly—predators. The discovery of non-additive sensory integration and the quantification of high-altitude migratory networks represent major shifts in our understanding of vector ecology. Moving forward, the refinement of these models will be instrumental in the "in silico" evaluation of novel bed nets, the optimization of genetic surveillance networks, and the eventual eradication of mosquito-borne diseases. The transition from simplistic diffusion models to high-fidelity, data-driven dynamical systems marks a paradigm shift that will define vector control strategy for decades to come.
--------------------------------------------------------------------------------
Agent-Based Modeling and Simulation of Mosquito-Borne Disease Transmission - IFAAMAS, https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p426.pdf
Predicting Mosquito Flight Behavior - Alexander E. Cohen, https://acoh64.github.io/mosquito_app/
A minimal 3D model of mosquito flight behaviour around the human baited bed net - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC7792054/
Pathogens spread by high-flying wind-borne mosquitoes - PNAS, https://www.pnas.org/doi/10.1073/pnas.2513739122
Predicting mosquito flight behavior using Bayesian dynamical systems learning - arXiv, https://arxiv.org/html/2505.13615v1
Computational aerodynamics of insect flight using volume penalization - Comptes Rendus de l'Académie des Sciences, https://comptes-rendus.academie-sciences.fr/mecanique/articles/10.5802/crmeca.129/
Predicting mosquito flight behavior using Bayesian dynamical systems learning - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12998517/
New model predicts how mosquitoes will fly | MIT News, https://news.mit.edu/2026/new-model-predicts-how-mosquitoes-will-fly-0318
BSAN: Behavioral State Attention Network for Modeling Mosquito Host-Seeking Behavior - AAAI Publications, https://ojs.aaai.org/index.php/AAAI/article/view/41280/45241
Experiments and Analysis of Mosquito Flight Behaviors in a Wind Tunnel: An Introduction, https://pmc.ncbi.nlm.nih.gov/articles/PMC12810408/
The sensory arsenal mosquitoes use to find us - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12352024/
How Mosquitoes Track You Using Heat and CO2 - Specter Pest Control, https://specterservice.com/how-mosquitoes-use-heat-and-carbon-dioxide-to-find-you/
Mosquitoes use vision to associate odor plumes with thermal targets ..., https://pmc.ncbi.nlm.nih.gov/articles/PMC4546539/
Virtual Reality Experiments Reveal that CO 2 Sharpens Mosquitoes' Senses - Kao Americas, https://www.kao.com/global/en/newsroom/news/release/2025/20250820-001/
Random walk models in biology - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC2504494/
Constructing a Stochastic Model of Bumblebee Flights from Experimental Data - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC3592844/
Analyzing Insect Movement as a Correlated Random-Walk - ResearchGate, https://www.researchgate.net/publication/225232354_Analyzing_Insect_Movement_as_a_Correlated_Random-Walk
Lévy flight - Wikipedia, https://en.wikipedia.org/wiki/L%C3%A9vy_flight
Mosquito Dispersal in Context - bioRxiv, https://www.biorxiv.org/content/biorxiv/early/2025/03/25/2025.03.22.642900.full.pdf
Lecture 12: Levy Flights (σ=∞), https://ocw.mit.edu/courses/18-366-random-walks-and-diffusion-fall-2006/f3723b5ae3c92a8840f132fc5ebee940_lecture12.pdf
(PDF) A diffusion model for mosquito control trials with spillover: application to calculations of power, sample size, and bias - ResearchGate, https://www.researchgate.net/publication/402817051_A_diffusion_model_for_mosquito_control_trials_with_spillover_application_to_calculations_of_power_sample_size_and_bias
(PDF) A diffusion model for mosquito control trials with spillover - ResearchGate, https://www.researchgate.net/publication/389720896_A_diffusion_model_for_mosquito_control_trials_with_spillover
Insights into Mosquito Behavior: Employing Visual Technology to Analyze Flight Trajectories and Patterns - MDPI, https://www.mdpi.com/2079-9292/14/7/1333
MIT, Georgia Tech Build First 3D Model of Mosquito Flight - Ground News, https://ground.news/article/mit-georgia-tech-build-first-3d-model-of-mosquito-flight_2823a7
FlightTrackAI: a robust convolutional neural network-based tool for tracking the flight behaviour of Aedes aegypti mosquitoes - Royal Society Publishing, https://royalsocietypublishing.org/rsos/article/11/10/240923/92279/FlightTrackAI-a-robust-convolutional-neural
Why Mosquitoes Swarm Your Head: They're Following Signals, Not Each Other - Georgia Tech College of Engineering, https://coe.gatech.edu/news/2026/03/why-mosquitoes-swarm-your-head-theyre-following-signals-not-each-other
Study of Mosquito Aerodynamics for Imitation as a Small Robot and Flight in a Low-Density Environment - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC8147425/
Modeling of Insect Flight Mechanics, https://www.me.washington.edu/research/faculty/ishen/Modeling-of-Insect-Flight-Mechanics
Making sense of mosquitoes with machine learning - WUR, https://www.wur.nl/en/longread/making-sense-mosquitoes-machine-learning
Characterisation and extension of a rigid body dynamics solver coupled with OpenFOAM for flight performance analysis of flapping-wing drones - arXiv, https://arxiv.org/html/2510.24518v1
Aerodynamics of insect flight - Wageningen University & Research, https://research.wur.nl/en/publications/aerodynamics-of-insect-flight/
Escaping blood-fed malaria mosquitoes minimize tactile detection without compromising on take-off speed - Company of Biologists journals, https://journals.biologists.com/jeb/article/220/20/3751/18745/Escaping-blood-fed-malaria-mosquitoes-minimize
Immersed Boundary Method in OpenFOAM: Numerical Validation and Applications to Wheel Geometries - POLITesi - Politecnico di Milano, https://www.politesi.polimi.it/retrieve/53d26a14-c991-469b-99bf-21030f88e9be/Tesi-4.pdf
Biomechanics of the Insect Flight Motor - White Rose eTheses Online, https://etheses.whiterose.ac.uk/id/eprint/35411/1/Tran_RHST_BiomedicalSciences_PhD_2024.pdf
How Insects Control Their Wings: The Mysterious Mechanics of Insect Flight, https://www.bbe.caltech.edu/news/how-insects-control-their-wings-the-mysterious-mechanics-of-insect-flight
The Impact of Climatic Factors on Temporal Mosquito Distribution and Population Dynamics in an Area Targeted for Sterile Insect Technique Pilot Trials - MDPI, https://www.mdpi.com/1660-4601/21/5/558
75th Annual Meeting of the Division of Fluid Dynamics - Event - Mosquito flight in turbulent airflow, https://meetings.aps.org/Meeting/DFD22/Session/L04.3
Humidity – The overlooked variable in the thermal biology of mosquito-borne disease - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC10299817/
Drivers of mosquito free-flight and resting behavior indoors | bioRxiv, https://www.biorxiv.org/content/10.64898/2026.02.11.705377v1.full-text
Drivers of mosquito free-flight and resting behavior indoors - bioRxiv, https://www.biorxiv.org/content/10.64898/2026.02.11.705377v1
Biomechanics of flying mosquitoes during capture and escape - Research@WUR, https://research.wur.nl/en/publications/biomechanics-of-flying-mosquitoes-during-capture-and-escape/
Microsimulation for Mosquito Ecology • ramp.micro, https://dd-harp.github.io/ramp.micro/
Agent-based modelling of mosquito foraging behaviour for malaria control - PMC - NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC2818421/
Mosquito Dispersal in Context - PMC - NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC11974703/
Micro-MoB (Microsimulation for mosquito-borne pathogens) - GitHub, https://github.com/dd-harp/MicroMoB
AnophelesModel: An R package to interface mosquito bionomics, human exposure and intervention effects with models of malaria intervention impact | PLOS Computational Biology, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011609
AnophelesModel: An R package to interface mosquito bionomics, human exposure and intervention effects with models of malaria intervention impact - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11424000/
Bayesian Optimization of insect trap distribution for pest monitoring efficiency in agroecosystems - Frontiers, https://www.frontiersin.org/journals/insect-science/articles/10.3389/finsc.2024.1509942/full
MGSurvE: A framework to optimize trap placement for genetic surveillance of mosquito populations | PLOS Computational Biology - Research journals, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012046
MGSurvE: A framework to optimize trap placement for genetic surveillance of mosquito population - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC10327167/
MGSurvE: A framework to optimize trap placement for genetic surveillance of mosquito populations - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11098508/
Bayesian Optimization of insect trap distribution for pest monitoring efficiency in agroecosystems - ResearchGate, https://www.researchgate.net/publication/388312662_Bayesian_Optimization_of_insect_trap_distribution_for_pest_monitoring_efficiency_in_agroecosystems
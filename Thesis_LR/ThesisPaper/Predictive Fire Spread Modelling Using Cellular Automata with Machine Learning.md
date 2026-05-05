### PREDICTIVE FIRE SPREAD MODELING USING CELLULAR

### AUTOMATA WITH MACHINE LEARNING IN URBAN LAYOUTS

```
A Thesis Proposal
Presented to the Faculty of the
Department of Computer, Information Sciences and Mathematics
University of San Carlos
```
```
In Partial Fulfillment of
the Requirements for the Degree
BACHELOR OF SCIENCE IN COMPUTER SCIENCE
```
```
Submitted By:
KRISTIAN LEMUEL W. DIAZ
KENT ANTHONY C. DULANGON
```
```
ANGIE M. CENIZA-CANILLO, PhD
Faculty Adviser
```
```
May 2025
```

## ABSTRACT.................................................................................................

Urban fires are a growing hazard in rapidly urbanizing cities, yet
existing predictive tools often fail to account for complex urban layouts and
variable environmental factors. This research proposes a hybrid cellular
automata with a machine learning framework for forecasting fire spread in
urban cities, using Lapu-Lapu City as a case study. The methodology
transforms high-resolution inputs like building footprints and materials,
terrain slope, vegetation fuel loads, and dynamic weather into a richly
featured Cellular Automata (CA) grid. Random Forest and Logistic
Regression models will be trained to estimate per-cell ignition probabilities
based on combustibility, wind alignment, slope, and neighborhood context,
replacing static Cellular Automata (CA) transition rules. The proposed
framework will be validated against historical fire events to assess
predictive accuracy and spatial correlation metrics. This approach aims to
provide fire agencies and urban planners with data-driven forecasts to
optimize resource deployment and inform zoning decisions. While the
methodology will be calibrated for Cebu and excludes suppression tactics,
it establishes a scalable foundation for multi-hazard urban risk modeling
that can be adapted to other rapidly urbanizing cities facing similar fire
management challenges.

```
ii
```

## TABLE OF CONTENTS..............................................................................

## TABLE OF CONTENTS

### APPENDICES

APPENDIX A Transmittal Letter
APPENDIX B Software Requirements Specifications
APPENDIX C Website UI
CURRICULUM VITAE


## LIST OF FIGURE........................................................................................

- ABSTRACT.................................................................................................
- TABLE OF CONTENTS..............................................................................
- LIST OF FIGURE........................................................................................
- LIST OF TABLES........................................................................................
- CHAPTER
   - 1.1 Rationale of the Study.....................................................................
   - 1.2 Statement of the Problem................................................................
      - 1.2.1 General Objective....................................................................
      - 1.2.2 Specific Objectives..................................................................
      - 1.3 Significance of the Study............................................................
      - 1.4 Scope and Limitations................................................................
- CHAPTER
- CHAPTER
- DESIGN AND METHODOLOGY..............................................................
   - 4.1 Research Environment..................................................................
   - 4.2 Research Instrument or Sources of Data......................................
   - 4.3 Research Procedure......................................................................
   - 4.5 Analysis and Design......................................................................
   - 4.6 Development Model.......................................................................
   - 4.7 Development Approaches.............................................................
   - 4.8 Software Development Tools.........................................................
   - 4.9 Project Management......................................................................
   - 4.10 Verification, Validation, and Testing.............................................
- BIBLIOGRAPHY.......................................................................................
- Figure 1. Urban Layout Grid................................................................ LIST OF FIGURE
- Figure 2. Urban Layout Grid (3x3 cells)...............................................
- Figure 3: Fire Spread Simulation Flow Chart.......................................
- Figure 4. Cebu City Index Map Scale 1:50,000 scale..........................
- Figure 5. Effects of Western Wind (Gao et al., 2008)..........................
- Figure 6. Conceptual Framework........................................................
- Figure 7. DataFlow Diagram (Level 0).................................................
- Figure 8. Incremental Development Model..........................................
- Figure 9. Bottom-Up Approach of CA-ML Fire System........................


## LIST OF TABLES........................................................................................

Table 1 Summary of Related Literature..................................................... 18
Table 2 Software Development Tools........................................................ 42
Table 3 Gantt Chart for Activities............................................................... 44
Table 4 Roles and Responsibilities........................................................... 45
Table 5 Proposed Budget and Cost.......................................................... 46

```
v
```

## CHAPTER

### INTRODUCTION

This chapter of the report provides the background in which the
study was conducted and outlines the issues that motivated the
investigation. This part also explains the goal of attempting to solve such a
challenge, the significance of the work done, and the results obtained as a
result of the study and investigation.

**1.1 Rationale of the Study**

Urban fires represent a critical challenge in densely populated
cities, where they threaten lives, destroy property, and disrupt economies.
In the Philippines, the escalating frequency of fire incidents underscores
the urgency of this issue. The Bureau of Fire Protection reported a 21.1%
increase in fire incidents in 2023, totaling 15,900 cases (Cariaso, 2024),
followed by a further rise to 18,256 in 2024, marking an 11.2% increase.
These fires, often occurring in residential areas with flammable materials
and limited firefighting access, result in significant economic losses,
estimated at nearly 14 billion pesos in 2024, and tragic human tolls, with
341 civilian fatalities recorded in the same year (Chavez, 2025). The rapid
urbanization of cities like Cebu, characterized by dense settlements and
complex infrastructure, amplifies these risks, making effective fire
management a pressing priority.
The layout of urban areas significantly influences fire‑spread
dynamics. Different street network patterns whether uniform grid, winding
organic blocks, or planned subdivisions affect how quickly and in what
directions fire propagates. Grid-like streets can channel flames along
straight corridors, while more irregular layouts may slow or divert the fire
front depending on building density, street width, and orientation relative to
prevailing winds (Ohgai et al., 2004). Understanding how these layouts
affect fire spread is essential for developing targeted prevention and


response strategies, yet current modeling approaches often fail to account
for the diverse geometries found in rapidly developing Philippine cities.
Previous research has explored fire spread modeling through
various approaches, with studies utilizing cellular automata for wildfire
simulation and others applying machine learning techniques to predict fire
behavior in urban environments. However, existing models often focus on
specific urban geometries or lack the integration of advanced machine
learning with cellular automata for diverse urban layouts. The gap in
comprehensive predictive models that can adapt to various urban street
patterns while incorporating real-time environmental data remains largely
unaddressed in current literature, particularly in the context of Southeast
Asian urban environments.
This thesis addresses these gaps by developing a predictive model
that simulates fire spread across urban layouts of differing geometries,
integrating cellular automata with machine learning techniques. The model
will utilize datasets on urban layouts, historical fire incidents, and
environmental conditions, such as those available from the Bureau of Fire
Protection and OpenStreetMap, to ensure relevance to Philippine cities
like Lapu-Lapu. By employing Random Forest and Logistic Regression,
the study aims to enhance the accuracy of fire spread predictions, offering
a tool that can inform urban planning and fire safety strategies.This study
contributes to the growing body of knowledge in urban disaster
management while developing practical solutions for fire-prone Philippine
cities, recognizing the urgent need for locally-relevant predictive tools that
can save lives and reduce economic losses.

**1.2 Statement of the Problem**

```
1.2.1 General Objective
This research aims to design and develop a predictive model
for fire spread in urban layouts using cellular automata integrated
```

```
with machine learning techniques, specifically Random Forest and
Logistic Regression, to enhance fire safety and disaster
management in urban areas.
```
```
1.2.2 Specific Objectives
This study specifically aims the following:
```
1. Gather and preprocess geospatial data for the study
    area, including urban layout, building characteristics,
    and topographical data from OpenStreetMap and
    NAMRIA.
2. Collect historical fire incident data from the Bureau of
    Fire Protection and weather data from
    OpenWeatherMap relevant to fire spread.
3. Design and develop a cellular automata model for fire
    spread simulation, considering factors such as
    building materials, wind direction, and topography.
4. Train Random Forest and Logistic Regression models
    to predict ignition probabilities based on the collected
    data and integrate it into the cellular automata model.
5. Validate the performance of the model using historical
    fire data based on precision, accuracy, f-measure.

**1.3 Significance of the Study**

The outcome of this study may be relevant and instrumental to the
following parties:
**Fire Agencies and First Responders.** The predictive model
provides fire departments with enhanced capabilities for proactive fire
management and strategic resource deployment. By delivering accurate,
layout-aware forecasts of fire spread patterns, the system enables
informed dispatch strategies and optimal pre-positioning of firefighting
resources. The Bureau of Fire Protection can utilize simulation outputs to


identify high-risk areas before incidents occur, potentially reducing
response times and minimizing casualties. Fire chiefs can leverage
scenario-based predictions to develop targeted intervention strategies
tailored to specific urban configurations.

**Urban Planners and Local Government.** The research provides
city planners with scientific insights into how different urban layouts
influence fire propagation dynamics. This knowledge can guide
evidence-based zoning regulations, building code revisions, and street
network redesigns that prioritize fire safety. City engineers can test
hypothetical infrastructure modifications through simulation before
committing to costly physical implementations. The model supports
informed decision-making in urban development projects, ensuring fire
safety considerations are integrated into planning processes.

**Property Owners.** Neighborhood associations and residents gain
practical value from interactive simulations that show how fire might
spread under different conditions in their neighborhoods. By producing
clear, localized risk maps, the system highlights vulnerable buildings and
suggests safe evacuation routes. Communities can plan targeted fire drills
and mitigation measures tailored to their specific layouts. Property owners
can make informed choices about fire-resistant modifications based on
quantified, location-specific risk assessments.

**Academia and Disaster Modeling Researchers.** This work
introduces a flexible methodology for urban fire modeling that combines
cellular automata with machine learning classifiers (Logistic Regression
and Random Forest) across diverse urban street patterns. It addresses a
notable gap in the literature, where most models focus on wildland fires or
uniform terrains. By demonstrating how ignition probabilities can be
learned from urban data and integrated into a CA framework, the study
advances computational fire-spread modeling. The open-source codebase


and validation framework will encourage reproducibility and provide a
foundation for future extensions.

**Researchers.** The research team will gain valuable experience in
interdisciplinary methodology, combining computer science, urban
planning, and disaster management approaches. Through model
development and validation, researchers will acquire practical skills in
simulation techniques, geospatial analysis, and predictive modeling.

**Future Researchers.** The study establishes a comprehensive
framework that opens avenues for further research in fire spread
modeling, urban disaster simulation, and machine learning applications in
emergency management. Future researchers can build upon this
foundation to explore advanced algorithms, expand to different urban
contexts, or integrate additional environmental factors..

**1.4 Scope and Limitations**

The study develops a predictive fire‑spread model specifically
designed for urban layouts, addressing the critical need for accurate fire
behavior forecasting in complex urban environments. The research
develops a predictive model that combines cellular automata simulation
with machine learning algorithms to forecast fire behavior in complex
urban environments. The system is designed to support fire prevention,
emergency response planning, and urban safety management through
data-driven predictions and scenario analysis.
The research focuses specifically on urban areas exhibiting diverse
urban layouts, with Lapu-Lapu City serving as the primary case study. The
model integrates multiple data sources including building footprints and
road networks from OpenStreetMap, terrain data from NAMRIA, historical
fire incident records from the Bureau of Fire Protection, and
meteorological data from OpenWeatherMap. Other datatypes such as
vegetation/fuel load, elevation, and building characteristics were used


throughout the study. The modeling approach employs Logistic
Regression and Random Forest algorithms to estimate per-cell ignition
probabilities, which are then integrated into a cellular automata framework
operating on a 2D grid with 3×3 meter resolution and discrete time steps.
The system evaluates fire spread scenarios under varied wind
conditions and different urban layout configurations, specifically comparing
various urban street patterns and neighborhood geometries. Model
validation employs multiple metrics including confusion matrix analysis
targeting an F1 score of 0.80 or higher, Jaccard index for spatial overlap
assessment, and burned area percentage agreement calculations. The
dataset utilized for training and testing encompasses historical fire
incidents specific to the study area, with weather data spanning multiple
years to capture seasonal variations and diverse environmental
conditions.
However, the study acknowledges several limitations that affect the
model's scope and applicability. The system relies on post-event fire
perimeter data and publicly available weather records, which may lack the
granularity or real-time accuracy of dedicated sensor networks. Any
inaccuracies in input GIS layers, such as outdated building footprints or
incomplete road network data, will propagate into simulation results. The
model incorporates several simplifications that may affect prediction
accuracy: firefighting interventions such as hose lines and aerial water
drops are not represented, potentially overestimating fire spread in
scenarios where suppression efforts would be effective. Additionally,
long-distance ember transport and secondary ignitions are excluded from
the model, which may result in underestimating fire spread during
high-wind conditions.
The model's generalizability is constrained by its training on
Lapu-Lapu’s specific urban layout and incident history, requiring potential
retraining or parameter adjustment for application to cities with different
construction materials, climate regimes, or urban development patterns.


Computational constraints limit the practical application scope, as the 3×
meter grid resolution, while balancing detail and runtime requirements,
may demand significant computing time and memory resources for very
large study areas. The research scope is further limited to two machine
learning algorithms with more advanced architectures such as CNN or
GNN reserved for future investigation.


## CHAPTER

### REVIEW OF RELATED LITERATURE

This chapter provides a critical synthesis of prior research relevant
to fire spread. Additionally, this component of the study assesses and
looks at how previous research has applied algorithms for fire spread
prediction and machine learning for fire spread modeling, as well as how
the findings of prior studies relate to the completion of the current study.

**Fire Spread**
Urban fire spread has been studied through a variety of modeling
approaches, with traditional empirical and physical models originally
developed for wildland fires being adapted or applied to urban contexts.
Traditional models, such as Rothermel's (1972) wildfire spread model,
have been adapted for urban contexts but often fail to account for the
complexities of diverse urban layouts. Early studies on the spread of
urban fires have mostly relied on physics-based models, providing
foundational understanding of fire behavior in built environments.
Himoto and Tanaka (2003) created a comprehensive model that
provides insights into fire dynamics by simulating the spread of fire
through heat transmission and combustion processes. Their approach
demonstrated how factors such as building materials, urban layout, wind
patterns, topography, and firefighting efforts influence fire propagation in
urban settings. Building on this foundation, Himoto and Tanaka (2008)
developed a more sophisticated physics-based urban fire spread model
that interpreted urban flames as collections of separate building fires
affected by firebrand spotting, wind-blown fire plumes, and thermal
radiation. Their model highlighted the importance of building-to-building
ignition in crowded metropolitan environments and was validated using
historical fire data from Japan.
Similarly, Vakalis et al. (2004) emphasized the influence of urban
layout on fire behavior by simulating fire propagation using a GIS-based


system. These foundational studies established that while some urban
configurations help predict fire spread patterns, others can result in more
chaotic propagation behaviors, demonstrating the necessity for models
that account for this variability in urban form and structure.
**Urban Layouts**
Urban layouts have a significant impact on fire dynamics,
particularly in how fire spreads through different combinations of roadways
and building configurations. The diverse nature of urban environments,
which can combine various street patterns with different building densities
and arrangements, creates unique challenges for fire spread modeling. He
and Weng (2025) point out that compact building configurations might
increase the risk of multiple hazards, suggesting that denser urban forms
can exacerbate fire spread patterns.
Different street network configurations produce distinct connectivity
patterns that affect fire propagation. Grid-like street networks typically yield
regular block patterns with predictable pathways, while radial or ring-road
layouts produce star-shaped connectivity with different flow
characteristics. Many urban areas combine these traits, creating complex
environments where major arterials may be interspersed with local grid
patterns, affecting how flames may leap across and around streets.
Recent studies confirm these layout effects on fire behavior.
Purnomo et al. (2024) found that layout configurations play a pivotal role in
fire spread, noting that "isolated islands of combustibles" significantly slow
fire progression. Conversely, fire can spread freely across urban fabric
when there are continuous fuel loads, such as densely packed wooden
structures. In informal settlements particularly, close-packed, asymmetrical
layouts with minimal spacing can function as a "chain of combustible
materials," enabling rapid fire propagation.
Urban planning literature provides additional context for
understanding diverse layout effects on fire behavior. Radial city plans, as
described by ArchDaily (2020), emphasize central elements like plazas or


government buildings, with streets extending outward and concentric
roads connecting them. In contrast, Urban Grid: Architecture & Design
Principles notes that urban grids represent a city planning model where
streets are arranged in a checkerboard pattern, making navigation
straightforward and efficient but creating different fire spread dynamics.

Shaham and Benenson (2018) further investigated urban fire
propagation in Mediterranean and Middle Eastern cities, finding that
flammable urban vegetation and building structures had major impacts on
fire transmission between buildings. Their research demonstrated that
various urban forms alter fire risk in different ways: dense building blocks
or closely spaced structures can intensify burning, while wide arterial
roads or open areas might serve as effective firebreaks.

The complexity of modern urban environments, which often feature
combinations of different planning approaches and organic development
patterns, requires sophisticated modeling approaches to capture their
unique fire dynamics. Street patterns and building layouts determine the
pathways available to spreading fire and are therefore critical factors in
any urban fire model. The interplay of various urban design elements in
contemporary cities, as seen in rapidly developing urban areas like Cebu,
Philippines, necessitates tailored modeling approaches to address these
complex fire propagation challenges.
**Cellular Automata**
Cellular Automata (CA) is a discrete modeling approach widely
adopted for simulating complex systems. Comprising a grid of cells with
finite states evolving via local transition rules. The capacity of cellular
automata (CA) to finely express temporal and spatial dynamics makes
them popular in fire spread modeling. The flexibility, computational
effectiveness, and simplicity of use of CA models make them superior to
alternative approaches such as empirical models or computational fluid
dynamics (CFD).


Freire and DaCâmara (2019) describe CA as one of the most
important stochastic models for wildfire spread, where space is discretized
and local interactions determine fire propagation. CA models may readily
accept any empirical or theoretical fire propagation mechanism, even
complicated ones, and directly incorporate spatial variation in terrain, fuel
properties, and climatic circumstances. According to Sullivan (2009),
cellular automata (CA) are among the most significant stochastic models.
Physical quantities are discretized into cells, and for each cell, they
assume a limited set of values. In discrete time, cells change based on a
set of transition rules and the conditions of their nearby cells.

Karafyllidis et al. (1997) introduced a CA model that uses local
rules based on fire spread relationships to predict the spread of forest fires
in urban settings. Their model was verified against actual fire data and
showed flexibility in including environmental elements such as topography
and weather. Hernandez Encinas et al. (2007) introduced a novel CA
algorithm for wildfire simulation, addressing distorted fire shapes by
allowing non-constrained spread directions, enhancing accuracy in
heterogeneous environments.
Unlike computational fluid dynamics models, Cellular Automata
have much lower computational cost, enabling rapid ensemble runs and
real-time scenarios. Cellular Automata offers faster simulations suitable for
large-scale scenarios. Patac and Vicente (2019) emphasized that
time-wise, automation is among the most straightforward techniques
whereas physics-based models demand high-end resources and complex
setup and in terms of memory, that can depict a phenomenon, like heavy
traffic.
Empirical models, while simpler, often lack the spatial detail
provided by CA’s grid-based approach. The ability to integrate CA with ML
further enhances its adaptability, making it a preferred choice for this
thesis. CA models can underestimate spread in highly irregular


environments if transition rules are not well-calibrated. Integrating ML to
derive these rules addresses this limitation.
**Cellular Automata on Fire Spread Simulation**
Cellular Automata has been applied successfully to both wildfire
and urban fire simulation and it produced accurate predictive data. In a
traditional Cellular Automata fire spread for both urban and wildfire, each
cell represents a specific location and its transition from a state from
unburned to burnt based on the parameters and other possibilities. Patac
and Vicente (2019) developed a CA model for urban fire spread in Basak,
Lapu-Lapu City, Philippines, integrating Extreme Learning Machine (ELM)
to derive transition rules. Their model achieved 78-83% accuracy in
predicting fire spread, demonstrating CA’s effectiveness in urban settings
when enhanced with ML. These implementations show how ignition
likelihood is adjusted by fuel load, moisture, building combustibility, wind,
slope, and other factors in typical CA parameterization. According to Collin
et al. (2011), usually the transition rules used by the CA are either set or
obtained by identification from experimental results.
These studies highlight that Cellular Automata (CA) deliver efficient,
high‑resolution fire‑spread simulations by discretizing landscapes into
simple, local interaction rules. When augmented with machine‑learned
transition probabilities such as the Extreme Learning Machine approach in
Basak, Lapu‑Lapu City. They achieve robust predictive accuracy
(78–83%) while remaining computationally lightweight. CA’s flexibility in
encoding diverse factors (fuel load, moisture, combustibility, wind, slope)
and its capacity to integrate data‑driven rule estimation make it an ideal
choice for modeling fire dynamics in complex urban and wildland
environments. The combination of CA’s spatial explicitness and ML’s
adaptive calibration yields a powerful framework for rapid, reliable fire‑risk
assessment and scenario analysis.
**Fire Spread Modelling using Machine Learning**


Machine learning (ML) has significantly enhanced fire spread
prediction by improving accuracy and handling complex data patterns.
Machine Learning (ML) enhances fire spread modeling by leveraging data
to improve prediction and decision-making. Andrianarivony and Akhloufi
(2024) stated that with the emergence of machine learning (ML) and deep
learning (DL), new methods have been developed that greatly improve
prediction accuracy. Tabular data points are used by machine learning
models, including support vector machines and ensemble models, to find
trends and forecast fire behavior. In wildfire applications, ML models which
use regression trees to neural networks, are trained on past fire events
and environmental variables to forecast spread rates or burned areas.
More current techniques concentrate on deep learning and
machine learning methodologies (Andrianarivony & Akhloufi, 2024). Zheng
et al. (2017) simulated the spread of forest fires by combining the
traditional Cellular Automaton (CA) model with the Extreme Machine
Learning (EML) model. The most effective machine learning models for
predicting the spread of grassland fires were found to be boosted trees,
bilayered neural networks, exponential Gaussian process regression, and
linear support vector regression. The performance of fire-predicting
models has increased with deep learning and multimodal data
(Khanmohammadi et al., 2022).
Deep learning, especially Convolutional Neural Networks (CNNs),
has also shown powerful performance. Marjani et al. (2022) achieved high
accuracy using a multi-kernel CNN on multi-source data. Radke et al.
(2019) introduced FireCast, a CNN model trained on geospatial data
(satellite imagery, elevation, weather, historical fire perimeters) to predict
wildfire spread. This model demonstrated high accuracy in identifying fire
spread patterns, highlighting DL’s ability to process multimodal data.
Importantly, ML approaches offer adaptability. Unlike static physical
models, they can incorporate real-time sensor or remote-sensing data to
update predictions.


Machine Learning (ML) and Deep Learning (DL) techniques provide
a data-driven substitute that can get beyond the drawbacks of
conventional models (Andrianarivony & Akhloufi, 2024). Machine Learning
(ML) and Deep Learning (DL) techniques enhance fire modeling by
handling nonlinear influences and exploiting large datasets, thus
improving the situational awareness and decision support in fire
management. ML’s contribution lies in its capacity to uncover nonlinear
relationships and optimize parameters, reducing reliance on manual
calibration. These advancements complement CA’s spatial framework,
offering a pathway to more accurate and scalable fire spread simulations,
particularly in complex urban environments.
Recent applications demonstrate their effectiveness in fire modeling
contexts across various environments. Shahzad et al. (2024) compared
machine learning algorithms for vegetation fire detection in Pakistan,
finding that the random forest model demonstrated the best overall
predictive capability, with an accuracy rate of 87.5% in forest fires,
highlighting the superior performance of ensemble methods. Moghim and
Mehrabi (2024) evaluated logistic regression and random forest algorithms
for wildfire susceptibility mapping, demonstrating that machine learning
approaches can effectively predict fire occurrence using historical data
and influential variables. Zheng et al. (2017) developed a cellular
automaton model integrated with machine learning for forest fire spread
simulation, showing how Cellular Automata models can be enhanced
through data-driven approaches.
These studies illustrate how machine learning methods can adapt
to environmental variations by learning from local fire history and
environmental data, providing dynamic transition probabilities that reflect
real-world fire behavior more accurately than static rule-based
approaches. The algorithms' ability to identify complex patterns in
multidimensional datasets makes them particularly valuable for modeling


fire behavior in heterogeneous environments where traditional empirical
models may struggle to capture the full range of influencing factors.
The integration of these methods with cellular automata represents
a significant advancement in fire modeling capability. Zheng et al. (2017)
demonstrated this integration by combining traditional cellular automaton
models with extreme learning machines for forest fire spread simulation,
achieving improved prediction accuracy. Rather than relying on
predetermined transition rules, CA models can use machine
learning-derived probabilities to determine cell state changes. This hybrid
approach combines the spatial explicitness of CA with the adaptive
learning capabilities of statistical models, creating more robust and
accurate fire spread simulations.
**Parameters Integrated in Fire Spread Modeling**
Fire spread modeling is a critical tool for predicting fire behavior,
aiding in risk management, and informing firefighting strategies. The
parameters used in these models vary significantly between urban and
forest environments due to differences in landscape characteristics, fuel
types, and fire dynamics. Urban areas, with their dense building
configurations and human infrastructure, require parameters that reflect
structural combustibility and spatial arrangement, while forest areas focus
on vegetation, terrain, and environmental factors. This analysis aims to
refine the understanding of these parameters, ensuring relevance to the
study's focus on urban fire spread modeling.
Forest fire spread models focus on parameters relevant to natural
landscapes, where vegetation, terrain, and environmental conditions
dominate fire behavior. Zhong et al. (2017) provides insights into
forest-specific parameters. The parameters are inferred from the study's
focus on forest fire dynamics and its validation using data from five fires in
the west of the United States, achieving good performance in predicting
cell ignition probabilities. The inclusion of land surface temperature and


enhanced vegetation index aligns with the model's emphasis on
environmental factors, as noted in related literature.
The study by Moghim and Mehrabi (2024) on wildfire prediction utilized 11
parameters to identify fire-prone areas. Namely, these parameters are
elevation, slope, aspect, curvature, temperature, precipitation, wind, land
use cover, vegetation index, and distance to roads. These primary
parameters were used to predict fire prone areas. While these parameters
are relevant for wildfire prediction, they are more aligned with forest and
natural landscapes, emphasizing terrain and vegetation. Urban fire
models, as discussed, require a shift toward structural and spatial
parameters, highlighting the need for tailored approaches.
Urban fire spread models, in contrast, must incorporate parameters
that reflect the built environment, where buildings, roads, and human
activity significantly influence fire dynamics. Patac and Vicente (2019)
developed a model integrating cellular automata with an Extreme Learning
Machine. The parameters include fuel load, moisture, building
combustibility, wind, and slope. These parameters achieved 78-83%
accuracy in predicting fire spread, highlighting their relevance to urban
contexts where building materials and urban layout are critical.
In their work on urban fire spread prediction, Takizawa et al. (2000)
emphasized building-related parameters such as width of the building,
distance to the next building, wind effects, time steps, fire probability of
wooden cells, gradual diminution rate by distance "d" from burning cell
which is a measure of how fire intensity decreases with distance from the
ignition source, affecting spread patterns. These parameters underscore
the importance of spatial and material factors in urban fire modeling,
aligning with the need to account for the built environment's complexity.
The differentiation between urban and forest fire modeling
parameters underscores the necessity for tailored approaches in fire
spread prediction. Urban models must account for the combustibility of
structures, the spatial arrangement of buildings, and factors like building


width and distance, which are critical for simulating fire jump and
propagation in dense urban areas. In contrast, forest models prioritize
vegetation characteristics, such as the enhanced vegetation index, and
environmental factors like land surface temperature, which are less
relevant in urban settings.
The analysis reveals that urban fire spread models require
parameters tailored to the built environment, such as fuel load, building
combustibility, and spatial factors, while forest models focus on vegetation
and terrain-related variables. This differentiation is essential for developing
effective predictive tools, ensuring accuracy in fire management and urban
planning. By incorporating insights from Patac and Vicente (2019),
Takizawa et al. (2000), and Zhong Zheng et al. (2017), the study can build
a robust foundation for urban fire spread modeling, addressing the specific
needs of high-risk urban areas.
**Logistic Regression & Random Forest for Fire Spread Modelling**
Random Forest (RF) and Logistic Regression (LR) are commonly
used algorithms for estimating ignition probabilities in fire‑spread
simulation. Logistic Regression offers a straightforward, interpretable
framework by fitting a logistic curve to historical burn/non‑burn labels,
directly producing a probability of ignition for each cell. Its coefficients
reveal the relative importance of predictors such as wind speed, fuel load,
and building combustibility (Vakalis et al., 2004). The minimal
computational requirements make LR suitable for rapid prototyping and for
use in resource‑constrained settings, while its probabilistic output
integrates seamlessly into cellular automata (CA) transition rules without
additional calibration (Moghim & Mehrabi, 2024).
Random Forest extends beyond LR’s linear assumptions by training
an ensemble of decision trees on bootstrap samples with random feature
subsets, thereby capturing nonlinear interactions among environmental
and structural factors. RF naturally yields per‑cell fire‑probability scores
computed as the proportion of trees voting “ignite” and has demonstrated


strong predictive accuracy in complex fire contexts (Moghim & Mehrabi,
2024). Its built-in feature‑importance measures help identify which
variables most influence spread, guiding both model refinement and
practical mitigation strategies. RF’s robustness to overfitting ensures
stable performance when simulating fire spread across heterogeneous
urban layouts, making it an effective complement to the simpler LR
baseline.
Both methods have established applications in CA‑driven fire
modeling: LR for its clarity and efficiency, and RF for its flexibility in
handling complex interactions (Zheng et al., 2017). By combining LR and
RF in our study, we leverage their complementary strengths—interpretable
probability estimates and robust nonlinear modeling—to drive accurate,
data‑driven CA simulations of fire spread in diverse urban settings.

Table 1
_Summary of Related Literature_

```
Authors Variables/Methods Findings
Ohgai et al., (2004) Flammability score
and 3x3 cell size
```
```
Used flammability score
to provide building
combustibility.
Himoto and Tanaka
(2003, 2008)
```
```
Building materials,
urban layout, wind
patterns,
topography,
firefighting efforts,
firebrand spotting,
wind-blown fire
plumes, thermal
radiation
```
```
Developed
physics-based models
for urban fire spread,
highlighting
building-to-building
ignition in dense areas,
validated with Japanese
historical fire data
```
```
Vakalis et al. (2004) Urban layout Used a GIS-based
system to simulate fire
propagation, showing
that urban
configurations influence
predictable vs. chaotic
```

```
fire spread patterns.
```
He & Weng (2025) Building
configurations,
urban density

```
Found that compact
urban forms increase
fire spread risk due to
multiple hazards
```
Purnomo et al.
(2024)

```
Layout
configurations, fuel
loads
```
```
Noted that isolated
combustible islands
slow fire spread, while
continuous fuel loads
(e.g., dense wooden
structures) enable rapid
propagation
```
Shaham &
Benenson (2018)

```
Flammable urban
vegetation, building
structures, urban
form
```
```
Demonstrated that
dense building blocks
intensify burning, while
wide roads or open
areas act as firebreaks
in Mediterranean and
Middle Eastern cities.
```
Freire & DaCâmara Terrain, fuel
properties, climatic
conditions

```
Described CA as a key
stochastic model for
wildfire spread, capable
of incorporating complex
fire propagation
mechanisms
```
Sullivan (2009) Physical quantities
(discretized into
cells)

```
Explained CA’s use of
discretized cells and
transition rules based on
neighboring conditions
for fire spread modeling.
```
Hernandez Encinas
et al. (2012)

```
Fire spread
directions
```
```
Developed a CA
algorithm for wildfire
simulation, improving
accuracy in
heterogeneous
environments by
allowing
non-constrained spread
directions
```

Patac and Vicente
(2019)

```
Fuel load, moisture,
building
combustibility, wind,
slope
```
```
CA model with Extreme
Learning Machine
achieved 78-83%
accuracy in urban fire
spread prediction
```
Radake et al. (2019) Satellite imagery,
elevation, weather,
historical fire
perimeters

```
Introduced FireCast, a
CNN model for wildfire
spread, demonstrating
high accuracy in
identifying spread
patterns.
```
Alexandridis et al.
(2011)

```
Spread probability
under no wind and
flat terrain, slope
parameter, moisture
parameter 1 and 2,
wind parameter 1
and 2, and
vegetation height
```
```
Produced an enhanced
CA framework that
accounts important
factors
```
Takizawa et al.
(2000)

```
Building width,
distance to next
building, wind, time
steps, fire
probability of
wooden cells,
gradual diminution
rate
```
```
Importance of spatial
and material factors in
urban fire modeling
```
Zheng et al. (2017) Cellular automata
and Extreme
Machine Learning,
time steps

```
Simulated forest fire
spread, enhancing CA
with machine learning
```
Moghim & Mehrabi
(2024)

```
Logistic Regression,
Random Forest,
cellular automata,
historical burn data,
wind speed, fuel
load, building
combustibility.
```
```
LR provides
interpretable probability
estimates while RF
captures nonlinear
interactions with higher
accuracy, both
integrating seamlessly
with cellular automata
for data-driven fire
```

spread simulations.


## CHAPTER

### TECHNICAL BACKGROUND

This chapter contains the definition of technical terms used in prior
chapters or to be used in the upcoming chapters. The said terminologies
are from the field of computer science and fire related.
**Algorithm**
A finite, well‐defined sequence of computational steps that
transforms input into output. Algorithms are the basis for all computer
programs, including those used in simulation and machine learning
(Cormen et al., 2009).
**Building Combustibility**
A qualitative measure of how readily a building material ignites and
contributes to fire growth, typically classified by standardized tests.
**FireCast**
FireCast is an algorithm that predicts wildfire spread prediction over
24 hours. It is more accurate than FARSITE (Radke et al., 2019).
**Cellular Automata**
A discrete, spatially explicit modeling framework in which a domain
is partitioned into cells that update their states simultaneously according to
local transition rules; CA are valued for modeling complex spatial
phenomena with simple, computationally efficient rules. A one-dimensional
cellular automaton consists of a line of sites, with each site carrying a
value 0 or 1 (or in general 0,... , k - 1). The value a, of the site at each
position i is updated in discrete time steps according to an identical
deterministic rule depending on a neighbourhood of sites around it
(Wolfram, 1984).
**Burn Probability**
The likelihood that a given spatial unit (cell) will transition from
unburned to burned under specified environmental conditions, often
estimated via ensemble CA simulations or statistical models.
**Random Forest**


An ensemble machine‐learning method that constructs a multitude
of decision trees at training time and outputs the mode (classification) or
mean (regression) of their predictions; known for robustness and ease of
interpretation (Breiman, 2001).
**Machine Learning**
The study and construction of algorithms that improve their
performance at some task through experience (data), encompassing
supervised, unsupervised, and reinforcement paradigms (Mitchell, 1997).
**Benchmarking**
The process of systematically evaluating and comparing
computational methods on standardized datasets and metrics to assess
accuracy, efficiency, and robustness.
**Fire Propagation**
The mechanisms by which fire spreads through a domain, including
heat transfer, ember transport, and convective processes; in modeling,
this refers to both the physical phenomena and their algorithmic
representations (Rothermel, 1972).
**Urban Layouts**
The spatial configuration of streets, blocks, buildings, and open
spaces in a city; different morphologies influence hazard propagation,
accessibility, and risk.
**Model Training**
The phase in machine learning during which an algorithm adjusts
its internal parameters (e.g., neural network weights or tree split criteria)
by minimizing a loss function on a labeled dataset (Bergmann & Stryker,
2025).
**Logistic Regression**
Logistic regression is a statistical method used for binary
classification problems that estimates the probability of an event occurring.
Unlike linear regression, it uses the logistic function (sigmoid curve) to
map any real-valued input to a value between 0 and 1, making it suitable


for probability estimation. In fire spread contexts, logistic regression can
predict the probability of fire occurrence or spread based on environmental
variables such as temperature, humidity, wind speed, and fuel load.
**Random Forest**
Random forest is an ensemble learning method that combines
multiple decision trees to improve prediction accuracy and reduce
overfitting. It works by training many decision trees on different subsets of
the training data and features, then averaging their predictions for
regression or taking a majority vote for classification. This approach
provides better generalization than single decision trees while maintaining
reasonable interpretability through feature importance rankings.
**Transition Rules**
In cellular automata models, transition rules define how cells
change state based on their current state and the states of neighboring
cells. For fire spread modeling, these rules determine the probability that
an unburned cell will ignite based on factors such as the number of
burning neighbors, environmental conditions, and local fuel
characteristics.
**Feature Importance**
Feature importance is a machine learning concept that quantifies
the relative contribution of each input variable to the model's predictions.
In fire modeling, feature importance helps identify which environmental
factors (such as wind speed, temperature, or fuel moisture) have the
greatest influence on fire spread, enabling better understanding and
management of fire risk.
**Fuel Load**
Refers to the amount of combustible material, particularly
vegetation, present in a given area.
**Geospatial**
Information that describes objects, events or other features with a
location on or near the surface of the earth.


### CHAPTER 4

## DESIGN AND METHODOLOGY..............................................................

This chapter details the tools and procedures used to assist the
researchers in formulating and developing a cellular based machine
learning model that predicts fire spread in an urban layout. This includes
the research environment, research instrument, research procedure,
concept, analysis and design, development model, development
approaches, software development tools, project management, and
verification, validation and testing.

**4.1 Research Environment**

The study leverages Python 3.13.3 as the core programming
language, utilizing libraries such as GeoPandas for geospatial data
handling, Scikit-Learn for machine learning, and Matplotlib for
visualization. Simulations are executed on a high-performance computing
cluster to manage the computational load of large-scale urban grids.

The training area for the two Machine Learning would be conducted
in Pusok and Pajo, two barangays located in Lapu-Lapu City, Philippines.
Pusok features dense residential and commercial zones, characterized by
closely packed buildings and limited open spaces, making it prone to rapid
fire spread. In contrast, Pajo encompasses a mix of residential, industrial,
and open spaces, offering a varied spatial configuration. Both barangays
have a documented history of fire incidents, as evidenced by datasets that
will be obtained from the Bureau of Fire Protection (BFP), which provide
critical historical data for validating the predictive fire spread model. The
diverse urban characteristics of Pusok and Pajo make them suitable
venues for testing the model’s effectiveness across different spatial
layouts, enhancing the study’s relevance to urban fire management.


**4.2 Research Instrument or Sources of Data**

The study combines established and custom tools to assemble a
robust dataset for modeling urban fire spread in grid and radial layouts.
OpenStreetMap, Bureau of Fire Protection historical records, and the
OpenWeatherMap API supply geospatial building and road data, past fire
perimeters, and weather variables such as wind speed and temperature.
Custom Python scripts and QGIS workflows preprocess these inputs to
generate derived features like flammability scores and to construct the
cellular automata grid. All code undergoes unit testing and expert peer
review, and automated outputs are routinely cross‑checked against
manual calculations to ensure accuracy and reliability.

**4.3 Research Procedure**

```
4.3.1 Gathering of Data
The process of gathering data is crucial for feeding the
necessary loads of data to train the Random Forest and Logistic
Regression models. The following data types will be used:
a. Urban Layout Grid
The urban layout grid will be generated programmatically
using QGIS by importing geospatial data from OpenStreetMap
(OSM). Overpass API and QuickOSM plugin will be used to gather
information on buildings, roads, etc. Discrepancies in building
footprints and road networks will be corrected using QGIS by
overlaying OSM data with satellite imagery, supplemented by
manual adjustments or automated Python scripts where necessary.
The urban layout grid is generated programmatically using QGIS,
with each cell containing attributes relevant to fire spread (e.g.,
building presence, vegetation cover, elevation). Each cell contains
attributes relevant to fire spread, such as building presence,
vegetation cover, and elevation. QGIS will import the OSM data,
creating a raster grid and assigning spatial attributes to each cell.
```

Geopandas will preprocess the grid for integration with machine
learning models, ensuring the compatibility with the CA framework.
**b. Building Characteristics**
Building characteristics, including material type (e.g., wood,
concrete), height, and occupancy type (residential, commercial,
industrial), will be derived from OSM data using tags. Each building
in the CA grid will be assigned to a flammability score based on its
material type and height. For example, wooden residential buildings
may have higher flammability than concrete commercial structures.
Cells without buildings will be marked as unburnable. These
attributes will serve as input features for Random Forest and
Logistic Regression to predict fire spread likelihood. QGIS will
extract and integrate the building data from the OSM into the grid.
Python with OpenCV and PyTorch will process street-view images
for material classification, with results imported into QGIS for spatial
integration
**c. Fire Incident Data**
Data on past fire incidents from the BFP archive, including
ignition points and burned areas, used to train machine learning
models by providing labeled outcomes (burned vs. not burned).
This data will label CA grid cells as "burned" or "not burned" for
past fire incidents, providing ground truth for training Random
Forest and Logistic Regression models. Geopandas will be used to
geocode the fire incident data, aligning it with the QGIS spatial
database. QGIS will overlay fire incidents on the urban layout grid
for visualization and analysis. Geopandas will geocode the data,
aligning it with the QGIS spatial database for integration into the
cellular automata (CA) framework.
**d. Vegetation/Fuel Load**
Vegetation data will be sourced from OSM. Satellite imagery
from Google Earth will supplement OSM data. Each CA grid cell will


be assigned a fuel load based on the vegetation type and density,
with higher fuel loads increasing ignition and spread probabilities.
These values will also serve as input features for machine learning
models to predict fire behavior in vegetated urban areas. QGIS will
process satellite imagery to create vegetation layers using Raster
Calculator. Python scripts will automate fuel load assignments
based on vegetation classification, integrating with the CA grid.
**e. Elevation**
Elevation data will be obtained from topographic maps
provided by the National Mapping and Resource Information
Authority (NAMRIA). : Elevation affects fire spread, as fires
propagate faster uphill. In the CA model, elevation differences
between neighboring cells will adjust transition rules, increasing
ignition probabilities or spread rates for uphill cells. Elevation,
slope, and aspect will serve as input features for Random Forest
and Logistic Regression models to predict ignition probabilities or
state transitions (e.g., unburned to burning), improving accuracy in
varied terrains. The elevation data will be converted into a Digital
Elevation Model (DEM) raster layer in QGIS, aligned with the CA
grid, ensuring each cell has associated topographical attributes.
**f. Weather**
Weather data, including wind speed, direction, temperature,
and humidity, will be sourced from the OpenWeatherMap API.
Weather data from OpenWeatherMap will be augmented with
real-time weather updates via an API connection, fetching wind
speed, direction, temperature, and humidity during simulations.
Weather parameters will be used as input features for Random
Forest and Logistic Regression to predict fire spread under varying
conditions. In the CA model, weather data will dynamically adjust
transition rules, such as increasing spread rates in the wind’s
direction, enhancing simulation realism. Python will download and


process weather data using Pandas, integrating it with the CA
simulation. QGIS can visualize weather impacts, such as wind
direction effects, on the urban layout grid.
**Data Harmonization**
All data will be standardized to the WGS 84 coordinate
system using GDAL’s ogr2ogr. Formats will be unified into
GeoPackage using GDAL, followed by spatial joins in GeoPandas
to create a single GeoDataFrame. Missing values will be imputed
(e.g., median for numerical data, mode for categorical) using
Pandas.
**4.3.2 Treatment of Data**
This section outlines the procedures for processing,
analyzing, and presenting the data to develop a predictive model
that integrates cellular automata (CA) with machine learning (ML)
techniques, specifically Random Forest and Logistic Regression.
The datasets include urban layout grids, building characteristics,
fire incident data, vegetation/fuel load, topographical data, and
weather data (wind speed and direction). The processing steps
ensure data quality and compatibility, while the analysis leverages
ML and CA to predict fire spread in grid and radial urban layouts.
**Urban Layout Grid**
The source for the Urban Layout Grid would be obtained
from OpenStreetMap (OSM) data, accessed via the QGIS’s
Quick-OSM plugin. The urban layout is created as a raster layer
with 3x3 meter resolution as recommended by Ohgai et al. (2004).
The urban layout is rasterized into a 3x3 meter grid using QGIS’s
Rasterize tool. Cells include attributes like building presence
(binary), vegetation cover (percentage), and elevation (meters).


## Figure 1. Urban Layout Grid................................................................ LIST OF FIGURE

## Figure 2. Urban Layout Grid (3x3 cells)...............................................

```
Cells are assigned attributes such as building presence
(binary: 1 for present, 0 for absent), vegetation cover (percentage),
and elevation (meters). Missing or inconsistent data (e.g.,
incomplete OSM tags) are handled by imputing median values for
numerical attributes or mode values for categorical attributes using
Python’s Pandas library. The grid is exported as a GeoTIFF for
```

integration with the CA simulation. Cells without buildings are
marked as unburnable to prevent unrealistic fire spread.
**Building Characteristics**
To gather building information such as materials and height,
OSM tags will be implemented
_building = residential,height_

Calculated per Ohgai et al. (2004), scores (0–1) reflect
material type and height for buildings; non-building cells score 0.
Each building is assigned a flammability score (0-1, where 1 is
highly flammable) based on the material type, with non-building
cells marked as unburnable (score = 0).

Gao et al. (2008) defined three factors which affect the fire
spread in a given area. These factors will be considered in the cell
attributes in the simulation.

1. Building factors - These will include the building composition,
    height of the building, and area size. These will be the basis
    for the possibility of a building (cell) to catch a fire.
2. Weather conditions - Includes the wind speed and direction.
    The higher each value is, the larger the possibility of it
    increasing the likelihood of fire spreading.
3. Landscape - Example for this would be vacant spots, roads,
    etc.
    Data inconsistencies (e.g., missing heights tags, material
type, etc.) are resolved by assigning values from similar building
types in the area. The processed data are stored as a vector layer
in QGIS and exported as a CSV for ML training.
**Residential Fire Incident Data**
Obtained from the Bureau of Fire Protection (BFP),
residential fire incident data include ignition points, burned areas,
and contributing factors (e.g., cause, date). The data are geocoded


using Python’s Geopandas to align with the QGIS spatial
database.Missing coordinates are inferred from address
descriptions using geocoding APIs (e.g., Nominatim). Each grid cell
is labeled from State 1 to State 5 based on historical fire
perimeters, creating a target variable for ML training. Outliers (e.g.,
erroneous coordinates) are removed by filtering points outside the
study area.
2D cells will be used to reduce complexity in this study. State
of the cell will be adapted from the studies of Patac & Vicente
(2019), Ohgai et al. (2004), and Gao et al. (2008). Each cell in the
grid will be described as following:
**State 1:** Not possible to burn; Nothing to be burned. **State 2:**
Not yet burning; A cell that has the potential to burn. **State 3:**
Ignited; Caught fire and started burning but has no ability to spread.
**State 4:** Blazing; Burning strongly and has the ability to affect
neighboring cells. **State 5:** Extinguished; Burned out.
**Fire Spread Simulation**
The work of Patac and Vicente (2019) served as the
foundation of the flow of how cells transition to another state. The
figure below summarizes the flow of the simulation:


## Figure 3: Fire Spread Simulation Flow Chart.......................................

```
Conversion From Each State to Another
For cell transition from one state to another, rules must be
established in order to achieve such results. These rules will be
established:
Rule 1: State 1 cells will never change throughout the simulation
period.
Rule 2: State 5 cells will remain burned out in continuing phases.
Rule 3: State 4 has the ability to transition to State 5 given the cell
attributes allows it to burn greater than the defined set time.
Rule 4: State 3 that has caught fire will be transitioned to State 4 if
the burning time is greater than the defined set time.
Rule 5: State 2 will transition to State 3 if one or more of its
neighbouring cells are burning.
```

**Time Threshold**
The following equations govern the time required for state
transitions in the cellular automata model for urban fire spread, as
per the specified transition rules.
**State transition of State 2 to State 3**
Transition occurs when at least one neighboring cell is in
State 4 (Blazing). The time to transition is given by:

```
𝑇 2 → 3 = 𝐹 · ( 1 + 0. 5 · 𝑐𝑜𝑠(θ𝑇)𝑏𝑎𝑠𝑒) · 𝑃𝑀𝐿, 𝑖𝑔𝑛𝑖𝑡𝑒 · 𝑁𝑏𝑙𝑎𝑧𝑖𝑛𝑔
```
Where:

- 𝑇𝑏𝑎𝑠𝑒 = Base transition time, based on historical fire data.
- 𝐹 = Flammability score of the building.
- θ = Angle of wind direction and spread direction.
- 𝑃𝑀𝐿, 𝑖𝑔𝑛𝑖𝑡𝑒 = Ignition Probability.
- 𝑁𝑏𝑙𝑎𝑧𝑖𝑛𝑔 = Number of neighboring cells in State 4.

**State transition of State 3 to State 4**
Transition occurs when the burning time exceeds a
threshold. The time to transition is:

```
𝑇 3 → 4 = 𝐹 · ( 1 + 0. 5 ·𝑇 𝑐𝑡ℎ𝑜,^ 𝑠𝑏𝑎𝑠𝑒(θ)) · 𝑃𝑀𝐿, 𝑖𝑔𝑛𝑖𝑡𝑒
```
- 𝑇𝑡ℎ, 𝑏𝑎𝑠𝑒 = Threshold time for blazing, based on cell attributes.
- 𝑃𝑀𝐿, 𝑏𝑙𝑎𝑧𝑒 = Probability for blazing transition.

**State transition of State 3 to State 4**
Transition occurs when the burning time exceeds a
threshold. The time to transition is:

```
𝑇 4 → 5 = 𝐹 𝑇· 𝑃𝑡ℎ𝑀𝐿,^ 𝑒𝑥𝑡𝑖𝑛𝑔𝑢𝑖𝑠ℎ, 𝑒𝑥𝑡𝑖𝑛𝑔𝑢𝑖𝑠ℎ
```
- 𝑇𝑡ℎ, 𝑒𝑥𝑡𝑖𝑛𝑔𝑢𝑖𝑠ℎ = Threshold time for extinguishment, based on

fuel depletion.

- 𝑃𝑀𝐿, 𝑒𝑥𝑡𝑖𝑛𝑔𝑢𝑖𝑠ℎ = Probability for extinguishment.


**Vegetation**
Vegetation data from OSM and Sentinel-2 satellite imagery
are processed in QGIS to create a raster layer of fuel load values
(0–1, where 1 indicates high fuel load). The Raster Calculator
assigns fuel loads based on vegetation type and density, validated
against ground truth if available.
_(e.g., grass = 0.5, trees = 0.8)_
Missing data are imputed using nearest-neighbor
interpolation. The layer is exported as a CSV for ML feature
engineering.
**Elevation**
Sourced from NAMRIA topographic maps (1:500,000 scale,
20m contour intervals), elevation data are converted into a Digital
Elevation Model (DEM) using QGIS’s “Contour to DEM” plugin.


## Figure 4. Cebu City Index Map Scale 1:50,000 scale..........................

```
Slope and aspect are derived using Raster Analysis tools.
Outliers (e.g., erroneous contour lines) are corrected by smoothing
the DEM with a Gaussian filter. The DEM is aligned with the CA
grid, and elevation, slope, and aspect are extracted as ML features.
Weather Data
Wind speed (m/s), wind direction (degrees), temperature
(°C), and humidity (%) are sourced from the OpenWeatherMap API,
aligned with historical fire incident timestamps. Missing values are
imputed using time-series interpolation (e.g., linear interpolation in
Pandas). Wind direction is decomposed into sine and cosine
components to handle its circular nature, ensuring compatibility with
ML models.
```
```
Figure 5. Effects of Western Wind (Gao et al., 2008)
```
**Wind Effect**
As shown in Figure 5, the number of burned cells increases
as the wind’s velocity increases. As the strength of the wind also
increases, the cells on the leeward side have more chances of
burning compared to the direction from where the wind is actually
coming from. This will be taken into consideration into the model.
**4.4 Concept**
The core concept of this research is to develop a predictive fire
spread modeling system that combines cellular automata (CA) simulation
with machine learning-derived ignition probabilities specifically designed


for diverse urban layouts. Unlike traditional CA fire models that rely on
static, hand-tuned transition rules based on physical heuristics such as
Rothermel's equations (Rothermel, 1972), this system learns ignition
probabilities from historical fire data and integrates these learned
probabilities into the CA transition mechanism.
Figure 6 illustrates the conceptual framework of the ML-enhanced
CA fire spread modeling system. The process begins with spatial data
preprocessing, where urban layout geometries, building characteristics,
and environmental factors are extracted from geospatial datasets using
GeoPandas. This spatial foundation provides the grid structure necessary
for CA simulation while preserving the unique characteristics of diverse
urban patterns that create complex fire propagation pathways. The
framework then proceeds with feature engineering, where relevant fire
spread drivers are identified and quantified. These include building
materials and density, topographical slope, wind patterns, fuel load
distribution, and neighborhood influence factors. Historical fire incident
data including fire location, date and time of incidents, cause of fire,
building materials involved, and urban layout characteristics such as street
patterns and building densities will be obtained from local fire station
records. This comprehensive dataset is processed alongside
environmental features to create training datasets that capture the
relationship between urban morphology and fire behavior.


_Figure 6._ Conceptual Framework
The machine learning component employs supervised learning
algorithms implemented via scikit-learn to learn ignition probability
functions from the historical data. These learned probabilities replace
traditional fixed transition rules, allowing the system to adapt to local
conditions and urban geometry patterns. This data-driven approach
addresses the limitation of conventional CA models that fail to account for
the directional biases and geometric constraints present in complex urban
environments. The CA simulation engine integrates these ML-derived
probabilities into the cellular automata framework, where each cell's state
transition is governed by the learned probability functions rather than
predetermined rules. The simulation propagates fire spread across the
urban grid, accounting for directional influences inherent in various urban
patterns and the varying fire spread dynamics these patterns create.
Finally, the validation and visualization component compares predicted fire
spread patterns against historical fire perimeters and provides interactive
visualization of fire progression scenarios. This end-to-end pipeline,
implemented entirely in Python, ensures seamless integration from data
ingestion through model training to simulation output and validation,
creating a robust framework for predictive fire spread modeling in complex
urban environments.


**4.5 Analysis and Design**

## Figure 7. DataFlow Diagram (Level 0).................................................

Figure 7 illustrates the process by which we evaluate two
machine‑learning models within our CA fire‑spread framework. The
methodology begins with Dataset Preparation, where all spatial and
temporal features (building materials, slope, wind, neighbor count, and
burn labels) are assembled. From this single dataset, we first Train
Logistic Regression, producing an LR model that outputs ignition
probabilities (𝑃𝑖𝑔𝑛𝑖𝑡𝑖𝑜𝑛.). Next, we run CA Simulation with LR, using those

probabilities to drive the transition of cells from “unburned” to “burning” to
“burned,” yielding Simulation‑LR results. We then Evaluate Simulation‑LR
against historical perimeters using metrics such as the Jaccard index,
ROC AUC, and confusion‑matrix statistics. Afterwards, we Train Random
Forest on the same dataset, Simulate CA with RF to produce
Simulation‑RF, and Evaluate Simulation‑RF using identical metrics. Finally,
compare & select, we choose the model LR or RF that best balances
predictive performance, computational efficiency, and interpretability.


**4.6 Development Model**

## Figure 8. Incremental Development Model..........................................

This study utilizes an Incremental Development Model to guide the
project's five-month implementation. By breaking the work into clear,
manageable stages, this approach allows for the continuous evaluation of
functional components. This ensures steady progress, facilitates early
feedback, and effectively minimizes project risks.

**Data Acquisition and Preparation:** The initial phase focuses on
acquiring and processing all required inputs. This includes collecting
building footprints and road networks from OpenStreetMap, terrain data
from NAMRIA, meteorological time series from OpenWeatherMap, and
historical fire records & perimeters from Bureau of Fire Protection (BFP).
Scripts are then implemented to convert these raw files into a unified,
clean feature matrix ready for machine learning.

**Baseline Modeling and Prototyping:** Using the prepared dataset,
an initial Logistic Regression model is trained to establish an interpretable
baseline. The predicted ignition probabilities from this model are
integrated into a minimal cellular automata (CA) engine, generating the


first prototype of the fire-spread simulation for an initial performance
evaluation.

**Secondary Model Development and Comparison:** Following the
baseline implementation, a Random Forest model will also be trained on
the same dataset. A second simulation is then run using this new model.
The results from both the Logistic Regression and Random Forest
simulations are compared side-by-side using identical metrics to
determine the superior approach.

**Model Selection and Refinement:** The better-performing model
from the previous stage is officially selected. It then undergoes a round of
sensitivity analysis varying key parameters like wind weighting and
machine learning thresholds to ensure its robustness and reliability.

**Final Analysis:** The final phase involves compiling all code,
simulation results, and analyses into a comprehensive final report. This
package will present the detailed performance comparison between the
models, graphical outputs from the simulations, final accuracy metrics,
and a concluding recommendation on which model is better suited for the
task based on the evidence.

**Risk Assessment and Mitigation:** Risks must be avoided as
much as possible. However, risks such as delays in obtaining BFP
historical fire data due to bureaucratic processes, inaccuracies or
incompleteness in OSM or NAMRIA data, unexpected changes in project
scope, etc. are possible and must be considered. Mitigation strategies will
be implemented. The researchers will use alternative data sources if
primary sources are delayed.

**4.7 Development Approaches**

The development approach utilizes the bottom‑up methodology.
This means the complete CA + ML fire‑spread system is constructed by


first building and validating individual modules, then progressively
integrating them into higher‑level workflows.

## Figure 9. Bottom-Up Approach of CA-ML Fire System........................

Figure 9 illustrates the bottom‑up approach for our predictive
fire‑spread simulator. We begin with Data Ingestion & Preprocessing,
where OpenStreetMap building footprints, NAMRIA topographic maps for
elevation data, OpenWeatherMap API for wind/humidity parameters, and
Bureau of Fire Protection (BFP) historical fire incident data are collected


and cleaned. Next, the Feature Engineering module computes static
attributes (material type, building height, slope, aspect, building
flammability scores) and dynamic attributes (wind‑alignment factor,
neighbor burn count, fuel‑load index) for each 3×3 m grid cell. These
engineered features feed into the Machine Learning module, where both
Logistic Regression and Random Forest models are trained to produce
ignition‑probability predictors 𝑃𝑖𝑔𝑛𝑖𝑡𝑖𝑜𝑛. Finally, the Cellular Automata (CA)

Engine module consumes 𝑃𝑖𝑔𝑛𝑖𝑡𝑖𝑜𝑛 to simulate fire spread over discrete

time steps using a 5-state system (State 1: Not possible to burn; State 2:
Not yet burning; State 3: Ignited; State 4: Blazing; State 5: Extinguished),
generating spatial burn patterns through established transition rules. Each
module is unit tested in isolation before integration; this bottom‑up
assembly ensures that errors are caught early and that each component
functions correctly within the full pipeline.

By advancing from the lowest‑level modules upward, we establish a
robust foundation where each piece is verified before moving on, leading
to a reliable, maintainable, and extensible system that fits our
development timeline and quality objectives.

**4.8 Software Development Tools**

The system suggested in this study requires a number of tools,
software programs, and library packages for development and
implementation. The specific software utilized, along with its version and
function in the project, is listed in detail in Table 2 below.

Table 2
_Software Development Tools_

```
Software Version Purpose
Google Chrome 138.0.7204.1
00/101
```
```
Web browser to access sites
```

### QGIS 3.44.0

```
"Solothurn"
```
```
Geospatial data management,
including creating and processing
urban layout grids, integrating
datasets
```
Python 3.13.3 Core programming language for
data processing, ML, and CA
simulation

Scikit-Learn 1.7.0 Training and evaluating Random
Forest and Logistic Regression
models for predicting ignition
probabilities

Pandas 2.3.0 Data manipulation and analysis for
preprocessing datasets

NumPy 2.3.1 Numerical computations and array
operations for ML and CA

GeoPandas 1.1.1 Geospatial data manipulation and
integration with CA grid

QuickOSM 2.4.1 QGIS plugin for importing OSM
data into geospatial grids

Seaborn v0. 12.0 Statistical data visualization for
enhanced graphical outputs

Git 2.50.1 Version control for collaborative
code management

Github 2.50.1 Repository used by researchers for
collaboration and development.

Visual Studio
Code

```
1.102 Integrated Development
Environment (IDE) for coding and
debugging
```
Cursor 1.0 AI-powered code editor

OpenWeather API One Call API
3.0

```
Retrieving historical weather data
```
Optuna 4.3.0 Hyperparameter optimization for
Random Forest and Logistic
Regression


```
Plotly 6.2.0 Creating interactive visualizations
for fire spread simulations
Rasterio 1.4.3 Processing raster data
Overpass API Latest Querying OpenStreetMap data for
urban layout and building
characteristics
```
**4.9 Project Management**

This section outlines the study’s development plan, including the
timeline, assigned responsibilities, and budget management.
**4.9.1 Schedule and Timeline**
This subsection outlines the project timeline observed by the
researchers to ensure compliance with submission deadlines and
the timely completion of research deliverables. Table 3 details the
tasks and deliverables scheduled across the various stages of the
study, spanning both the first and second semesters of the
Academic Year 2024–2025. The objective for the first semester is to
complete the thesis proposal and commence with model training
and system development, with the overall aim of adhering to the
schedule through the research conference and eventual publication
of the paper.


Table 3
_Gantt Chart for Activities_

**4.9.2 Responsibilities**
The following roles reflect the collaborative nature of this
study. Both researchers contributed equally to the paper's
conceptualization, development, and writing. Specific
responsibilities are detailed to showcase each member's main
areas of focus within this joint project.

Table 4
_Roles and Responsibilities_

```
Member Roles Responsibility
Kristian
Lemuel W.
Diaz
```
```
Developer &
Researcher
```
```
Implement data ingestion and
preprocessing pipelines. Write
Python scripts using GeoPandas
to download and clean
OpenStreetMap data.
Develop and tune the Logistic
Regression model.
Compute and document
performance metrics (ROC AUC,
confusion matrix, Jaccard index).
```

**4.9.3 Budget and Cost Management**
The proposed budget and cost management presented in
the table below will be followed.

Table 5
_Proposed Budget and Cost_

```
Lead the literature review on
CA + ML fire modeling.
Draft the Conceptual Framework
and Methodology chapters
Interpret model results and
contribute to the discussion of
findings.
Kent
Anthony C.
Dulangon
```
```
Developer &
Researcher
```
```
Engineer feature‐extraction code
(static and dynamic cell features).
Implement the Random Forest
model and integrate both models
into the CA simulation engine
Generate “Simulation‑LR” and
“Simulation‑RF” outputs for
comparison.
Assist with the Review of Related
Literature on ML in fire spread
and CA modeling.
Draft the Conceptual Framework
and Methodology chapters
Interpret model results and
contribute to the discussion of
findings.
```
```
Items Cost
Laptop Price
(Acer Nitro 5 & HP AMD Ryzen 5)
```
```
Php 93,000.00
```
```
Conferences Php 60,000.00
```

**4.10 Verification, Validation, and Testing**

This section outlines the plan of activities to ensure the system
developed for the thesis, Predictive Fire Spread Modeling Using Cellular
Automata with Machine Learning in Urban Layouts, is robust, accurate,
and reliable. Quantitative and qualitative measures, along with appropriate
accuracy analysis metrics, are incorporated to achieve the specific
objectives of predicting fire spread in urban layouts, benchmarking
machine learning (ML) models, and enhancing urban fire safety.
Validation ensures that the system meets the research objectives of
accurately predicting fire spread in mixed urban layouts and benchmarking
ML models. The following activities will validate the system’s alignment
with these goals:

1. Comparison with Historical Data
- This study will use historical fire incident data from the BFP
(2019–present) to validate the system’s predictions. Mr. Domidor Suico, a
professional fire fighter, will compare simulated fire spread patterns from
both Simulation-LR (Logistic Regression) and Simulation-RF (Random
Forest) with actual fire perimeters, focusing on ignition points and burned
areas in Cebu City.
2. Quantitative Metrics
Confusion Matrix provides a detailed breakdown of true positives (TP),
false positives (FP), true negatives (TN), and false negatives (FN) for both

```
Printing Php 3,000.00
Miscellaneous Php 5,000.00
Total Php 161,000.00
```

Logistic Regression and Random Forest models. This allows for a
comprehensive assessment of classification performance.

- Accuracy: Measures the proportion of correctly classified cells:

𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦 = (^) 𝑇𝑃 + 𝑇𝑇𝑁𝑃^ ++^ 𝑇𝐹𝑁𝑃 + 𝐹𝑁

- Precision: Measures the proportion of correctly predicted burned
cells:

𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 = (^) 𝑇𝑃 𝑇+𝑃 𝐹𝑃

- Recall: Measures the proportion of actual burned cells correctly
    predicted:

𝑅𝑒𝑐𝑎𝑙𝑙 = (^) 𝑇𝑃𝑇 +𝑃 𝐹𝑁

- F1 Score: The harmonic mean of precision and recall, serving as
the primary metric with a target of 0.80 or higher:

```
𝐹 1 = 2 × 𝑃𝑃𝑟𝑟𝑒𝑒𝑐𝑐𝑖𝑖𝑠𝑠𝑖𝑖𝑜𝑜𝑛𝑛 ×+^ 𝑅𝑅𝑒𝑒𝑐𝑐𝑎𝑎𝑙𝑙𝑙𝑙
```
- Area Under the ROC Curve (AUC-ROC): Evaluates the model’s
ability to distinguish between burned and unburned cells.

The Jaccard Index, defined as 𝐽 = ||𝐴𝐴^ ∩∪^ 𝐵𝐵||, where A is the predicted

burned area and B is the actual burned area, quantifies spatial overlap.
Additional metrics, such as the Dice coefficient, may be used to
complement the Jaccard Index.

- Jaccard Similarity Coefficient: Measures the spatial overlap
between predicted and actual burned areas:

```
𝐽 = ||𝐴𝐴^ ∩∪^ 𝐵𝐵||
```
where A is the predicted burned area and B is the actual burned area.

Model validation compares simulations (Simulation-LR and
Simulation-RF) to BFP historical fire data (2019–present) using recall
(primary, target ≥ 0.80), F1 score (≥ 0.80), AUC-ROC, Jaccard index, and
confusion matrix. Recall is prioritized to minimize false negatives, critical


for fire safety, as supported by A Forest Fire Prediction Model (2024).
Sensitivity analysis ensures robustness across conditions.

Unit testing of individual modules will be implemented to verify
each reliability. Datasets from OpenStreeMap, Bureau of Fire Protection,
National Mapping and Resource Information Authority, and
OpenWeatherMap will be correctly cleaned, integrated, and formatted.
This includes checking for missing values, ensuring data type consistency
(e.g., numerical values for wind speed, categorical labels for building
materials), and confirming spatial alignment in QGIS using tools like the
“Layer Properties” and “Raster Calculator” to validate grid cell attributes.
The integration of ML models with the CA framework will be tested to
ensure that ignition probabilities predicted by Random Forest and Logistic
Regression are correctly incorporated into CA transition rules. This
includes verifying that the grid updates accurately reflect ML predictions
and environmental factors like wind speed and direction. For example,
verify that the output of the Random Forest model (e.g., probability value
between 0 and 1) is used in the CA transition rule, such as
𝑃𝑖𝑔𝑛𝑖𝑡𝑖𝑜𝑛=𝑃𝑏𝑎𝑠𝑒 × ( 1 + 𝑘 1 · ∆ℎ + 𝑘 2 · 𝑤𝑖𝑛𝑑𝑠𝑝𝑒𝑒𝑑 · 𝑐𝑜𝑠(θ)), where

∆ℎ is elevation difference, θ is the wind direction angle, and 𝑘 1 , 𝑘 2 are

constants.


### BIBLIOGRAPHY

Andrianarivony, H. S., & Akhloufi, M. A. (2024). Machine learning and
Deep learning for wildfire spread prediction: a review. Fire, 7(12),

482. https://doi.org/10.3390/fire7120482.
Himoto, K., Tanaka, T., (2003). A Physically-Based Model for Urban Fire
Spread. Fire Safety Science,(7), 129–140.
Himoto, K., Tanaka, T. (2008). Development and validation of a
physics‑based urban fire spread model. _Fire Safety Journal, 43_ (7),
477–494. https://doi.org/10.1016/j.firesaf.2007.12.008
Vakalis, D., Sarimveis, H., Kiranoudis, C., Alexandridis, A., & Bafas, G.
(2004). A GIS based operational system for wildland fire crisis
management I: Mathematical modelling and simulation. _Applied
Mathematical Modelling, 28_ (4), 389–410.
https://doi.org/10.1016/j.apm.2003.10.005
He, W., & Weng, Q. (2025). Disparities of urban morphology effects on
compound natural risks: A multiscale study across the USA. _npj
Urban Sustainability, 5_ , Article 39.
https://doi.org/10.1038/s42949-025-00233-9
Moreira, S. (2020, November 24). _Radial City Plan: Nine Examples
Around the World Seen From Above_ (T. Duduch, Trans.). ArchDaily.
https://www.archdaily.com/951587/radial-city-plan-nine-examples-ar
ound-the-world-seen-from-above (archdaily.com)
Stevens, S., & Rush, D. (2025). Urban fire spread modelling: a review of
dynamic computational models and potential for application to
informal settlement fires. _International Journal of Disaster Risk
Reduction_ , _124_ , 105528. https://doi.org/10.1016/j.ijdrr.2025.105528
StudySmarter Editorial Team. (2024, August 9). _Urban grid: Architecture &
design principles_. StudySmarter.
https://www.studysmarter.co.uk/explanations/architecture/urban-des
ign-in-architecture/urban-grid/ (studysmarter.co.uk)


Shaham, Y., & Benenson, I. (2018). Modeling fire spread in cities with
non-flammable construction. International Journal of Disaster Risk
Reduction, 31, 1337-1353.
https://doi.org/10.1016/j.ijdrr.2018.03.010
Freire, J. G., & DaCamara, C. C. (2019). Using cellular automata to
simulate wildfire propagation and to assist in fire management.
_Natural Hazards and Earth System Sciences_ , _19_ (1), 169–179.
https://doi.org/10.5194/nhess-19-169-2019
Karafyllidis, I., & Thanailakis, A. (1997). A model for predicting forest fire
spreading using cellular automata. _Ecological Modelling, 99_ (1),
87–97. https://doi.org/10.1016/S0304-3800(96)01942-4
Hernández Encinas, A., Hernández Encinas, L., Hoya White, S., Martín
del Rey, A., & Rodríguez Sánchez, G. (2007). Simulation of forest
fire fronts using cellular automata. _Advances in Engineering
Software, 38_ (6), 372–378.
https://doi.org/10.1016/j.advengsoft.2006.09.002
Patac, J. C. J., & Vicente, A. J. O. (2019). Urban fire spread modelling and
simulation using cellular automaton with extreme learning machine.
_The International Archives of the Photogrammetry, Remote Sensing
and Spatial Information Sciences, XLII_ ‑ _4/W19_ , 319.
https://doi.org/10.5194/isprs-archives-XLII-4-W19-319-2019
(isprs-archives.copernicus.org)
Sullivan, A. L. (2009). Wildland surface fire spread modelling, 1990 - 2007.
2: Empirical and quasi-empirical models. _International Journal of
Wildland Fire_ , _18_ (4), 369. https://doi.org/10.1071/wf06142
Collin, A., Bernardin, D., & Séro-Guillaume, O. (2011). A Physical-Based
cellular automaton model for Forest-Fire propagation. _Combustion
Science and Technology_ , _183_ (4), 347–369.
https://doi.org/10.1080/00102202.2010.508476
Zheng, Z., Huang, W., Li, S., & Zeng, Y. (2017). Forest fire spread
simulating model using cellular automaton with extreme learning


machine. _Ecological Modelling, 348_ , 33–43.
https://doi.org/10.1016/j.ecolmodel.2016.12.022
Khanmohammadi, S., Arashpour, M., Mohammadi Golafshani, E., Cruz,
M. G., Rajabifard, A., & Bai, Y. (2022). Prediction of wildfire rate of
spread in grasslands using machine learning methods.
_Environmental Modelling & Software, 156_ , Article 105507.
https://doi.org/10.1016/j.envsoft.2022.105507
Marjani, M., & Mesgari, M. S. (2023). THE LARGE-SCALE WILDFIRE
SPREAD PREDICTION USING a MULTI-KERNEL
CONVOLUTIONAL NEURAL NETWORK. _ISPRS Annals of the
Photogrammetry, Remote Sensing and Spatial Information
Sciences_ , _X-4/W1-2022_ , 483–488.
https://doi.org/10.5194/isprs-annals-x-4-w1-2022-483-2023
Radke, D., Hessler, A., & Ellsworth, D. (2019). _FireCast: Leveraging deep
learning to predict wildfire spread_. IJCAI.
https://www.ijcai.org/proceedings/2019/636
Wolfram, S. (1984). Cellular automata as models of complexity. _Nature_ ,
_311_ (5985), 419–424. https://doi.org/10.1038/311419a0
Breiman, L. (2001). Random Forest. _Machine Learning_ , _45_ (1), 5–32.
https://doi.org/10.1023/a:1010933404324.
Mitchell, T. M. (1997). _Machine Learning_. McGraw‑Hill.
Rothermel, R. C. (1972). _A mathematical model for predicting fire spread
in wildland fuels_. US Forest Service Research and Development.
https://research.fs.usda.gov/treesearch/32533.
Bergmann, D., & Stryker, C. (2025, April 17). Model training. _What is
Model Training?_ https://www.ibm.com/think/topics/model-training
Ashraf, I., Hurrah, N. N., Sofi, S. A., Sharma, A., Panesar, G. S., & Reshi,
Z. A. (2025). A comprehensive review of empirical and dynamic
wildfire simulators and machine learning techniques used for the
prediction of wildfire in Australia. _Technology, Knowledge and
Learning_ , 1-38. https://doi.org/10.1007/s10758-024-09755-w


Bamdale, V., Sharma, A., & Tiwari, P. (2021). Wildfire risk assessment
using machine learning techniques: A review. _Journal of Forestry
Research_ , 32(2), 405-418.
https://doi.org/10.1007/s11676-020-01124-4
Costafreda-Aumedes, S., Comas, C., & Vega-Garcia, C. (2018). Predictive
modeling of wildfires: A new dataset and machine learning
approach. _Fire Safety Journal_ , 104, 130-146.
https://doi.org/10.1016/j.firesaf.2018.11.002
Liang, H., Zhang, M., & Wang, H. (2022). A forest fire prediction model
based on cellular automata and machine learning. _IEEE Access_ ,
10, 45834-45843. https://doi.org/10.1109/ACCESS.2022.3171310
Moghim, S., & Mehrabi, M. (2024). Wildfire assessment using machine
learning algorithms in different regions. Fire Ecology, 20(1).
https://doi.org/10.1186/s42408-024-00335-2
Shahzad, M., Baig, M. H. A., Ur Rehman, A., Tariq, A., & Ul-Haq, Z.
(2024). Comparing machine learning algorithms to predict
vegetation fire detections in Pakistan. _Fire Ecology_ , 20(1), 1-16.
https://doi.org/10.1186/s42408-024-00250-0
Ohgai, A., Gohnai, Y., Ikaruga, S., Murakami, M., & Watanabe, K. (2004).
Cellular Automata Modeling for fire spreading as a tool to aid
Community-Based Planning for disaster mitigation. In _Springer
eBooks_ (pp. 193–209). https://doi.org/10.1007/1-4020-2409-6_13
Gao, N., Weng, W., Ma, W., Ni, S., Huang, Q., & Yuan, H. (2008). Fire
spread model for old towns based on cellular automaton. _Tsinghua
Science and Technology, 13_ (4), 566–572.
Chavez, C. (2025, January 4). BFP: Nationwide 2024 fire incidents up by
1,823 from previous year. Manila Bulletin.
https://mb.com.ph/5/1/2025/bfp-nationwide-2024-fire-incidents-up-b
y-1-823-from-previous-year


Takizawa, A., et al.,(2000). Simulation of spreads of fire on city site by
stochastic cellular automata. Retrieved from
[http://www.iitk.ac.in/nicee/wcee/article/2334.pdf](http://www.iitk.ac.in/nicee/wcee/article/2334.pdf)
Alexandridis, A., Russo, L., Vakalis, D., Bafas, G. V., & Siettos, C. I.
(2011). Wildland fire spread modelling using cellular automata:
evolution in large-scale spatially heterogeneous environments
under fire suppression tactics. International Journal of Wildland
Fire, 20(5), 633. https://doi.org/10.1071/wf091 19


### APPENDICES


```
Appendix A
```
**Transmittal Letter**



```
Appendix B
Software Requirements Specifications
```
**1. Introduction**

**1.1 Purpose**

The purpose of this system is to provide a web-based fire spread
simulation application that models fire propagation across urban layouts
using cellular automata integrated with machine learning techniques.
Testing is focused on Lapu-Lapu City but the approach is replicable to
other urban areas with sufficient data.

**1.2 Scope**
This application demonstrates a novel fire spread prediction model
based on cellular automata and machine learning algorithms. The system
uses Random Forest and Logistic Regression algorithms to calculate
ignition probabilities, then simulates fire spread through a cellular
automata framework. The frontend displays the fire spread simulation
interactively with real-time visualization.

**2. Overall Description**

**2.1 Product Perspective**
This system is a standalone, browser-accessible web application. It
features a client-server architecture where the client provides simulation
parameters and the server processes the fire spread simulation using
cellular automata integrated with machine learning models.

**2.2 Product Functions**
● Accept user input for barangay selection and simulation
parameters.
● Load urban layout data from OpenStreetMap.
● Allow users to set ignition points by clicking on buildings.
● Configure wind speed, direction, and simulation duration.
● Process fire spread using cellular automata with 3×3 meter grid
resolution.
● Update visualization in real-time with color-coded building states.
● Generate simulation statistics and export results.

**2.3 User Characteristics**


Intended users are researchers, urban planners, and emergency response
personnel. No advanced technical knowledge is required.

**2.4 Assumptions and Dependencies**

**Assumptions:**
● OpenStreetMap data for urban layouts is sufficiently detailed and
current.
● Wind patterns can be adequately modeled using meteorological
data.
**Dependencies:**
● Availability of OpenStreetMap data for target urban areas.
● Machine learning libraries (scikit-learn) for Random Forest and
Logistic Regression.
● Leaflet library for frontend visualization.
● Stable internet connection for loading maps and processing
simulations.

**3. Specific Requirements**

**3.1 Functional Requirements**
● **FR1** : The system shall accept barangay selection and simulation
parameters.
● **FR2** : The system shall process fire spread using cellular automata
with machine learning.
● **FR3** : The system shall visualize fire spread with real-time
color-coded updates.
● **FR4** : The system shall provide simulation controls and export
capabilities.

**3.2 Non-Functional Requirements**
● **NFR1** : The system shall work on modern browsers.
● **NFR2** : The system shall complete simulations within 30 seconds.
● **NFR3** : The system shall handle invalid inputs with error messages.

**3.3 System Architecture Overview**
● **Frontend** : Built with React and Leaflet; handles user interaction
and visualization.
● **Backend** : Built with Python and scikit-learn; processes simulation
requests.


**4. External Interface Requirements**

**4.1 User Interface**

```
● A Leaflet map interface with building layouts.
● Dropdown menu for barangay selection.
● Parameter controls for wind speed and direction.
● Simulation control buttons.
```
**4.2 Software Interfaces**

```
● OpenStreetMap via API for urban layout data
● Machine learning libraries (scikit-learn)
● Leaflet mapping library
```

**Appendix C**

```
Website UI
```


### CURRICULUM VITAE

### CONTACT INFORMATION

Name: Kristian Lemuel W. Diaz
Address: Small Rd. Sangi St., Pajo,
Lapu-Lapu City Cebu
Telephone: 032 3838295
Cellphone: +63 967 651 8702
Email: diazkristian321@gmail.com

**PERSONAL INFORMATION**
Birthday: May 10, 2004
Religion: Roman Catholic
Civil Status: Single

**EDUCATION**
University of San Carlos - Talamban Campus
Bachelor of Science in Computer Science
Tertiary Level (2022 - Present)

Indiana Aerospace University
Senior High School (2020 - 2022)

Dr. Caridad C. Labe Center of Excellence, Inc.
Secondary Level (2016 - 2020)

St. Jerome Integrated School of Cabuyao, Inc.
Primary Level (2007 - 2016)

**TECHNICAL SKILLS**
● Programming Languages: HTML/CSS, Javascript, C, Java, Python,
PHP, SQL.


● Tools & Platforms: Git, GitHub, Visual Studio Code, Cursor,
MongoDB, MySQL, Node.js.
● Framework & Libraries: React & Express.js.

**CERTIFICATION
TRAININGS**
Google Developer Student Clubs San Carlos (Human Relation
Committee)
Creative Web Design


### CONTACT INFORMATION

Name: Kent Anthony C. Dulangon
Address: Phase 2 Block 1 Lot 22 Collinwood
Subdivision Basak Lapu-Lapu City, Cebu
Telephone: 494 3460
Cellphone: 0943 502 7835
Email: dkentanthony@gmail.com

**PERSONAL INFORMATION**
Birthday: June 14, 2003
Religion: Roman Catholic
Civil Status: Single

**EDUCATION**
University of San Carlos - Talamban Campus
Bachelor of Science in Computer Science
Tertiary Level ( 2022- present )

CCL CentrEx
Senior High School ( 2020-2022 )

St.Alphonsus Catholic School
Secondary Level ( 2018-2020 )

Science and Technology Education Center
Secondary Level ( 2016-2018 )

Science and Technology Education Center
Early Education - Primary Level ( 2008 - 2016 )

**TECHNICAL SKILLS**


● Programming Languages: C, Java, Python, HTML/CSS, React,
Javascript, SQL.
● Tools & Platforms: Git, GitHub, Visual Studio Code, MongoDB,
MySQL, Node.js, Android Studio (Java)
● Frameworks & Libraries: React, Node.js

**TRAININGS**
Google Developer Student Clubs San Carlos
Creative Web Design
README
Overview

This repository contains the full simulation code used to investigate the evolutionary emergence of hemispheric specialization under different environmental structures.

The simulations compare two environmental conditions:

Spatial environment: containing multiscale global–local structure

Planar environment: lacking multiscale spatial structure (control)

All non-environmental factors (random seeds, evolutionary algorithm, population size, mutation schedule) are strictly matched between conditions. Hemispheric specialization emerges solely through interaction with environmental structure.

Files
hemispheric_specialization.py

This script implements the main lineage-level evolutionary analysis reported in the paper.

It performs the following steps:

Defines a minimal two-processor model with independent left and right spatial integration scales.

Generates spatially structured and planar sensory environments.

Evolves populations under each environment using identical random seeds.

Evaluates hemispheric specialization as
|aR − aL| for the best-performing individual of each lineage.

Outputs lineage-level results used for:

Final hemispheric asymmetry statistics

Paired spatial vs planar comparisons

Figure 1 and Figure 2 in the manuscript

This script corresponds to the primary analysis described in the Methods and Results sections.

run_generation_sim.py

This script performs the best-of-generation evolutionary trajectory analysis.

It tracks:

The best-performing individual at each generation

The evolution of hemispheric asymmetry across generations

Differences between spatial and planar environments over time

Key features:

Identical evolutionary parameters to the main simulation

Best-of-generation selection rather than population averages

Independent test evaluation using new random seeds

The output of this script is used to generate Figure 3, illustrating how hemispheric asymmetry is retained among optimal performers despite overall convergence toward symmetry.

Reproducibility

All simulations are fully deterministic given the specified random seeds.

Spatial and planar conditions are always paired using identical seeds.

No manual intervention or post hoc filtering is applied.

Running the scripts as provided reproduces all quantitative results reported in the manuscript.

Computational scope

The model is intentionally minimal and abstract.
It is not intended as a biologically realistic neural simulation, but as a computational proof-of-principle isolating environmental structure as a sufficient driver of hemispheric specialization.

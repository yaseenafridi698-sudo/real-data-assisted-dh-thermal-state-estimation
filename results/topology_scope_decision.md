# Topology Scope Decision

## Available evidence

- The public Sonderborg, Flensburg, and XAI4HEAT records used here do not provide a verified pipe graph aligned with the retained temperature and load timestamps.
- The distributed benchmark is a declared 20 km unbranched supply/return corridor with one aggregate consumer.
- The existing graph/no-graph ablation has mixed rankings: removing graph propagation lowers several direct errors, while the graph variant is lower for selected flow and boundary indicators.

## Decision

The manuscript does not claim graph superiority or utility-topology validation. The PI-GNN-GRU architecture is described as topology-capable, and the title emphasizes thermal state estimation with simulator-assisted hydraulics rather than graph novelty.

A separate branched network was not added merely to create a favourable graph result. A defensible cross-topology study must use a recognized benchmark with published topology, boundary conditions, and independently reproducible reference states, followed by common-protocol retraining of graph and non-graph estimators. This remains a defined next validation step.

## Safe interpretation

The current corridor isolates sparse-sensor transport and heat-loss reconstruction. It cannot determine whether graph message passing improves estimation on branched or meshed district-heating networks.

# Traffic Light Signalling Control Optimization

This is my take on optimizing traffic light controls as a discrete type problem with genetic algorithms.

## Setup & Usage

Setup and run instructions live in the `src/docs/` folder. Pick the guide that matches your environment:

- [Docker setup](src/docs/docker_setup.md) — build and run everything inside Docker.
- [Linux setup](src/docs/linux_setup.md) — run directly with Python 3 on Linux (no Docker).

## Algorithm & Pipeline Docs

- [Pipeline — Custom Optimizer (LT-OM Algorithm)](src/docs/pipeline_custom_optimizer.md) — end-to-end walkthrough of the Custom Linkage-Tree Optimal Mixing optimizer.
- [Pipeline — SHADE with Cluster Crossover (v3)](src/docs/pipeline_differential_evolution_cluster_v3.md) — end-to-end walkthrough of SHADE with cluster crossover.
- [Novel Bin-Crossing Crossover (v3)](src/docs/novel_bin_crossing_v3.md) — the walk-decomposition cluster crossover used in place of SHADE's binomial crossover.
- [Traffic Light Normalisation & Pair-wise Mutation](src/docs/traffic_light_normalization.md) — how raw gene values become a valid 90 s cycle, and how pair-wise mutation works.

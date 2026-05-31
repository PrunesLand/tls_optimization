# Linux Setup

Instructions for running the project directly with Python 3 on Linux (no Docker).

## Prerequisites

Before following the steps below, install the system dependencies, set up SUMO,
and install the Python requirements.

1. **Install system dependencies (requires sudo)**

    Installs the libraries required by the libsumo C++ extension as well as SUMO
    and its tools.

    ```bash
    apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        libatomic1 \
        libfontconfig1 \
        sumo \
        sumo-tools \
        && rm -rf /var/lib/apt/lists/*
    ```

2. **Set the `SUMO_HOME` environment variable**

    Point `SUMO_HOME` to the installed SUMO directory.

    ```bash
    export SUMO_HOME=/usr/share/sumo
    ```

3. **Install Python dependencies**

    ```bash
    pip install --no-cache-dir -r requirements.txt
    ```

## How to run

1. **Run GA with PyGad**

    This will run the PyGad Implementation of the Genetic Algorithm.

    ```bash
    python3 -m src.algorithms.simple_genetic_algorithm
    ```

2. **View map statistics**

    Will display statistics of the downloaded map and traffic network

    ```bash
    python3 -m src.sumo_setup.statistics
    ```

3. **Generate network data**

    This will generate network data that will have phase durations assigned for individual TLS. This is required step to run the optimization algorithm.

    ```bash
    python3 -m src.sumo_setup.generation
    ```

4. **Generate map**

    This will generate a new map following `osm.netccfg` configurations.

    ```bash
    cd src/sumo_setup && netconvert -c osm.netccfg
    ```

5. **Configure Simulator variables**

    Configure SUMO Simulator variables when running every simulation. We are now using precalculated routes. this step is not necessary.

    ```bash
    cd src/sumo_setup && python3 $SUMO_HOME/tools/randomTrips.py -n osm.net.xml.gz -o [name of your routes file].rou.xml
    ```

    You can change from random trips to a specific configuration such as setting the specific number of cars generated per second or setting the total number of cars within every simulation.

6. **Discover TLS linkage**

    This will discover linkage of TLS by Direct Linkage Empirical Discovery.

    ```bash
    python3 -m src.decomposition.dled_optimizer
    ```

7. **Execute DG2 Grouping**

    This will execute Differential Grouping method as an alternative to linkage discovery. Theoretically it is faster than Embpirical Linkage Learning.

    ```bash
    python3 -m src.decomposition.DG2_grouping
    ```

8. **Execute IRRG**

    ```bash
    python3 -m src.decomposition.IRRG
    ```

9. **Execute Random Search**

    Performs random search of n solutions and m evaluations for each solutions.

    ```bash
    python3 -m src.algorithms.random_search
    ```

10. **Execute Custom Optimizer**

    Runs the Custom Optimizer algorithm which was implemented based on LT-GOMEA. 

    ```bash
    python3 -m src.algorithms.custom_optimizer
    ```

11. **Generate Distance Matrices**

    Calculates the network distance matrices (Shortest, Euclidian, Fastest) used by the optimizer.

    ```bash
    python3 -m src.plot.tls_distances
    ```

12. **Generate Clustering Dendrograms**

    Generates hierarchical clustering dendrogram plots to visualize the linkage tree structure.

    ```bash
    python3 -m src.plot.plot_dendrograms
    ```

13. **Analyze Linkage Statistics**

    Analyzes the calculated distance matrices to determine the optimal clustering thresholds, calculating max valid non-singleton clusters and the median cluster size for each distance metric.

    ```bash
    python3 -m src.algorithms.linkage_analyzer
    ```

14. ***Run Baseline Configuration***

    Evaluates the current configuration of the instance map.

    ```bash
    python3 -m src.algorithms.evaluate_baseline
    ```

15. **Run LT-GA with PyGad**

    This will run the PyGad Implementation of the custom LT - Genetic Algorithm.

    ```bash
    python3 -m src.algorithms.lt_genetic_algorithm
    ```

16. **Run Differential Evolution (SHADE)**

    Runs the EvoX SHADE implementation on the baseline TLS configuration.
    Behavior depends on `NOVEL_MUTATION` in `config.py`: when `False`, a single
    plain SHADE run is performed; when `True`, three runs are executed (one per
    Ward distance tree: shortest / euclidian / fastest) with end-of-generation
    pair-cluster mutation applied to a `MUTATION_RATE` fraction of the
    population. Requires step 11 (distance matrices) when `NOVEL_MUTATION=True`.

    ```bash
    python3 -m src.algorithms.differential_evolution
    ```

## Quick Start Pipeline (Rental Machine Setup)

If you would like to setup the whole pipeline, please complete the prerequisites
above, then run steps 3, 4, 11, and 10 in order.

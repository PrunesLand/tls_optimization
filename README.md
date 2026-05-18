# Traffic Light Signalling Control Optimization

This is my take on optimizing traffic light controls as a discrete type problem with genetic algorithms.

## Optimization process



## Quick Start Pipeline (Rental Machine Setup)

If you need to quickly run the full setup and LT-GOMEA optimization pipeline (Steps 1, 5, 6, 13, and 12), you can use the automated bash script provided in the root directory:

**Using the Bash Script:**
```bash
./run_pipeline.sh
```

**Using the Python Script:**
```bash
python run_pipeline.py
```

**Using a Single Docker One-Liner (if you prefer no external scripts):**
```bash
docker build -t tls_optimization . && docker run --rm -v $(pwd):/app -w /app tls_optimization bash -c "python -m src.sumo_setup.generation && cd src/sumo_setup && netconvert -c osm.netccfg && cd /app && python -m src.pygad.tls_distances_shortest && python -m src.pygad.tls_distances_euclidian && python -m src.pygad.tls_distances_fastest && python -m src.pygad.plot_dendrograms && python -m src.pygad.lt_gomea_optimizer"
```

## How to run

1. **Building with Docker**

    This will build the project with Docker. This must be done before running the program.

    ```bash
    docker build -t tls_optimization .
    ```
2. **Running with Docker**

    This will run the project with Docker.

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization
    ```

3. **Run GA with PyGad**

    This will run the PyGad Implementation of the Genetic Algorithm.

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.simple_genetic_algorithm
    ```

4. **View map statistics**

    Will display statistics of the downloaded map and traffic network

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.sumo_setup.statistics
    ```
    
5. **Generate network data**

    This will generate network data that will have phase durations assigned for individual TLS. This is required step to run the optimization algorithm.

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.sumo_setup.generation
    ```
6. **Generate map**

    This will generate a new map following `osm.netccfg` configurations.

    ```bash
    docker run --rm -v $(pwd):/app -w /app/src/sumo_setup tls_optimization netconvert -c osm.netccfg
    ```

7. **Configure Simulator variables**

    Configure SUMO Simulator variables when running every simulation. We are now using precalculated routes. this step is not necessary.

    ```bash
    docker run --rm -v $(pwd):/app -w /app/src/sumo_setup tls_optimization bash -c 'python $SUMO_HOME/tools/randomTrips.py -n osm.net.xml.gz -o [name of your routes file].rou.xml'
    ```

    You can change from random trips to a specific configuration such as setting the specific number of cars generated per second or setting the total number of cars within every simulation.

8. **Discover TLS linkage**

    This will discover linkage of TLS by Direct Linkage Empirical Discovery.

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.decomposition.dled_optimizer
    ```
9. **Execute DG2 Grouping**

    This will execute Differential Grouping method as an alternative to linkage discovery. Theoretically it is faster than Embpirical Linkage Learning.

   ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.decomposition.DG2_grouping
   ```
10. **Execute IRRG**

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.decomposition.IRRG
    ```

11. **Execute Random Search**

    Performs random search of n solutions and m evaluations for each solutions.

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.random_search
    ```

12. **Execute LT-GOMEA Optimizer**

    Runs the Linkage Tree Gene-pool Optimal Mixing Evolutionary Algorithm (LT-GOMEA). Uses threshold-based clusters from distance matrices to guide the optimal mixing operator. This will execute the entire 9-run experiment matrix (3 trees × 3 population strategies).

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.lt_gomea_optimizer
    ```



13. **Generate Distance Matrices & Dendrograms**

    Calculates the network distance matrices (Shortest, Euclidian, Fastest) and generates hierarchical clustering dendrogram plots used by the LT-GOMEA optimizer.

    ```bash
    # Generate the distance matrices
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.tls_distances_shortest
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.tls_distances_euclidian
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.tls_distances_fastest

    # Plot the dendrograms
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.plot_dendrograms
    ```

14. **Analyze Linkage Statistics**

    Analyzes the calculated distance matrices to determine the optimal clustering thresholds, calculating max valid non-singleton clusters and the median cluster size for each distance metric.

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.linkage_analyzer
    ```

## Docker cleaning commands

Docker has build-in commands that are ment to be used for house keeping tasks:
- `docker image prune`: delete all dangling images (as in without an assigned tag)
- `docker image prune -a`: delete all images not used by any container
- `docker system prune`: delete stopped containers, unused networks and dangling image + dangling build cache
- `docker system prune -a`: delete stopped containers, unused networks, images not used by any container + all build cache


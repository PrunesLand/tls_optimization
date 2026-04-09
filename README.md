# Traffic Light Signalling Control Optimization

This is my take on optimizing traffic light controls as a discrete type problem with genetic algorithms.

## Optimization process



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
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.pygad_genetic_algorithm
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
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.dled_optimizer
    ```

## Docker cleaning commands

Docker has build-in commands that are ment to be used for house keeping tasks:
- `docker image prune`: delete all dangling images (as in without an assigned tag)
- `docker image prune -a`: delete all images not used by any container
- `docker system prune`: delete stopped containers, unused networks and dangling image + dangling build cache
- `docker system prune -a`: delete stopped containers, unused networks, images not used by any container + all build cache


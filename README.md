# Traffic Light Signalling Control Optimization

This is my take on optimizing traffic light controls as a discrete type problem with genetic algorithms.

## How to run

1. **Building with Docker**

    This will build the project with Docker. This must be done before running the program.

    ```bash
    docker build -t tls_optimization .
    ```
2. **Running with Docker**

    This will run the project with Docker.

    ```bash
    docker run --rm tls_optimization
    ```

3. **View map statistics**

    Will display statistics of the downloaded map and traffic network

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.sumo_setup.statistics
    ```
    
4. **Generate network data**

    This will generate network data that will have phase durations assigned for individual TLS. This is required step to run the optimization algorithm.

    ```bash
    docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.sumo_setup.generation
    ```
5. **Generate map**

    This will generate a new map following `osm.netccfg` configurations.

    ```bash
    docker run --rm -v $(pwd):/app -w /app/src/sumo_setup tls_optimization netconvert -c osm.netccfg
    ```

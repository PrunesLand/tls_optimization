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
    docker run --rm -w /app/src/sumo_setup tls_optimization python statistics.py
    ```


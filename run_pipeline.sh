#!/bin/bash
set -e

echo "=================================================="
echo "Running: 1. Building with Docker"
echo "=================================================="
docker build -t tls_optimization .

echo "=================================================="
echo "Running: 5. Generate network data"
echo "=================================================="
docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.sumo_setup.generation

echo "=================================================="
echo "Running: 6. Generate map"
echo "=================================================="
docker run --rm -v $(pwd):/app -w /app/src/sumo_setup tls_optimization netconvert -c osm.netccfg

echo "=================================================="
echo "Running: 13. Generate Distance Matrices & Dendrograms"
echo "=================================================="
docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.tls_distances_shortest
docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.tls_distances_euclidian
docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.tls_distances_fastest
docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.plot_dendrograms

echo "=================================================="
echo "Running: 12. Execute LT-GOMEA Optimizer"
echo "=================================================="
docker run --rm -v $(pwd):/app -w /app tls_optimization python -m src.pygad.lt_gomea_optimizer

echo "=================================================="
echo "Pipeline completed successfully!"
echo "=================================================="

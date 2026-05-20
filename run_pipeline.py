import subprocess
import os

def run_command(cmd, cwd=None):
    print(f"\n[{'='*50}]")
    print(f"Running: {cmd}")
    print(f"[{'='*50}]\n")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        exit(result.returncode)

if __name__ == "__main__":
    # Note: This script assumes you are running inside the Docker container
    # or have all dependencies and SUMO installed locally.
    
    # 5. Generate network data
    run_command("python -m src.sumo_setup.generation")
    
    # 6. Generate map
    run_command("netconvert -c osm.netccfg", cwd="src/sumo_setup")
    
    # 13. Generate Distance Matrices & Dendrograms
    run_command("python -m src.pygad.tls_distances_shortest")
    run_command("python -m src.pygad.tls_distances_euclidian")
    run_command("python -m src.pygad.tls_distances_fastest")
    run_command("python -m src.pygad.plot_dendrograms")
    
    # 12. Execute Custom Optimizer
    run_command("python -m src.pygad.custom_optimizer")

#!/bin/bash
set -e

# =============================================================================
# TLS Optimization - Docker Entrypoint
# =============================================================================
# This script orchestrates the full pipeline for running the TLS optimization
# on a fresh machine (e.g. Vast.ai). It handles:
#   1. Network data generation
#   2. Map generation with netconvert
#   3. Running the IRRG optimization algorithm
#
# Usage:
#   ./entrypoint.sh [COMMAND]
#
# Commands:
#   setup       - Run steps 1-2 only (generation + netconvert)
#   run         - Run step 3 only (IRRG), assumes setup was already done
#   all         - Run the full pipeline (setup + run) [default]
#   <other>     - Pass through any arbitrary command (e.g. bash, python ...)
# =============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_step() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  STEP: $1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ---------------------------------------------------------------------------
# Step 1: Generate network data (phase durations for individual TLS)
# ---------------------------------------------------------------------------
run_generation() {
    log_step "Generating network data"
    python -m src.sumo_setup.generation
    log_info "Network data generation complete."
}

# ---------------------------------------------------------------------------
# Step 2: Generate the map using netconvert
# ---------------------------------------------------------------------------
run_netconvert() {
    log_step "Generating map with netconvert"
    cd /app/src/sumo_setup
    netconvert -c osm.netccfg
    cd /app
    log_info "Map generation complete."
}

# ---------------------------------------------------------------------------
# Step 3: Run the IRRG optimization algorithm
# ---------------------------------------------------------------------------
run_irrg() {
    log_step "Running IRRG optimization"
    python -m src.pygad.IRRG
    log_info "IRRG optimization complete."
}

# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
run_setup() {
    run_generation
    run_netconvert
}

run_all() {
    log_step "Starting full TLS optimization pipeline"
    run_setup
    run_irrg
    log_info "Full pipeline finished successfully."
}

# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
COMMAND="${1:-all}"

case "$COMMAND" in
    setup)
        log_info "Running setup only (generation + netconvert)..."
        run_setup
        ;;
    run)
        log_info "Running IRRG optimization only (skipping setup)..."
        run_irrg
        ;;
    all)
        log_info "Running full pipeline..."
        run_all
        ;;
    generation)
        run_generation
        ;;
    netconvert)
        run_netconvert
        ;;
    *)
        # Pass through arbitrary commands (e.g. bash, python script.py, etc.)
        log_info "Executing custom command: $@"
        exec "$@"
        ;;
esac

"""
Fitness evaluation for traffic light optimization via SUMO.

Contains:
- fitness_function        : runs a SUMO simulation for a given TLS duration dict
- evaluate_individual     : evaluates a single individual
- evaluate_population     : evaluates an entire population
- TrafficFitnessWrapper   : converts a flat gene vector → per-TLS durations → SUMO
- build_traffic_fitness_wrapper : builds a picklable wrapper from baseline data

All algorithms should import fitness_function and build_traffic_fitness_wrapper
from this module.
"""

import json
import copy
import sys
from pathlib import Path

import numpy as np
import libsumo as traci

# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import SUMO_ARGS, BASELINE_TRAFFIC_DATA, PHASE_BOUNDS, CYCLE_LENGTH

with open(BASELINE_TRAFFIC_DATA, 'r') as f:
    BASELINE_DATA = json.load(f)



# ── SUMO fitness function ───────────────────────────────────────────────────

def fitness_function(tls_durations):
    """
    Evaluates a lightweight dictionary of {tls_id: [durations]}
    """
    traci.start(SUMO_ARGS)

    # 2. Iterate through the lightweight dictionary provided by PyGAD
    for tl_id, durations in tls_durations.items():
        phase_list = []
        
        # Look up the baseline phases for this specific traffic light
        baseline_phases = BASELINE_DATA["tls_data"][tl_id]
        phase_keys = sorted(baseline_phases.keys())

        # Combine the lightweight duration with the baseline state string
        for i, phase_key in enumerate(phase_keys):
            duration = int(durations[i])
            state = baseline_phases[phase_key]["state"] 
            
            # Create the phase object for SUMO
            phase_list.append(traci.trafficlight.Phase(duration, state))

        # Assign the logic to the simulator
        logic = traci.trafficlight.Logic("custom", 0, 0, phase_list)
        traci.trafficlight.setProgramLogic(tl_id, logic)
        traci.trafficlight.setProgram(tl_id, "custom")

    # 3. Run the simulation and calculate costs
    total_delay = 0.0
    total_vehicles = 0

    simulation_steps = int(traci.simulation.getEndTime())

    for step in range(simulation_steps):
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        for vid in vehicle_ids:
            speed = traci.vehicle.getSpeed(vid)
            max_speed = traci.vehicle.getMaxSpeed(vid)
            if max_speed > 0:
                total_delay += (1.0 - speed / max_speed)

        total_vehicles = max(total_vehicles, len(vehicle_ids))

    traci.close()

    # Calculate final fitness score
    fitness = total_delay + (total_vehicles * 10)
    return fitness


def evaluate_individual(individual):
    fitness = fitness_function(individual)

    evaluated_individual = copy.deepcopy(individual)
    evaluated_individual["fitness"] = fitness

    return evaluated_individual

def evaluate_population(population):
    evaluated_population = []
    for individual in population:
        evaluated_individual = evaluate_individual(individual)
        evaluated_population.append(evaluated_individual)

    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / "population_tls_data_with_fitness.json"
    
    with open(json_filepath, "w") as f:
        json.dump(evaluated_population, f, indent=4)
    
    return evaluated_population


# ── Phase-type inference ─────────────────────────────────────────────────────

def phase_type(state: str) -> str:
    """Infer phase type (green/yellow/red) from a SUMO TLS state string."""
    s = state.lower()
    counts = {"green": s.count("g"), "yellow": s.count("y"), "red": s.count("r")}
    for ptype in ("green", "yellow", "red"):
        if counts[ptype] == max(counts.values()):
            return ptype
    return "red"


# ── Dynamic per-TLS green/red ceiling ───────────────────────────────────────

def phase_upper_bounds(phase_types, durations, cycle_length=CYCLE_LENGTH):
    """
    Per-phase upper bound for one TLS within a fixed ``cycle_length`` cycle.

    For each non-yellow (green/red) phase, the ceiling is the most that phase
    can occupy while every *other* non-yellow phase sits at its per-type
    minimum and the (frozen) yellow phases keep their current duration::

        ceiling_j = cycle_length - Σ(yellow durations)
                                 - Σ(min of every OTHER non-yellow phase)

    where each non-yellow minimum comes from ``PHASE_BOUNDS[ptype][0]``
    (green=24, red=5).  Yellow phases keep their static upper bound
    (``PHASE_BOUNDS["yellow"][1]``): they are frozen and clamped to ``[3, 6]``
    elsewhere, so their ceiling does not depend on the cycle.

    Examples
    --------
    3-phase ``G, y, r`` with yellow=6 → green ceiling 90-6-5 = 79,
    red ceiling 90-6-24 = 60.
    4-phase ``G, y, G, y`` with yellows 6+6 → each green 90-12-24 = 54.
    """
    yellow_sum = sum(d for d, pt in zip(durations, phase_types) if pt == "yellow")
    nonyellow = [i for i, pt in enumerate(phase_types) if pt != "yellow"]
    mins = {i: PHASE_BOUNDS[phase_types[i]][0] for i in nonyellow}

    bounds = []
    for i, ptype in enumerate(phase_types):
        if ptype == "yellow":
            bounds.append(float(PHASE_BOUNDS[ptype][1]))
            continue
        others_min = sum(mins[j] for j in nonyellow if j != i)
        bounds.append(float(cycle_length - yellow_sum - others_min))
    return bounds


# ── Cycle-length normalisation (shared between eval-time and save-time) ─────

def normalize_to_cycle(raw_durations, phase_types, cycle_length=CYCLE_LENGTH,
                       upper_bounds=None):
    """
    Clamp each phase duration to its per-type bounds (PHASE_BOUNDS), then
    bring the total to ``cycle_length`` by absorbing the remainder into the
    smallest green/red phase.  If that would push the target below its
    minimum, the absorption falls back to the largest green/red phase.

    When ``upper_bounds`` is given (per-phase, from :func:`phase_upper_bounds`),
    each phase's clamp ceiling is tightened to ``min(PHASE_BOUNDS hi, bound)``.
    The remainder absorption already respects that ceiling by construction: a
    phase only grows when it is the *smallest* adjustable one, and the most it
    can reach is exactly ``cycle_length - yellows - other mins`` — the bound.

    Used by both the fitness wrapper (so SUMO sees a fixed cycle) and by
    _rebuild_json (so the saved JSON matches what was simulated).
    """
    durations = []
    for i, (raw_val, ptype) in enumerate(zip(raw_durations, phase_types)):
        lo, hi = PHASE_BOUNDS[ptype]
        if upper_bounds is not None:
            hi = min(hi, upper_bounds[i])
        durations.append(int(round(max(lo, min(hi, raw_val)))))

    remainder = cycle_length - sum(durations)
    if remainder != 0:
        adjustable = [i for i, pt in enumerate(phase_types) if pt in ("green", "red")]
        if adjustable:
            target_idx = min(adjustable, key=lambda i: durations[i])
            durations[target_idx] += remainder

            lo, _ = PHASE_BOUNDS[phase_types[target_idx]]
            if durations[target_idx] < lo:
                durations[target_idx] = lo
                fallback = max(adjustable, key=lambda i: durations[i])
                durations[fallback] += cycle_length - sum(durations)

    return durations


# ── Traffic-light fitness wrapper (picklable for multiprocessing) ────────────

class TrafficFitnessWrapper:
    """Converts a gene vector into per-TLS durations and calls the SUMO fitness function."""

    def __init__(self, fitness_function, tls_mapping):
        self.fitness_function = fitness_function
        self.tls_mapping = tls_mapping

    def __call__(self, vector: np.ndarray) -> float:
        tls_durations = {}

        for tls in self.tls_mapping:
            raw = list(vector[tls["start_idx"]: tls["end_idx"]])
            tls_durations[tls["tls_id"]] = normalize_to_cycle(
                raw, tls["phase_types"], upper_bounds=tls["upper_bounds"]
            )

        return self.fitness_function(tls_durations)


def build_traffic_fitness_wrapper(baseline_data, fitness_function):
    """Build a picklable fitness wrapper from baseline data. Returns (wrapper, n, lb, ub, labels)."""
    tls_mapping = []
    gene_idx = 0
    x_lower_list, x_upper_list, labels = [], [], []

    for tls_id in sorted(baseline_data["tls_data"].keys()):
        phase_keys = sorted(baseline_data["tls_data"][tls_id].keys())
        phase_types, baseline_durs = [], []

        for pk in phase_keys:
            phase = baseline_data["tls_data"][tls_id][pk]
            ptype = phase_type(phase.get("state", ""))
            phase_types.append(ptype)
            baseline_durs.append(phase.get("duration", PHASE_BOUNDS[ptype][1]))

        # Dynamic per-TLS upper bound from this TLS's (frozen) yellow durations.
        upper_bounds = phase_upper_bounds(phase_types, baseline_durs)

        for pk, ptype, ub in zip(phase_keys, phase_types, upper_bounds):
            x_lower_list.append(float(PHASE_BOUNDS[ptype][0]))
            x_upper_list.append(float(ub))
            labels.append(f"{tls_id}_{pk}")

        tls_mapping.append({
            "tls_id": tls_id, "num_phases": len(phase_keys),
            "phase_types": phase_types,
            "upper_bounds": upper_bounds,
            "start_idx": gene_idx, "end_idx": gene_idx + len(phase_keys),
        })
        gene_idx += len(phase_keys)

    f = TrafficFitnessWrapper(fitness_function, tls_mapping)
    return f, gene_idx, np.array(x_lower_list), np.array(x_upper_list), labels

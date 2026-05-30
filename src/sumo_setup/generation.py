import json
from pathlib import Path

from config import BASELINE_TRAFFIC_DATA, OPTIMIZE_PHASE_COUNTS

from .extraction import extract_traffic_light_data

def generate_data():

    output_data = extract_traffic_light_data(detail=False)

    # Drop TLSs whose phase count is outside OPTIMIZE_PHASE_COUNTS so that
    # every downstream consumer (fitness wrapper, gene map, all optimizer
    # variants) sees the same filtered set. Excluded TLSs keep whatever
    # durations the SUMO network file holds during simulation.
    excluded = [tid for tid, phases in output_data["tls_data"].items()
                if len(phases) not in OPTIMIZE_PHASE_COUNTS]
    for tid in excluded:
        del output_data["tls_data"][tid]
    if excluded:
        print(f"Excluded {len(excluded)} TLS(s) with phase counts outside "
              f"{sorted(OPTIMIZE_PHASE_COUNTS)}: {excluded}")
    print(f"Kept {len(output_data['tls_data'])} TLS(s) in baseline.")

    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / BASELINE_TRAFFIC_DATA

    with open(json_filepath, "w") as f:
        json.dump(output_data, f, indent=4)

    return

if __name__ == "__main__":
    generate_data()
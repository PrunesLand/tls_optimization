"""Count how many vehicles cross each traffic-light intersection, per instance.

For every instance (Jakarta / Kota Kinabalu / Beijing) this reads the SUMO
network (``osm.net.xml.gz``) and the route file the instance actually simulates
(``precalculated_routes.rou.xml``, referenced by ``osm.sumocfg``) and reports,
for each traffic-light junction, how many vehicles drive *through* it.

DEFINITION of "crosses": a vehicle crosses TLS junction J if one of J's
incoming edges appears in the vehicle's route at a position that is not the last
edge — i.e. the vehicle enters J on a controlled approach and keeps going past
the junction. A route that merely ends on an incoming edge stops before the
junction and is not counted. Each vehicle is counted at most once per junction.

This is a purely static, route-based count (no SUMO/libsumo process). It
reflects the *scheduled* crossings from the assigned routes; it does not model
teleports or incomplete trips. The TLS set mirrors the network's
``junction type="traffic_light"`` entries (the same intersections whose
``tlLogic`` programs are optimised).

Usage:
  python -m src.experiments.intersection_crossings
  python -m src.experiments.intersection_crossings --instances beijing
  python -m src.experiments.intersection_crossings --top 15
  python -m src.experiments.intersection_crossings --csv-dir src/outputs/crossings
"""

import argparse
import csv
import gzip
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from config import OPTIMIZE_PHASE_COUNTS  # noqa: E402

# Instance name → setup directory (mirrors instance_info / best_config experiments).
INSTANCES = {
    "jakarta": ROOT / "src" / "sumo_setup",
    "kotakinabalu": ROOT / "src" / "sumo_setup_kotakinabalu",
    "beijing": ROOT / "src" / "sumo_setup_beijing",
}


def edge_of_lane(lane_id: str) -> str:
    """``"659449459#2_0"`` → ``"659449459#2"`` (strip the ``_<laneindex>`` suffix)."""
    return lane_id.rsplit("_", 1)[0]


def optimizable_tls(net_path: Path) -> set:
    """Junction ids of TLS the optimizer actually tunes (phase count ∈ OPTIMIZE_PHASE_COUNTS).

    Mirrors ``instance_info.read_tls_groups`` / ``generation.generate_data``:
    count only phases with duration ≥ 1 s, skip TLS with no green/red logic,
    then keep those whose phase count is in ``config.OPTIMIZE_PHASE_COUNTS``.
    """
    keep = set()
    with gzip.open(net_path, "rt") as f:
        for _, el in ET.iterparse(f):
            # Clear only tlLogic elements: clearing every element would wipe the
            # child <phase> attributes before this parent's end-event fires.
            if el.tag != "tlLogic":
                continue
            phases = [(int(float(p.get("duration", 0))), p.get("state", ""))
                      for p in el.findall("phase")]
            phases = [(d, s) for d, s in phases if d >= 1]
            has_logic = any("g" in s.lower() or "r" in s.lower() for _, s in phases)
            if phases and has_logic and len(phases) in OPTIMIZE_PHASE_COUNTS:
                keep.add(el.get("id"))
            el.clear()
    return keep


def tls_incoming_edges(net_path: Path) -> dict:
    """Map each *optimizable* traffic-light junction id → set of its incoming edge ids.

    Only TLS the optimizer tunes (3- or 4-phase, see :func:`optimizable_tls`) are
    included, so the crossing stats line up with what optimisation can affect.
    Internal junctions (negative/internal lanes) are ignored; only the named
    approach edges listed in ``incLanes`` are kept.
    """
    keep = optimizable_tls(net_path)
    incoming = {}
    with gzip.open(net_path, "rt") as f:
        for _, el in ET.iterparse(f):
            if (el.tag == "junction" and el.get("type") == "traffic_light"
                    and el.get("id") in keep):
                lanes = (el.get("incLanes") or "").split()
                edges = {edge_of_lane(l) for l in lanes if l and not l.startswith(":")}
                incoming[el.get("id")] = edges
            el.clear()
    return incoming


def vehicle_routes(rou_path: Path):
    """Yield ``(vehicle_id, [edge, ...])`` for each vehicle in the route file."""
    for _, el in ET.iterparse(rou_path):
        if el.tag == "vehicle":
            route = el.find("route")
            edges = (route.get("edges") if route is not None else "") or ""
            yield el.get("id"), edges.split()
            el.clear()


def count_crossings(net_path: Path, rou_path: Path):
    """Return ``({tls_id: vehicle_count}, n_vehicles)`` for one instance.

    A vehicle is counted for a TLS if one of that TLS's incoming edges occurs in
    its route before the final edge (so the vehicle actually drives through the
    junction rather than terminating on the approach).
    """
    incoming = tls_incoming_edges(net_path)
    # Reverse index: incoming edge → set of TLS ids it feeds, so each route is a
    # single linear scan instead of (#TLS × route length).
    edge_to_tls = defaultdict(set)
    for tls_id, edges in incoming.items():
        for e in edges:
            edge_to_tls[e].add(tls_id)

    crossings = {tls_id: 0 for tls_id in incoming}
    n_vehicles = 0
    for _vid, edges in vehicle_routes(rou_path):
        n_vehicles += 1
        crossed = set()
        # Exclude the last edge: the vehicle stops there and never traverses
        # the downstream junction.
        for e in edges[:-1]:
            crossed.update(edge_to_tls.get(e, ()))
        for tls_id in crossed:
            crossings[tls_id] += 1
    return crossings, n_vehicles


def print_instance(name: str, crossings: dict, n_vehicles: int, top: int):
    total_tls = len(crossings)
    crossed_tls = sum(1 for c in crossings.values() if c > 0)
    counts = sorted(crossings.values(), reverse=True)
    total_crossings = sum(counts)

    never = sum(1 for c in counts if c == 0)
    once = sum(1 for c in counts if c == 1)
    multi = sum(1 for c in counts if c > 1)
    mean_all = total_crossings / total_tls if total_tls else 0
    mean_crossed = total_crossings / crossed_tls if crossed_tls else 0

    print(f"\n{'#' * 70}\n# Instance: {name}\n{'#' * 70}")
    print(f"Vehicles: {n_vehicles}   |   TLS junctions: {total_tls}   "
          f"|   TLS with ≥1 crossing: {crossed_tls}")
    print(f"Total vehicle-crossings: {total_crossings}   "
          f"|   busiest: {counts[0] if counts else 0}")
    print("  TLS breakdown:")
    print(f"    never crossed (=0)    : {never:>4}  ({100 * never / total_tls:4.1f}%)")
    print(f"    crossed exactly once  : {once:>4}  ({100 * once / total_tls:4.1f}%)")
    print(f"    crossed >1 time       : {multi:>4}  ({100 * multi / total_tls:4.1f}%)")
    print(f"  Mean crossings per TLS  : {mean_all:.2f}  (over all {total_tls} TLS)")
    print(f"  Mean over crossed TLS   : {mean_crossed:.2f}  (over the {crossed_tls} with traffic)")

    ranked = sorted(crossings.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ranked if top <= 0 else ranked[:top]
    label = "all" if top <= 0 else f"top {len(shown)}"
    print(f"\n  Crossings per intersection ({label} of {total_tls}):")
    print(f"    {'TLS id':<18} {'vehicles':>9}  {'% of fleet':>10}")
    for tls_id, c in shown:
        print(f"    {tls_id:<18} {c:>9}  {100 * c / n_vehicles:>9.1f}%")


def write_csv(csv_dir: Path, name: str, crossings: dict, n_vehicles: int):
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"crossings_{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tls_id", "vehicles_crossing", "pct_of_fleet"])
        for tls_id, c in sorted(crossings.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([tls_id, c, round(100 * c / n_vehicles, 2)])
    print(f"  -> wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="*", default=list(INSTANCES.keys()),
                    help=f"Subset of instances (default: {list(INSTANCES.keys())}).")
    ap.add_argument("--top", type=int, default=20,
                    help="Show only the N busiest intersections (0 = all). Default 20.")
    ap.add_argument("--csv-dir", type=Path, default=None,
                    help="If set, also write per-instance crossings_<name>.csv here.")
    args = ap.parse_args()

    for name in args.instances:
        if name not in INSTANCES:
            print(f"WARNING: unknown instance '{name}' ignored "
                  f"(known: {list(INSTANCES.keys())})")
            continue
        setup = INSTANCES[name]
        net_path = setup / "osm.net.xml.gz"
        rou_path = setup / "precalculated_routes.rou.xml"
        if not net_path.exists() or not rou_path.exists():
            print(f"\n### {name}: missing net/route file in {setup} — skipped")
            continue

        crossings, n_vehicles = count_crossings(net_path, rou_path)
        print_instance(name, crossings, n_vehicles, args.top)
        if args.csv_dir is not None:
            write_csv(args.csv_dir, name, crossings, n_vehicles)


if __name__ == "__main__":
    main()

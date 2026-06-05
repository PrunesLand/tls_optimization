"""Print traffic-light phase statistics for each instance (Jakarta/Beijing/Kota Kinabalu).

For every instance, TLSs are grouped by their phase count (3-phase, 4-phase, …)
and for each group this prints:

  * number of TLS in the group (and whether the count is optimised, i.e. in
    ``config.OPTIMIZE_PHASE_COUNTS``);
  * per phase-type (green / yellow / red) count + min / mean / max duration,
    plus how many greens are 3 s or below the green floor;
  * a histogram of the green durations in that group.

CLASSIFICATION (matches the project convention in ``statistics.py`` /
``extraction.py``): a phase is GREEN if its state string contains any ``g``/``G``;
else YELLOW if it contains ``y``/``u``; else RED. Only phases with
duration ≥ 1 s are counted, and a TLS with no green/red logic is skipped — same
filtering ``extraction.extract_traffic_light_data`` applies.

Data is read straight from each instance's ``osm.net.xml.gz`` (the ``tlLogic``
static program), so no SUMO/libsumo process is started.

Usage:
  python -m src.experiments.instance_info
  python -m src.experiments.instance_info --instances beijing kotakinabalu
  python -m src.experiments.instance_info --bin-width 5
"""

import argparse
import gzip
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from config import GREEN_FLOOR, CYCLE_LENGTH, OPTIMIZE_PHASE_COUNTS  # noqa: E402

# Instance name → SUMO network file (mirrors best_config_instance_experiments).
INSTANCES = {
    "jakarta": ROOT / "src" / "sumo_setup" / "osm.net.xml.gz",
    "kotakinabalu": ROOT / "src" / "sumo_setup_kotakinabalu" / "osm.net.xml.gz",
    "beijing": ROOT / "src" / "sumo_setup_beijing" / "osm.net.xml.gz",
}


def classify(state: str) -> str:
    """Phase type from a SUMO state string — project 'any-g' convention."""
    s = state.lower()
    if "g" in s:
        return "green"
    if "y" in s or "u" in s:
        return "yellow"
    return "red"


def read_tls_groups(net_path):
    """Group a network's TLSs by phase count.

    Returns ``{phase_count: [tls, …]}`` where each ``tls`` is a list of
    ``(duration:int, ptype:str)`` for its phases (duration ≥ 1 s only). Mirrors
    extraction.py: drop sub-second phases, skip TLSs with no green/red logic.
    """
    groups = defaultdict(list)
    with gzip.open(net_path, "rt") as f:
        for _, el in ET.iterparse(f):
            if el.tag != "tlLogic":
                continue
            phases = [(int(float(p.get("duration", 0))), p.get("state", ""))
                      for p in el.findall("phase")]
            phases = [(d, s) for d, s in phases if d >= 1]
            has_logic = any("g" in s.lower() or "r" in s.lower() for _, s in phases)
            if phases and has_logic:
                groups[len(phases)].append([(d, classify(s)) for d, s in phases])
            el.clear()
    return groups


def green_histogram(greens, width):
    """Render a green-duration histogram with ``width``-second bins."""
    if not greens:
        print("      (no green phases)")
        return
    n = len(greens)
    n_bins = max(greens) // width + 1
    counts = [0] * n_bins
    for d in greens:
        counts[d // width] += 1
    for i, cnt in enumerate(counts):
        lo, hi = i * width, i * width + width - 1
        bar = "#" * round(40 * cnt / n)
        print(f"      {lo:>3}-{hi:<3}s: {cnt:>5} ({100 * cnt / n:5.1f}%) {bar}")


def print_instance(name, net_path, bin_width):
    if not net_path.exists():
        print(f"\n### {name}: network not found at {net_path} — skipped")
        return

    groups = read_tls_groups(net_path)
    total_tls = sum(len(v) for v in groups.values())
    dist = {pc: len(v) for pc, v in sorted(groups.items())}

    print(f"\n{'#' * 70}\n# Instance: {name}   ({net_path.relative_to(ROOT)})\n{'#' * 70}")
    print(f"Total TLS with logic: {total_tls}   |   cycle length: {CYCLE_LENGTH}s")
    print(f"Phase-count distribution: {dist}")

    for pc in sorted(groups):
        tls_list = groups[pc]
        tag = "optimised" if pc in OPTIMIZE_PHASE_COUNTS else "not optimised"
        print(f"\n----- {pc}-phase TLS: {len(tls_list)} TLS  [{tag}] -----")

        by_type = {"green": [], "yellow": [], "red": []}
        for tls in tls_list:
            for d, pt in tls:
                by_type[pt].append(d)

        print("  Phase-type durations (green = any state with 'g'):")
        for pt in ("green", "yellow", "red"):
            a = by_type[pt]
            if a:
                print(f"    {pt.upper():<6}: n={len(a):<5} "
                      f"min={min(a):>3}s mean={sum(a) / len(a):>5.1f}s max={max(a):>3}s")
            else:
                print(f"    {pt.upper():<6}: none")

        greens = by_type["green"]
        if greens:
            n3 = sum(1 for d in greens if d == 3)
            nsub = sum(1 for d in greens if d < GREEN_FLOOR)
            ng = len(greens)
            print(f"    short greens: =3s -> {n3} ({100 * n3 / ng:.1f}%)   "
                  f"<{int(GREEN_FLOOR)}s (floor) -> {nsub} ({100 * nsub / ng:.1f}%)")
            top = Counter(greens).most_common(6)
            print("    most common green durations: "
                  + ", ".join(f"{d}s×{c}" for d, c in sorted(top, key=lambda x: -x[1])))
            print(f"  Green-duration histogram (bin = {bin_width}s):")
            green_histogram(greens, bin_width)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="*", default=list(INSTANCES.keys()),
                    help=f"Subset of instances (default: {list(INSTANCES.keys())}).")
    ap.add_argument("--bin-width", type=int, default=10,
                    help="Green-histogram bin width in seconds (default: 10).")
    args = ap.parse_args()

    for name in args.instances:
        if name not in INSTANCES:
            print(f"WARNING: unknown instance '{name}' ignored "
                  f"(known: {list(INSTANCES.keys())})")
            continue
        print_instance(name, INSTANCES[name], args.bin_width)


if __name__ == "__main__":
    main()

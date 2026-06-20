"""Map the 2-TLS Ward "pairs" each distance strategy produces, over the routes.

Companion to ``src.plot.visualize_uncrossed_tls``.  Where that map highlights
the traffic lights no vehicle crosses, this one highlights the **pair clusters**
that the pair-cluster mutation operates on: the size-2 nodes of the Ward linkage
tree built from each distance strategy (euclidian / shortest / fastest / random,
see ``config.TREE_STRATEGIES``).

For one instance (Malaysia / Kota Kinabalu by default) and each strategy it:

  * reads the strategy's ``tls_distances_<name>.json`` (TLS coords live there),
  * rebuilds the Ward tree via ``build_all_tree_masks`` to get its 2-TLS pairs,
  * renders an interactive Folium map where every pair gets its own colour —
    both paired lights AND the line joining them share that colour, so a pair
    is visually obvious;
  * draws the vehicle routes underneath as a faded grey highlight for context;
  * shows TLS that ended up in no pair as small grey dots.

One HTML map is written per strategy to ``src/outputs/tls_pairs/<instance>/``.

Usage:
  python -m src.plot.visualize_tls_pairs
  python -m src.plot.visualize_tls_pairs --strategies euclidian fastest
  python -m src.plot.visualize_tls_pairs --instance kotakinabalu
"""

import argparse
import colorsys
import json
import sys
from pathlib import Path

import sumolib

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from config import INSTANCES, TREE_STRATEGIES  # noqa: E402
from src.experiments.intersection_crossings import INSTANCES as SETUP_DIRS  # noqa: E402
from src.novel.distance_trees import distance_tree_paths  # noqa: E402
from src.novel.linkage_tree import build_all_tree_masks  # noqa: E402

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

OUT_DIR = ROOT / "src" / "outputs" / "tls_pairs"


def edge_shape_lonlat(net, edge_id):
    """Return [(lat, lon), ...] for an edge's geometry, or [] if unknown."""
    try:
        edge = net.getEdge(edge_id)
    except KeyError:
        return []
    return [net.convertXY2LonLat(x, y)[::-1] for x, y in edge.getShape()]


def vehicle_routes(rou_path):
    """Yield ``[edge, ...]`` for each vehicle in the route file."""
    import xml.etree.ElementTree as ET
    for _, el in ET.iterparse(rou_path):
        if el.tag == "vehicle":
            route = el.find("route")
            edges = (route.get("edges") if route is not None else "") or ""
            yield edges.split()
            el.clear()


def route_polylines(net, rou_path):
    """Yield one [(lat, lon), ...] polyline per vehicle, following its full route."""
    for edges in vehicle_routes(rou_path):
        line = []
        for e in edges:
            line.extend(edge_shape_lonlat(net, e))
        if line:
            yield line


def distinct_colours(n):
    """Return ``n`` visually distinct hex colours, evenly spaced around the wheel."""
    cols = []
    for i in range(n):
        h = i / max(n, 1)
        # Alternate saturation/value a little so neighbours stay separable.
        s = 0.65 + 0.25 * (i % 2)
        v = 0.95 - 0.20 * (i % 3) / 2
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return cols


def tls_coords_from_json(dist_json):
    """Return ``{tls_id: (lat, lon)}`` from a tls_distances JSON's traffic_lights."""
    with open(dist_json) as f:
        data = json.load(f)
    return {t["id"]: (t["lat"], t["lon"]) for t in data["traffic_lights"]}


def build_map(instance, strategy, dist_json, net, rou_path, route_lines, out_dir):
    if not dist_json.exists():
        print(f"### {instance}/{strategy}: missing {dist_json} — skipped")
        return

    coords = tls_coords_from_json(dist_json)
    _, pairs, _ = build_all_tree_masks(str(dist_json))
    # Keep only pairs whose endpoints both have coordinates.
    pairs = [(a, b) for a, b in pairs if a in coords and b in coords]
    paired_ids = {t for ab in pairs for t in ab}
    unpaired = {t: c for t, c in coords.items() if t not in paired_ids}

    print(f"\n{instance} / {strategy}: {len(pairs)} pairs "
          f"({len(paired_ids)} paired, {len(unpaired)} unpaired TLS)")
    for (a, b), col in zip(pairs, distinct_colours(len(pairs))):
        print(f"    {col}  {a:<14} <-> {b}")

    if not HAS_FOLIUM:
        print("  (install folium to render the HTML map: pip install folium)")
        return
    if not coords:
        print("  no TLS coordinates resolved — map skipped")
        return

    center = [sum(c[0] for c in coords.values()) / len(coords),
              sum(c[1] for c in coords.values()) / len(coords)]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # Vehicle paths underneath everything, no start/finish markers.
    for line in route_lines:
        folium.PolyLine(line, color="#999999", weight=1.2, opacity=0.30).add_to(m)

    # Unpaired lights next (grey) for context.
    for t, (lat, lon) in unpaired.items():
        folium.CircleMarker(
            location=[lat, lon], radius=3, color="gray", fill=True,
            fill_opacity=0.5, opacity=0.5, popup=f"TLS {t} — unpaired",
        ).add_to(m)

    # Pairs on top: both endpoints + the connecting line share one colour.
    for (a, b), col in zip(pairs, distinct_colours(len(pairs))):
        (lat_a, lon_a), (lat_b, lon_b) = coords[a], coords[b]
        folium.PolyLine(
            [[lat_a, lon_a], [lat_b, lon_b]], color=col, weight=3, opacity=0.9,
            popup=f"pair: {a} <-> {b}",
        ).add_to(m)
        for t, (lat, lon) in ((a, (lat_a, lon_a)), (b, (lat_b, lon_b))):
            folium.CircleMarker(
                location=[lat, lon], radius=7, color=col, weight=2,
                fill=True, fill_color=col, fill_opacity=0.9,
                popup=f"TLS {t} — pair with {b if t == a else a}",
            ).add_to(m)

    out_dir = out_dir / instance
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tls_pairs_{instance}_{strategy}.html"
    m.save(str(out_path))
    print(f"  -> {out_path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="kotakinabalu",
                    help="Instance to map (default: kotakinabalu / Malaysia).")
    ap.add_argument("--strategies", nargs="*", default=list(TREE_STRATEGIES),
                    help=f"Distance strategies (default: {list(TREE_STRATEGIES)}).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help=f"Output directory for HTML maps (default: {OUT_DIR}).")
    args = ap.parse_args()

    if args.instance not in INSTANCES:
        print(f"unknown instance '{args.instance}' (known: {list(INSTANCES)})")
        return

    setup = SETUP_DIRS[args.instance]
    net_path = setup / "osm.net.xml.gz"
    rou_path = setup / "precalculated_routes.rou.xml"
    if not net_path.exists() or not rou_path.exists():
        print(f"### {args.instance}: missing net/route file in {setup} — aborting")
        return

    out_base = INSTANCES[args.instance]["out_dir"]
    paths = distance_tree_paths(out_base)

    print(f"Loading SUMO network for {args.instance} ...")
    net = sumolib.net.readNet(str(net_path))
    # Build the route polylines once and reuse them across every strategy map.
    print("Tracing vehicle routes ...")
    route_lines = list(route_polylines(net, rou_path))
    print(f"  {len(route_lines)} vehicle routes traced.")

    for strategy in args.strategies:
        if strategy not in paths:
            print(f"WARNING: unknown strategy '{strategy}' ignored "
                  f"(known: {list(paths)})")
            continue
        build_map(args.instance, strategy, paths[strategy], net, rou_path,
                  route_lines, args.out_dir)


if __name__ == "__main__":
    main()

"""Map the traffic lights that no vehicle ever crosses, per instance.

For each instance (Jakarta / Kota Kinabalu / Beijing) this reuses the route-based
crossing count from ``src.experiments.intersection_crossings`` to find the TLS
junctions with zero crossings, looks up their geographic coordinates from the
SUMO network (via sumolib's XY→lon/lat conversion, same as ``plot_tls_map.py``),
and renders an interactive Folium map:

  * RED markers   = traffic lights never crossed by any vehicle;
  * grey dots     = traffic lights that are crossed (shown for context, with
                    their crossing count in the popup).

One HTML map is written per instance to ``src/outputs/uncrossed_tls/``.

Usage:
  python -m src.plot.visualize_uncrossed_tls
  python -m src.plot.visualize_uncrossed_tls --instances beijing
"""

import argparse
import sys
from pathlib import Path

import sumolib

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from src.experiments.intersection_crossings import (  # noqa: E402
    INSTANCES, count_crossings, vehicle_routes,
)

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

OUT_DIR = ROOT / "src" / "outputs" / "uncrossed_tls"


def edge_shape_lonlat(net, edge_id):
    """Return [(lat, lon), ...] for an edge's geometry, or [] if unknown."""
    try:
        edge = net.getEdge(edge_id)
    except KeyError:
        return []
    return [net.convertXY2LonLat(x, y)[::-1] for x, y in edge.getShape()]


def route_polylines(net, rou_path):
    """Yield one [(lat, lon), ...] polyline per vehicle, following its full route."""
    for _vid, edges in vehicle_routes(rou_path):
        line = []
        for e in edges:
            line.extend(edge_shape_lonlat(net, e))
        if line:
            yield line


def tls_coords(net, tls_ids):
    """Return ``{tls_id: (lat, lon)}`` for the given junction ids using sumolib."""
    coords = {}
    for tls_id in tls_ids:
        try:
            x, y = net.getNode(tls_id).getCoord()
        except KeyError:
            continue
        lon, lat = net.convertXY2LonLat(x, y)
        coords[tls_id] = (lat, lon)
    return coords


def build_map(name, setup, out_dir):
    net_path = setup / "osm.net.xml.gz"
    rou_path = setup / "precalculated_routes.rou.xml"
    if not net_path.exists() or not rou_path.exists():
        print(f"### {name}: missing net/route file in {setup} — skipped")
        return

    crossings, n_vehicles = count_crossings(net_path, rou_path)
    net = sumolib.net.readNet(str(net_path))
    coords = tls_coords(net, crossings.keys())

    never = {t: coords[t] for t, c in crossings.items() if c == 0 and t in coords}
    crossed = {t: coords[t] for t, c in crossings.items() if c > 0 and t in coords}

    print(f"\n{name}: {len(never)} never-crossed TLS "
          f"(of {len(crossings)} total) | vehicles: {n_vehicles}")
    for t in sorted(never):
        lat, lon = never[t]
        print(f"    {t:<18} lat={lat:10.6f} lon={lon:10.6f}")

    if not HAS_FOLIUM:
        print("  (install folium to render the HTML map: pip install folium)")
        return

    all_pts = list(never.values()) + list(crossed.values())
    if not all_pts:
        print("  no TLS coordinates resolved — map skipped")
        return
    center = [sum(p[0] for p in all_pts) / len(all_pts),
              sum(p[1] for p in all_pts) / len(all_pts)]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # Vehicle paths underneath everything, no start/finish markers.
    for line in route_polylines(net, rou_path):
        folium.PolyLine(line, color="#3388ff", weight=1.5, opacity=0.35).add_to(m)

    # Crossed lights next (grey) for context.
    for t, (lat, lon) in crossed.items():
        folium.CircleMarker(
            location=[lat, lon], radius=3, color="gray", fill=True,
            fill_opacity=0.5, opacity=0.5,
            popup=f"TLS {t} — crossed {crossings[t]}×",
        ).add_to(m)

    # Never-crossed lights on top (red markers).
    for t, (lat, lon) in never.items():
        folium.Marker(
            location=[lat, lon],
            popup=f"NEVER CROSSED — TLS {t}",
            icon=folium.Icon(color="red", icon="remove-sign"),
        ).add_to(m)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"uncrossed_tls_{name}.html"
    m.save(str(out_path))
    print(f"  -> {out_path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="*", default=list(INSTANCES.keys()),
                    help=f"Subset of instances (default: {list(INSTANCES.keys())}).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help=f"Output directory for HTML maps (default: {OUT_DIR}).")
    args = ap.parse_args()

    for name in args.instances:
        if name not in INSTANCES:
            print(f"WARNING: unknown instance '{name}' ignored "
                  f"(known: {list(INSTANCES.keys())})")
            continue
        build_map(name, INSTANCES[name], args.out_dir)


if __name__ == "__main__":
    main()

"""Visualize precalculated vehicle routes (start/end points + full paths) per instance.

For each SUMO instance we parse the precalculated route file, take the first edge of
each vehicle's route as the start point and the last edge as the end point, convert the
geometry to lon/lat using the network's geo-projection, and render an interactive
folium map (one HTML per instance) plus a CSV of origin/destination coordinates.
"""

import csv
import gzip
import os
import xml.etree.ElementTree as ET

import folium
import sumolib

INSTANCES = {
    "jakarta": "sumo_setup",
    "beijing": "sumo_setup_beijing",
    "kotakinabalu": "sumo_setup_kotakinabalu",
}

# This file lives in src/plot/; instance setups and outputs live under src/.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "outputs", "vehicle_maps")


def load_net(setup_dir):
    """sumolib can't read .gz directly in all versions; decompress to a temp .net.xml."""
    gz_path = os.path.join(setup_dir, "osm.net.xml.gz")
    plain_path = os.path.join(setup_dir, "osm.net.xml")
    if not os.path.exists(plain_path):
        with gzip.open(gz_path, "rb") as fin, open(plain_path, "wb") as fout:
            fout.write(fin.read())
    return sumolib.net.readNet(plain_path)


def edge_endpoint(net, edge_id, which):
    """Return (lon, lat) of the start ('first') or end ('last') of an edge's shape."""
    try:
        edge = net.getEdge(edge_id)
    except KeyError:
        return None
    shape = edge.getShape()
    x, y = shape[0] if which == "first" else shape[-1]
    lon, lat = net.convertXY2LonLat(x, y)
    return lon, lat


def edge_shape_lonlat(net, edge_id):
    try:
        edge = net.getEdge(edge_id)
    except KeyError:
        return []
    return [net.convertXY2LonLat(x, y) for x, y in edge.getShape()]


def parse_routes(rou_path):
    """Yield (veh_id, [edge_ids]) for every vehicle in the route file."""
    for _, elem in ET.iterparse(rou_path):
        if elem.tag == "vehicle":
            route = elem.find("route")
            if route is not None and route.get("edges"):
                yield elem.get("id"), route.get("edges").split()
            elem.clear()


def build_map(name, setup_dir, draw_full_paths=True):
    net = load_net(setup_dir)
    rou_path = os.path.join(setup_dir, "precalculated_routes.rou.xml")

    rows = []
    starts, ends = [], []
    paths = []
    for veh_id, edges in parse_routes(rou_path):
        start = edge_endpoint(net, edges[0], "first")
        end = edge_endpoint(net, edges[-1], "last")
        if start is None or end is None:
            continue
        rows.append({
            "vehicle_id": veh_id,
            "start_lon": start[0], "start_lat": start[1],
            "end_lon": end[0], "end_lat": end[1],
            "n_edges": len(edges),
        })
        starts.append((start[1], start[0]))  # folium wants (lat, lon)
        ends.append((end[1], end[0]))
        if draw_full_paths:
            line = []
            for e in edges:
                line.extend((lat, lon) for lon, lat in edge_shape_lonlat(net, e))
            paths.append(line)

    # Center on mean of all start points.
    clat = sum(p[0] for p in starts) / len(starts)
    clon = sum(p[1] for p in starts) / len(starts)
    fmap = folium.Map(location=[clat, clon], zoom_start=13, tiles="cartodbpositron")

    if draw_full_paths:
        path_layer = folium.FeatureGroup(name="Routes", show=True)
        for line in paths:
            folium.PolyLine(line, color="#3388ff", weight=1.5, opacity=0.35).add_to(path_layer)
        path_layer.add_to(fmap)

    # Traffic lights: junction nodes typed as 'traffic_light'.
    tl_nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    tl_layer = folium.FeatureGroup(name=f"Traffic lights ({len(tl_nodes)})", show=True)
    for node in tl_nodes:
        lon, lat = net.convertXY2LonLat(*node.getCoord())
        folium.CircleMarker(
            (lat, lon), radius=4, color="#d62728", fill=True, fill_color="#d62728",
            fill_opacity=0.9, weight=1, popup=f"TLS {node.getID()}",
        ).add_to(tl_layer)
    tl_layer.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    os.makedirs(OUT_DIR, exist_ok=True)
    html_path = os.path.join(OUT_DIR, f"{name}_routes.html")
    csv_path = os.path.join(OUT_DIR, f"{name}_od.csv")
    tls_csv_path = os.path.join(OUT_DIR, f"{name}_traffic_lights.csv")
    fmap.save(html_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(tls_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tls_id", "lon", "lat"])
        for node in tl_nodes:
            lon, lat = net.convertXY2LonLat(*node.getCoord())
            w.writerow([node.getID(), lon, lat])
    return len(rows), len(tl_nodes), html_path


if __name__ == "__main__":
    for name, sub in INSTANCES.items():
        setup_dir = os.path.join(SRC_DIR, sub)
        n, n_tls, html_path = build_map(name, setup_dir)
        print(f"{name:14s} {n:3d} routes, {n_tls:3d} traffic lights -> {os.path.relpath(html_path, SRC_DIR)}")

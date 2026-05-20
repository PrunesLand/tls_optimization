import os
import sys
from pathlib import Path
import sumolib

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    net_file = project_root / "src" / "sumo_setup" / "osm.net.xml.gz"
    out_map = project_root / "src" / "outputs" / "tls_map.html"

    print(f"Loading SUMO network from: {net_file} ...")
    net = sumolib.net.readNet(str(net_file))

    tls_coordinates = []

    for node in net.getNodes():
        if node.getType() == "traffic_light":
            x, y = node.getCoord()
            lon, lat = net.convertXY2LonLat(x, y)
            tls_id = node.getID()
            
            tls_coordinates.append({
                "id": tls_id,
                "lat": lat,
                "lon": lon
            })
            print(f"TLS Node: {tls_id:15} | Lat: {lat:10.6f} | Lon: {lon:10.6f}")

    print(f"\nFound {len(tls_coordinates)} Traffic Light Nodes.\n")

    if HAS_FOLIUM and tls_coordinates:
        print("Generating interactive map with Folium...")
        
        center_lat = sum(t["lat"] for t in tls_coordinates) / len(tls_coordinates)
        center_lon = sum(t["lon"] for t in tls_coordinates) / len(tls_coordinates)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        for tls in tls_coordinates:
            folium.Marker(
                location=[tls["lat"], tls["lon"]],
                popup=f"Traffic Light: {tls['id']}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

        m.save(str(out_map))
        print(f"Map successfully generated! Open this file in your browser:")
        print(f"-> {out_map}")
    else:
        print("\nIf you want to generate an interactive HTML map, please install folium:")
        print("    pip install folium")

if __name__ == "__main__":
    main()

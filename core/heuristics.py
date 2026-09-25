"""
Módulo de cálculo de heurísticas de alto rendimiento para búsqueda informada (Greedy y A*).
Utiliza coordenadas cacheadas en memoria para garantizar tiempos de ejecución en microsegundos.
"""
import math
from typing import Any, Dict, Tuple


def fast_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula la distancia ortodrómica (Haversine) en metros entre dos pares de coordenadas (lat, lon).
    Optimizado a bajo nivel con funciones matemáticas directas.
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

    return 6371008.8 * c


def haversine_distance(node_a: Any, node_b: Any, G_or_coords: Any) -> float:
    """
    Calcula la distancia Haversine en metros entre dos nodos.
    Soporta tanto diccionarios de coordenadas ultrarrápidos como grafos NetworkX.
    """
    if node_a == node_b:
        return 0.0

    if isinstance(G_or_coords, dict):
        coords_a = G_or_coords[node_a]
        coords_b = G_or_coords[node_b]
        return fast_haversine(coords_a[0], coords_a[1], coords_b[0], coords_b[1])

    # Fallback si se pasa un grafo NetworkX
    data_a = G_or_coords.nodes[node_a]
    data_b = G_or_coords.nodes[node_b]
    return fast_haversine(float(data_a["y"]), float(data_a["x"]), float(data_b["y"]), float(data_b["x"]))

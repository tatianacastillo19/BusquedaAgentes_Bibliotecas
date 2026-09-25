"""
Agente de Búsqueda Voraz (Greedy Best-First Search).
Optimizado con heurística Haversine directa y listas de adyacencia de alta velocidad.
"""
import heapq
import itertools
import time
from typing import Any, Dict, List, Set, Tuple
from core.heuristics import haversine_distance, fast_haversine
from core.map_loader import BogotaRoadMap, get_edge_cost


def greedy_search(
    road_map: Any,
    origen_node: Any,
    destino_node: Any
) -> Tuple[List[Any], float, int, float]:
    """
    Ejecuta el algoritmo de Búsqueda Voraz (Greedy Best-First).
    """
    start_time = time.perf_counter()

    if origen_node == destino_node:
        elapsed_time = time.perf_counter() - start_time
        return [origen_node], 0.0, 1, elapsed_time

    is_custom = isinstance(road_map, BogotaRoadMap)
    coords = road_map.coords if is_custom else None
    adj = road_map.adj if is_custom else None

    dest_lat, dest_lon = coords[destino_node] if is_custom else (0.0, 0.0)

    counter = itertools.count()
    if is_custom:
        o_lat, o_lon = coords[origen_node]
        initial_h = fast_haversine(o_lat, o_lon, dest_lat, dest_lon)
    else:
        initial_h = haversine_distance(origen_node, destino_node, road_map)

    frontier = []
    heapq.heappush(frontier, (initial_h, next(counter), origen_node))

    visited: Set[Any] = set()
    parent_map: Dict[Any, Any] = {}
    visited_nodes_count = 0
    found = False

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)
        visited_nodes_count += 1

        if current == destino_node:
            found = True
            break

        if is_custom:
            for neighbor, _ in adj.get(current, []):
                if neighbor not in visited:
                    if neighbor not in parent_map:
                        parent_map[neighbor] = current
                    n_lat, n_lon = coords[neighbor]
                    h_val = fast_haversine(n_lat, n_lon, dest_lat, dest_lon)
                    heapq.heappush(frontier, (h_val, next(counter), neighbor))
        else:
            for neighbor in road_map.successors(current):
                if neighbor not in visited:
                    if neighbor not in parent_map:
                        parent_map[neighbor] = current
                    h_val = haversine_distance(neighbor, destino_node, road_map)
                    heapq.heappush(frontier, (h_val, next(counter), neighbor))

    elapsed_time = time.perf_counter() - start_time

    if not found:
        return [], float("inf"), visited_nodes_count, elapsed_time

    # Reconstruir camino
    path = []
    curr = destino_node
    while curr is not None:
        path.append(curr)
        curr = parent_map.get(curr)
    path.reverse()

    # Calcular costo métrico
    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += get_edge_cost(road_map, path[i], path[i + 1])

    return path, total_cost, visited_nodes_count, elapsed_time

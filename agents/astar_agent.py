"""
Agente de Búsqueda A* (A-Star Search).
Optimizado con función f(n) = g(n) + h(n), coordenadas cacheadas y listas de adyacencia directa.
"""
import heapq
import itertools
import time
from typing import Any, Dict, List, Tuple
from core.heuristics import haversine_distance, fast_haversine
from core.map_loader import BogotaRoadMap, get_edge_cost


def astar_search(
    road_map: Any,
    origen_node: Any,
    destino_node: Any
) -> Tuple[List[Any], float, int, float]:
    """
    Ejecuta el algoritmo de Búsqueda A* en tiempo ultrarrápido.
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

    # Frontier: (f_score, tie_breaker, current_node, g_score)
    frontier = []
    heapq.heappush(frontier, (initial_h, next(counter), origen_node, 0.0))

    g_score: Dict[Any, float] = {origen_node: 0.0}
    parent_map: Dict[Any, Any] = {}
    visited_nodes_count = 0
    found = False

    while frontier:
        f_val, _, current, current_g = heapq.heappop(frontier)
        visited_nodes_count += 1

        if current == destino_node:
            found = True
            break

        if current_g > g_score.get(current, float("inf")):
            continue

        if is_custom:
            for neighbor, edge_weight in adj.get(current, []):
                tentative_g = current_g + edge_weight
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent_map[neighbor] = current
                    n_lat, n_lon = coords[neighbor]
                    h_val = fast_haversine(n_lat, n_lon, dest_lat, dest_lon)
                    f_score = tentative_g + h_val
                    heapq.heappush(frontier, (f_score, next(counter), neighbor, tentative_g))
        else:
            for neighbor in road_map.successors(current):
                edge_weight = get_edge_cost(road_map, current, neighbor)
                tentative_g = current_g + edge_weight
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent_map[neighbor] = current
                    h_val = haversine_distance(neighbor, destino_node, road_map)
                    f_score = tentative_g + h_val
                    heapq.heappush(frontier, (f_score, next(counter), neighbor, tentative_g))

    elapsed_time = time.perf_counter() - start_time

    if not found:
        return [], float("inf"), visited_nodes_count, elapsed_time

    # Reconstrucción del camino óptimo
    path = []
    curr = destino_node
    while curr is not None:
        path.append(curr)
        curr = parent_map.get(curr)
    path.reverse()

    return path, g_score[destino_node], visited_nodes_count, elapsed_time

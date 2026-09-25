"""
Agente de Búsqueda de Costo Uniforme (UCS).
Optimizado con heapq y listas de adyacencia de alta velocidad para grafos viales grandes.
"""
import heapq
import itertools
import time
from typing import Any, Dict, List, Tuple
from core.map_loader import BogotaRoadMap, get_edge_cost


def ucs_search(
    road_map: Any,
    origen_node: Any,
    destino_node: Any
) -> Tuple[List[Any], float, int, float]:
    """
    Ejecuta el algoritmo de Búsqueda de Costo Uniforme (UCS).
    """
    start_time = time.perf_counter()

    if origen_node == destino_node:
        elapsed_time = time.perf_counter() - start_time
        return [origen_node], 0.0, 1, elapsed_time

    is_custom = isinstance(road_map, BogotaRoadMap)
    adj = road_map.adj if is_custom else None

    counter = itertools.count()
    frontier = []
    heapq.heappush(frontier, (0.0, next(counter), origen_node))

    cost_so_far: Dict[Any, float] = {origen_node: 0.0}
    parent_map: Dict[Any, Any] = {}
    visited_nodes_count = 0
    found = False

    while frontier:
        current_cost, _, current = heapq.heappop(frontier)
        visited_nodes_count += 1

        if current == destino_node:
            found = True
            break

        if current_cost > cost_so_far.get(current, float("inf")):
            continue

        if is_custom:
            for neighbor, edge_weight in adj.get(current, []):
                new_cost = current_cost + edge_weight
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    parent_map[neighbor] = current
                    heapq.heappush(frontier, (new_cost, next(counter), neighbor))
        else:
            for neighbor in road_map.successors(current):
                edge_weight = get_edge_cost(road_map, current, neighbor)
                new_cost = current_cost + edge_weight
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    parent_map[neighbor] = current
                    heapq.heappush(frontier, (new_cost, next(counter), neighbor))

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

    return path, cost_so_far[destino_node], visited_nodes_count, elapsed_time

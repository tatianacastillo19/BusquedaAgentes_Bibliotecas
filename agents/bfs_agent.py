"""
Agente de Búsqueda en Anchura (BFS).
Optimizado para grafos viales grandes con lista de adyacencia directa en memoria.
"""
import time
from collections import deque
from typing import Any, Dict, List, Set, Tuple
from core.map_loader import BogotaRoadMap, get_edge_cost
from core.search_tree import SearchTreeTrace, record_tree_state, start_tree_trace


def bfs_search(
    road_map: Any,
    origen_node: Any,
    destino_node: Any,
    tree_trace: SearchTreeTrace = None
) -> Tuple[List[Any], float, int, float]:
    """
    Ejecuta Búsqueda en Anchura (BFS) sobre la red vial.
    """
    start_time = time.perf_counter()
    root_trace_id = start_tree_trace(tree_trace, origen_node)

    if origen_node == destino_node:
        elapsed_time = time.perf_counter() - start_time
        return [origen_node], 0.0, 1, elapsed_time

    is_custom = isinstance(road_map, BogotaRoadMap)
    adj = road_map.adj if is_custom else None

    queue = deque([(origen_node, root_trace_id, 0.0)])
    visited: Set[Any] = {origen_node}
    parent_map: Dict[Any, Any] = {}
    nodes_evaluated = 0
    found = False

    while queue:
        current, current_trace_id, current_g = queue.popleft()
        nodes_evaluated += 1

        if current == destino_node:
            found = True
            break

        # Obtener vecinos de forma ultra-optimizada
        if is_custom:
            neighbors = adj.get(current, [])
            for neighbor, edge_weight in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    child_trace_id = None
                    next_g = current_g
                    if tree_trace is not None and current_trace_id is not None:
                        next_g += edge_weight
                        child_trace_id = record_tree_state(tree_trace, neighbor, current_trace_id, next_g)
                    queue.append((neighbor, child_trace_id, next_g))
        else:
            for neighbor in road_map.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    child_trace_id = None
                    next_g = current_g
                    if tree_trace is not None and current_trace_id is not None:
                        next_g += get_edge_cost(road_map, current, neighbor)
                        child_trace_id = record_tree_state(tree_trace, neighbor, current_trace_id, next_g)
                    queue.append((neighbor, child_trace_id, next_g))

    elapsed_time = time.perf_counter() - start_time

    if not found:
        return [], float("inf"), nodes_evaluated, elapsed_time

    # Reconstrucción del camino
    path = []
    curr = destino_node
    while curr is not None:
        path.append(curr)
        curr = parent_map.get(curr)
    path.reverse()

    # Cálculo del costo métrico
    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += get_edge_cost(road_map, path[i], path[i + 1])

    return path, total_cost, nodes_evaluated, elapsed_time

"""
Módulo de carga y optimización del mapa vial vehicular de Bogotá.
Implementa caché binario ultrarrápido (pickle) y preprocesamiento de listas de adyacencia
y segmentos de calles para renderizado instantáneo en Matplotlib y ejecución de algoritmos en milisegundos.
"""
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import osmnx as ox
import networkx as nx
from config.bibliotecas import BIBLIOTECAS


@dataclass
class BogotaRoadMap:
    """Estructura de datos optimizada en memoria para ruteo vehicular y renderizado cartográfico."""
    adj: Dict[Any, List[Tuple[Any, float]]]  # node -> [(neighbor, length_meters)]
    coords: Dict[Any, Tuple[float, float]]    # node -> (lat, lon)
    street_lines: List[List[Tuple[float, float]]]  # [ [(lon1, lat1), (lon2, lat2)], ... ]
    library_nodes: Dict[str, Any]             # "Biblioteca ..." -> nearest_node_id
    total_nodes: int
    total_edges: int


def _build_optimized_structure(G: nx.MultiDiGraph) -> BogotaRoadMap:
    """Preprocesa el grafo de NetworkX a estructuras indexadas en memoria pura."""
    print("[MapLoader] Indexando nodos, coordenadas y adyacencias...")
    coords: Dict[Any, Tuple[float, float]] = {}
    for node, data in G.nodes(data=True):
        coords[node] = (float(data["y"]), float(data["x"]))

    adj: Dict[Any, List[Tuple[Any, float]]] = {}
    street_lines: List[List[Tuple[float, float]]] = []

    # Recolectar aristas dirigidas y segmentos de calles
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 1.0))
        if u not in adj:
            adj[u] = []
        adj[u].append((v, length))

        # Almacenar coordenadas para dibujar el mapa de calles
        if u in coords and v in coords:
            u_lat, u_lon = coords[u]
            v_lat, v_lon = coords[v]
            street_lines.append([(u_lon, u_lat), (v_lon, v_lat)])

    # Asegurar que todos los nodos existan en adj
    for node in coords:
        if node not in adj:
            adj[node] = []

    # Precomputar nodos más cercanos para las 20 bibliotecas sin dependencias externas
    print("[MapLoader] Precomputando nodos de bibliotecas BibloRed...")
    library_nodes: Dict[str, Any] = {}
    node_keys = list(coords.keys())
    node_lat_lon = [coords[k] for k in node_keys]

    for name, data in BIBLIOTECAS.items():
        lib_lat, lib_lon = data["lat"], data["lon"]
        best_idx = min(
            range(len(node_lat_lon)),
            key=lambda i: (node_lat_lon[i][0] - lib_lat) ** 2 + (node_lat_lon[i][1] - lib_lon) ** 2
        )
        library_nodes[name] = node_keys[best_idx]

    return BogotaRoadMap(
        adj=adj,
        coords=coords,
        street_lines=street_lines,
        library_nodes=library_nodes,
        total_nodes=len(coords),
        total_edges=len(street_lines)
    )


def load_bogota_graph(
    place_name: str = "Bogotá, Colombia",
    cache_graphml: str = "bogota_drive.graphml",
    cache_fast: str = "bogota_fast_cache.pkl"
) -> BogotaRoadMap:
    """
    Carga el mapa vehicular de Bogotá.
    1. Si existe 'bogota_fast_cache.pkl', carga en milisegundos sin conexión.
    2. Si no existe pero está 'bogota_drive.graphml', procesa y crea el caché rápido.
    3. Si ninguno existe, descarga desde OpenStreetMap y genera ambos cachés.
    """
    # 1. Carga instantánea desde caché binario optimizado
    if os.path.exists(cache_fast):
        print(f"[MapLoader] Cargando mapa vehicular desde caché binario ultrarrápido: {cache_fast}...")
        try:
            with open(cache_fast, "rb") as f:
                road_map = pickle.load(f)
            print(f"[MapLoader] Mapa listo en memoria ({road_map.total_nodes:,} nodos, {road_map.total_edges:,} vías).")
            return road_map
        except Exception as e:
            print(f"[MapLoader] Error al leer {cache_fast}, regenerando desde GraphML... ({e})")

    # 2. Cargar desde GraphML existente
    if os.path.exists(cache_graphml):
        print(f"[MapLoader] Leyendo archivo local {cache_graphml}...")
        G = ox.load_graphml(filepath=cache_graphml)
    else:
        # 3. Descarga inicial si no existe ningún archivo
        print(f"[MapLoader] Descargando red vial de Bogotá desde OpenStreetMap...")
        ox.settings.use_cache = True
        G = ox.graph_from_place(place_name, network_type="drive", simplify=True)
        print(f"[MapLoader] Guardando copia en {cache_graphml}...")
        ox.save_graphml(G, filepath=cache_graphml)

    # Generar estructura rápida y guardar en disco
    road_map = _build_optimized_structure(G)
    print(f"[MapLoader] Guardando caché optimizado en {cache_fast}...")
    try:
        with open(cache_fast, "wb") as f:
            pickle.dump(road_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[MapLoader] Archivo de caché creado exitosamente.")
    except Exception as e:
        print(f"[MapLoader] Aviso: No se pudo escribir {cache_fast}: {e}")

    return road_map


def get_nearest_node(road_map: Any, lat: float, lon: float) -> Any:
    """Encuentra el nodo más cercano a partir de las coordenadas del mapa indexado."""
    if isinstance(road_map, BogotaRoadMap):
        best_node = None
        min_dist_sq = float("inf")
        for node, (n_lat, n_lon) in road_map.coords.items():
            d = (lat - n_lat) ** 2 + (lon - n_lon) ** 2
            if d < min_dist_sq:
                min_dist_sq = d
                best_node = node
        return best_node
    return ox.distance.nearest_nodes(road_map, X=lon, Y=lat)


def get_edge_cost(road_map: Any, u: Any, v: Any) -> float:
    """Retorna la distancia en metros entre el nodo u y el nodo v."""
    if isinstance(road_map, BogotaRoadMap):
        for neighbor, length in road_map.adj.get(u, []):
            if neighbor == v:
                return length
        return float("inf")

    edge_data = road_map.get_edge_data(u, v)
    if not edge_data:
        return float("inf")
    lengths = [data.get("length", 1.0) for data in edge_data.values()]
    return min(lengths) if lengths else 1.0

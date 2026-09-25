"""
Paquete de Agentes de Búsqueda Clásica para Ruteo Vehicular en BibloRed.
Incluye BFS, UCS, Greedy Best-First y A*.
"""
from .bfs_agent import bfs_search
from .ucs_agent import ucs_search
from .greedy_agent import greedy_search
from .astar_agent import astar_search

__all__ = ["bfs_search", "ucs_search", "greedy_search", "astar_search"]

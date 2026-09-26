"""Thread-safe in-memory cache for route-search results."""
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, Hashable, Optional, Tuple


RouteCacheKey = Tuple[Hashable, Hashable, str]
_cache: Dict[RouteCacheKey, Dict[str, Any]] = {}
_cache_lock = RLock()


def get_cached_route(origin: Hashable, destination: Hashable, algorithm: str) -> Optional[Dict[str, Any]]:
    """Return an independent copy of a cached result, if available."""
    key = (origin, destination, algorithm)
    with _cache_lock:
        result = _cache.get(key)
        return deepcopy(result) if result is not None else None


def cache_route_result(origin: Hashable, destination: Hashable, algorithm: str, result: Dict[str, Any]) -> None:
    """Store a copy so UI changes cannot mutate the cached search result."""
    key = (origin, destination, algorithm)
    with _cache_lock:
        _cache[key] = deepcopy(result)


def clear_route_cache() -> None:
    """Clear all memoized routes."""
    with _cache_lock:
        _cache.clear()
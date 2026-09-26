"""JSON persistence for completed and saved route records."""
import json
import os
import tempfile
import uuid
from datetime import datetime
from threading import RLock
from typing import Any, Dict, List, Optional


class RouteHistory:
    """Loads and safely updates the local route-history JSON file."""

    def __init__(self, file_path: Optional[str] = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = file_path or os.path.join(project_root, "historial_rutas.json")
        self._lock = RLock()
        if not os.path.exists(self.file_path):
            self._write_records([])

    def _read_records(self) -> List[Dict[str, Any]]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as history_file:
                records = json.load(history_file)
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        directory = os.path.dirname(self.file_path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                prefix=".historial_rutas_",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                json.dump(records, temporary_file, ensure_ascii=False, indent=2)
            os.replace(temporary_path, self.file_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    def list_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_records()

    def add_record(
        self,
        *,
        origin_name: str,
        destination_name: str,
        origin_node: Any,
        destination_node: Any,
        algorithm: str,
        distance_km: float,
        nodes_evaluated: int,
        path: List[Any],
        record_type: str,
    ) -> Dict[str, Any]:
        record = {
            "id": str(uuid.uuid4()),
            "fecha_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
            "origen": origin_name,
            "destino": destination_name,
            "nodo_origen": origin_node,
            "nodo_destino": destination_node,
            "algoritmo": algorithm,
            "distancia_km": distance_km,
            "nodos_evaluados": nodes_evaluated,
            "camino": path,
            "tipo": record_type,
        }
        with self._lock:
            records = self._read_records()
            records.insert(0, record)
            self._write_records(records)
        return record

    def delete_record(self, record_id: str) -> bool:
        with self._lock:
            records = self._read_records()
            remaining = [record for record in records if record.get("id") != record_id]
            if len(remaining) == len(records):
                return False
            self._write_records(remaining)
            return True

    def clear(self) -> None:
        with self._lock:
            self._write_records([])
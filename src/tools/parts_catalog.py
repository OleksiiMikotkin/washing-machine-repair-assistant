import csv
import os
from pathlib import Path

_DEFAULT_CSV = Path(__file__).parent.parent.parent / "data" / "parts.csv"

_TEXT_FIELDS = {"sku", "name", "brand", "aliases", "description", "category"}


class CatalogSearch:
    def __init__(self, csv_path: Path | str | None = None):
        path = Path(csv_path) if csv_path else Path(os.getenv("PARTS_CSV", str(_DEFAULT_CSV)))
        self._rows: list[dict] = self._load(path)

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def search(self, filters: dict | None = None, top_k: int = 5) -> list[dict]:
        """
        filters keys map to CSV columns; values are substring-matched (case-insensitive).
        'model' is special: matched against the pipe-separated 'compatibility' column.
        Example: {"name": "drain pump", "model": "WW60J"}
        """
        filters = {k: v.lower().strip() for k, v in (filters or {}).items() if v}
        model = filters.pop("model", None)

        results = []
        for row in self._rows:
            if model and not self._compatible(row, model):
                continue
            if all(self._field_matches(row, key, val) for key, val in filters.items()):
                results.append(dict(row))
            if len(results) >= top_k:
                break
        return results

    def get_by_sku(self, sku: str) -> dict | None:
        sku = sku.strip().upper()
        for row in self._rows:
            if row.get("sku", "").upper() == sku:
                return dict(row)
        return None

    @staticmethod
    def _field_matches(row: dict, key: str, value: str) -> bool:
        cell = row.get(key, "").lower()
        if key in _TEXT_FIELDS:
            return value in cell
        return cell == value

    def list_categories(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for row in self._rows:
            cat = row.get("category", "").strip()
            if cat and cat not in seen:
                seen.add(cat)
                result.append(cat)
        return sorted(result)

    def find_compatible_parts(self, model: str, top_k: int = 20) -> list[dict]:
        model = model.lower().strip()
        return [dict(row) for row in self._rows if self._compatible(row, model)][:top_k]

    @staticmethod
    def _compatible(row: dict, model: str) -> bool:
        return any(model in c.lower() for c in row.get("compatibility", "").split("|"))

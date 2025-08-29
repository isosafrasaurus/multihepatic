import os
import csv
import json
from typing import Dict, Any, Sequence

class CsvWriter:
    @staticmethod
    def write_dict(path: str, d: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "value"])
            for k, v in d.items():
                w.writerow([k, v])

    @staticmethod
    def write_rows(path: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in rows:
                w.writerow(r)


class JsonWriter:
    @staticmethod
    def write(path: str, d: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


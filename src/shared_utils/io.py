"""I/O + logging helpers (decoder-only research pipeline)."""
import csv, json, logging, os
from typing import List
import yaml

def setup_logging(name, level="INFO"):
    lg = logging.getLogger(name); lg.setLevel(getattr(logging, level))
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h)
    return lg

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(rows, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_csv(path, header, rows):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

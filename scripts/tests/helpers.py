"""Shared test helpers: path setup, module loading, synthetic xlsx fixtures."""

import importlib.util
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

CONFIG_PATH = os.path.join(SCRIPTS_DIR, "configs", "riddles_config.yaml")


def load_script(filename, modname):
    """Load a numbered pipeline script (e.g. '04_compute_vectors.py') as a module."""
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Canonical riddles header (matches the real benchmark files).
HEADER = [
    "Number", "Topic", "Author",
    "Riddle (Original Language)", "Riddle (English Translation)",
    "Hints (Optional)", "Comments (Optional)",
    "Reference Answer (Original)", "Reference Answer (English)",
]

# Reordered header (Bengali_India style: answers interleaved).
HEADER_REORDERED = [
    "Number", "Topic", "Author",
    "Riddle (Original Language)", "Reference Answer (Original)",
    "Riddle (English Translation)", "Reference Answer (English)",
]


def make_workbook(path, sheets):
    """
    Write an .xlsx with the given sheets.
    `sheets` = list of (sheet_name, list_of_rows); each row is a list of cells.
    """
    import openpyxl
    wb = openpyxl.Workbook()
    default = wb.active
    for i, (name, rows) in enumerate(sheets):
        ws = default if i == 0 else wb.create_sheet()
        ws.title = name
        for row in rows:
            ws.append(row)
    wb.save(path)

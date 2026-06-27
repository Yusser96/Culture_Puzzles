import os, tempfile, unittest
from src.shared_utils.io import (
    save_json, load_json, save_jsonl, load_jsonl, save_csv, ensure_dir, load_config,
)

class TestIO(unittest.TestCase):
    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json"); save_json({"a": 1, "u": "ünì"}, p)
            self.assertEqual(load_json(p), {"a": 1, "u": "ünì"})

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl"); rows = [{"i": 1}, {"i": 2}]
            save_jsonl(rows, p); self.assertEqual(load_jsonl(p), rows)

    def test_csv_and_ensure_dir(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "a/b"); ensure_dir(sub)
            self.assertTrue(os.path.isdir(sub))
            p = os.path.join(sub, "x.csv"); save_csv(p, ["k", "v"], [["a", 1]])
            self.assertEqual(open(p).read().splitlines()[0], "k,v")

if __name__ == "__main__":
    unittest.main()

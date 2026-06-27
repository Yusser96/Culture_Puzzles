import os, tempfile, unittest
from src.modules.metadata.build import build_metadata
from src.shared_utils.store import MetadataTable

class TestMetadata(unittest.TestCase):
    def test_build_from_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            pz = os.path.join(d, "puzzles", "Arabic_Egypt"); os.makedirs(pz)
            open(os.path.join(pz, "original.txt"), "w").write("مرحبا\n")
            import json
            with open(os.path.join(pz, "riddles.jsonl"), "w") as f:
                f.write(json.dumps({"riddle_original": "مرحبا", "topic": "Politics",
                                    "topic_key": "politics"}) + "\n")
            cfg = {"paths": {"raw_dir": d}, "lang_regions":
                   [{"key": "Arabic_Egypt", "base_language": "Arabic"}],
                   "topic_label_map": {"Politics": "politics"},
                   "canonical_topics": ["politics"]}
            df = build_metadata(cfg)
            self.assertEqual(list(df.columns), MetadataTable.COLUMNS)
            row = df.iloc[0]
            self.assertEqual(row["language"], "Arabic"); self.assertEqual(row["region"], "Egypt")
            self.assertEqual(row["topic_canonical"], "politics"); self.assertEqual(row["script"], "ARABIC")
            self.assertEqual(row["source"], "puzzles_original")

if __name__ == "__main__":
    unittest.main()

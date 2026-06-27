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

    def test_sample_ids_unique_across_cultural_topics(self):
        with tempfile.TemporaryDirectory() as d:
            # Create two cultural topics with identically-named region
            topics = ["politics", "food"]
            for topic in topics:
                topic_dir = os.path.join(d, "cultural", topic)
                os.makedirs(topic_dir, exist_ok=True)
                with open(os.path.join(topic_dir, "Arabic_Egypt.txt"), "w") as f:
                    f.write("Line 1\n")
                    f.write("Line 2\n")
            cfg = {
                "paths": {"raw_dir": d},
                "lang_regions": [{"key": "Arabic_Egypt", "base_language": "Arabic", "region": "Egypt"}],
                "topic_label_map": {},
                "canonical_topics": ["politics", "food"]
            }
            df = build_metadata(cfg)
            # Should have 4 cultural rows (2 topics x 2 lines each)
            cultural_df = df[df["source"] == "cultural"]
            self.assertEqual(len(cultural_df), 4, f"Expected 4 cultural rows, got {len(cultural_df)}")
            # All sample_ids should be unique
            self.assertTrue(df["sample_id"].is_unique, "sample_id column contains duplicates")
            # Verify sample_ids include topic
            sample_ids = set(cultural_df["sample_id"])
            self.assertTrue(any("politics" in sid for sid in sample_ids), "Expected 'politics' in sample_ids")
            self.assertTrue(any("food" in sid for sid in sample_ids), "Expected 'food' in sample_ids")

if __name__ == "__main__":
    unittest.main()

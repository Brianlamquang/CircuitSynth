from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.checkpoint import CheckpointStore, write_jsonl
from src.data import duplicate_count, leakage_count, stratified_sample
from src.schema import SchemaSpec
from src.student import split_silver
from src.verifier import verify_candidate


class PipelineInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    def test_sampling_duplicate_and_leakage_checks(self) -> None:
        rows = [
            {"source_id": f"id-{index}", "record_sha256": f"hash-{index}", "category": index % 2}
            for index in range(20)
        ]
        first = stratified_sample(rows, 8, ["category"], 42)
        second = stratified_sample(rows, 8, ["category"], 42)
        self.assertEqual(first, second)
        self.assertEqual(duplicate_count(first), 0)
        self.assertEqual(leakage_count({"train": first, "test": first[:1]}), 1)

    def test_verifier_rejects_unknown_zebra_clue(self) -> None:
        schema = SchemaSpec("z", "v1", "zebra_easy", {
            "grid_size": 1, "categories": {"person": ["A"]},
            "clues": [{"type": "unsupported", "left": {"category": "person", "item": "A"}}],
        })
        candidate = {"plan": {"domain": "zebra_easy", "assignment": {"person": {"A": 0}}}, "text": "person A position 0"}
        result = verify_candidate(candidate, schema)
        self.assertFalse(result.phi_valid)
        self.assertEqual(result.reject_reason, "unknown_clue")

    def test_group_aware_student_split(self) -> None:
        rows = []
        for source in range(10):
            rows.append({
                "source_id": f"s-{source}", "domain": "webnlg",
                "plan": {"domain": "webnlg", "triples": [{"subject": str(source), "predicate": "p", "object": "o"}]},
                "text": "text",
            })
        self.config["student"]["split"]["train_fraction"] = 0.8
        self.config["student"]["split"]["validation_fraction"] = 0.2
        train, validation = split_silver(rows, self.config)
        self.assertEqual((len(train), len(validation)), (8, 2))
        self.assertFalse({row["source_id"] for row in train} & {row["source_id"] for row in validation})

    def test_checkpoint_integrity_and_staleness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifacts" / "data" / "records.jsonl"
            write_jsonl(artifact, [{"id": 1}])
            store = CheckpointStore(root, self.config)
            store.save(0, "complete", [artifact], record_count=1)
            self.assertEqual(store.validate(0), (True, "complete"))
            changed = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
            changed["data"]["sample_fraction"] = 0.25
            self.assertEqual(CheckpointStore(root, changed).validate(0)[1], "stale scientific configuration")


if __name__ == "__main__":
    unittest.main()

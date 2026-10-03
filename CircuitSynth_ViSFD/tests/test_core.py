from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import yaml

from src.circuits import compile_and_fit, project_distribution, sample_plans
from src.data import _parse_visfd_label, generate_builtin_zebralogic, half_up_count, largest_remainder, normalize_record
from src.evaluate import bpt_and_perplexity, coverage_error, evaluate_outputs, evaluation_variant_budgets, jensen_shannon, marginal_jensen_shannon
from src.fsa import TokenFSA, _evaluation_records, controlled_realizations
from src.schema import SchemaSpec, schema_from_record, serialize_plan
from src.student import checkpoint_steps, deterministic_training_rows, dynamic_optimizer_steps
from src.teacher import plan_text_from_plan, silver_quotas


class CoreTests(unittest.TestCase):
    def test_counts_and_quota_allocation(self) -> None:
        self.assertEqual(half_up_count(13211, 0.5), 6606)
        self.assertEqual(half_up_count(1667, 0.5), 834)
        self.assertEqual(largest_remainder(30, {"a": 0.4, "b": 0.6}, ["a", "b"]), {"a": 12, "b": 18})
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(silver_quotas(config), {
            "webnlg": 0, "dart": 0, "zebra_easy": 0, "zebra_medium": 0, "zebra_hard": 0, "visfd": 700,
        })

    def test_builtin_zebra_generation_is_valid(self) -> None:
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        spec = dict(config["data"]["zebralogic"])
        spec["full_counts"] = {"easy": 3, "medium": 4, "hard": 3}
        rows = generate_builtin_zebralogic(spec, 42)
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["domain"] for row in rows}, {"zebra_easy", "zebra_medium", "zebra_hard"})
        for row in rows:
            schema = schema_from_record(row)
            self.assertEqual(schema.validate_plan(row["reference_plan"]), [])

    def test_schema_and_cnf(self) -> None:
        triple = {
            "subject": "Paris", "predicate": "locatedIn", "object": "France",
            "subject_type": "City", "object_type": "Country",
        }
        schema = SchemaSpec("web:test", "v1", "webnlg", {
            "allowed_predicates": ["locatedIn"], "allowed_triples": [triple],
            "domain_range": {"locatedIn": ["City", "Country"]},
            "cardinality": [1, 1], "disjoint_types": [],
        })
        self.assertEqual(schema.validate_plan({"domain": "webnlg", "triples": [dict(triple)]}), [])
        self.assertTrue(schema.to_cnf().clauses)
        invalid = {"domain": "webnlg", "triples": [{**triple, "predicate": "forbidden"}]}
        self.assertTrue(schema.validate_plan(invalid))

    def test_student_steps_and_checkpoints(self) -> None:
        self.assertEqual(dynamic_optimizer_steps(28500, 1, 1, 16), 1782)
        config = {"student": {"checkpoint": {
            "every_steps": 500, "explicit_steps": [500, 1000, 1500],
            "save_final": True, "strict_explicit_steps": False,
        }}}
        self.assertEqual(checkpoint_steps(config, 1782), [500, 1000, 1500, 1782])


    def test_fallback_realization_matches_constrained_surface(self) -> None:
        plans = [
            {"domain": "webnlg", "triples": [{"subject": "Paris", "predicate": "locatedIn", "object": "France"}]},
            {"domain": "dart", "schema_id": "d", "fields": [{"row": "A", "column": "manager", "value": "B"}]},
            {"domain": "zebra_easy", "assignment": {"name": {"Alice": 0, "Bob": 1}, "color": {"red": 1, "blue": 0}}},
            {"domain": "visfd", "pairs": [{"aspect": "BATTERY", "sentiment": "Negative"}], "others": False},
        ]
        for plan in plans:
            self.assertEqual(plan_text_from_plan(plan), controlled_realizations(plan)[0])

    def test_training_shuffle_is_deterministic_and_mixed(self) -> None:
        rows = [{"domain": domain, "id": index} for domain in ("dart", "webnlg", "zebra_easy") for index in range(20)]
        first = deterministic_training_rows(rows, 6042, True)
        second = deterministic_training_rows(rows, 6042, True)
        self.assertEqual(first, second)
        self.assertNotEqual(first, rows)
        self.assertGreater(len({row["domain"] for row in first[:12]}), 1)

    def test_fsa_and_metrics(self) -> None:
        automaton = TokenFSA.from_sequences([[4, 5], [4, 6]], eos_token_id=2)
        state = automaton.step(0, 4)
        self.assertEqual(automaton.next_tokens(state), {5, 6})
        with self.assertRaises(ValueError):
            automaton.step(state, 7)
        self.assertEqual(coverage_error(np.array([0.5]), np.array([0.5])), 0.0)
        self.assertEqual(jensen_shannon(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 0.0)
        bpt, ppl = bpt_and_perplexity([-np.log(2.0)])
        self.assertAlmostEqual(bpt or 0.0, 1.0)
        self.assertAlmostEqual(ppl or 0.0, 2.0)

    def test_variant_budgets_and_weighted_prefix(self) -> None:
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        budgets = evaluation_variant_budgets(config)
        self.assertEqual(budgets["full_pipeline"], 200)
        self.assertEqual(budgets["distill_only"], 20)
        records = [{"source_id": f"visfd-{index}", "domain": "visfd", "official_split": "test"} for index in range(30)]
        selected = _evaluation_records(config, records, 30)
        self.assertEqual(len(selected), 30)
        self.assertEqual({row["domain"] for row in selected}, {"visfd"})

    def test_paper_metric_report_contains_all_tables(self) -> None:
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        config["evaluation"]["variants"] = {name: name == "full_pipeline" for name in config["evaluation"]["variants"]}
        config["evaluation"]["variant_budgets"]["full_pipeline"] = 1
        triple = {"subject": "Paris", "predicate": "locatedIn", "object": "France", "subject_type": "City", "object_type": "Entity"}
        schema = SchemaSpec("web:a", "v1", "webnlg", {
            "allowed_predicates": ["locatedIn"], "allowed_triples": [triple],
            "domain_range": {"locatedIn": ["City", "Entity"]}, "cardinality": [1, 1], "disjoint_types": [],
        })
        accepted = [{"source_id": "a", "plan": {"domain": "webnlg", "triples": [triple]}}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            circuit = compile_and_fit(config, root, accepted, {"a": schema})["circuit"]
            text = "Paris locatedIn France."
            outputs = [{
                "variant": "full_pipeline", "eval_id": "eval-a", "generation_seed": 1, "generation_index": 0,
                "source_id": "a", "domain": "webnlg", "dataset": "webnlg", "evaluation_group": "City",
                "plan": {"domain": "webnlg", "triples": [triple]}, "text": text,
                "token_log_probabilities": [-0.1, -0.2], "generation_ms": 5.0,
                "validation": {
                    "parse_valid": True, "schema_valid": True, "plan_consistent": True,
                    "required_entities_present": True, "required_relations_present": True, "fsa_accepted": True,
                },
            }]
            report = evaluate_outputs(config, root, outputs, circuit, records=[])["report"]
            block = report["paper_table1_full_pipeline_by_dataset"]["webnlg"]
            self.assertEqual(block["schema_validity"], 1.0)
            self.assertEqual(block["factuality_precision"], 1.0)
            for key in ("schema_validity", "factuality_precision", "coverage_error", "jensen_shannon", "rare_combination_coverage", "bits_per_token", "perplexity"):
                self.assertIn(key, block)
            self.assertIn("full_pipeline", report["paper_table3_ablation"])

    def test_circuit_mle_projection_and_sampling(self) -> None:
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        config["sampling"]["total_plans"] = 12
        triple_a = {"subject": "Paris", "predicate": "locatedIn", "object": "France", "subject_type": "City", "object_type": "Entity"}
        triple_b = {"subject": "Rome", "predicate": "locatedIn", "object": "Italy", "subject_type": "City", "object_type": "Entity"}
        schemas = {
            "a": SchemaSpec("web:a", "v1", "webnlg", {"allowed_predicates": ["locatedIn"], "allowed_triples": [triple_a], "domain_range": {"locatedIn": ["City", "Entity"]}, "cardinality": [1, 1], "disjoint_types": []}),
            "b": SchemaSpec("web:b", "v1", "webnlg", {"allowed_predicates": ["locatedIn"], "allowed_triples": [triple_b], "domain_range": {"locatedIn": ["City", "Entity"]}, "cardinality": [1, 1], "disjoint_types": []}),
        }
        accepted = [
            {"source_id": "a", "plan": {"domain": "webnlg", "triples": [triple_a]}},
            {"source_id": "a", "plan": {"domain": "webnlg", "triples": [triple_a]}},
            {"source_id": "b", "plan": {"domain": "webnlg", "triples": [triple_b]}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted = compile_and_fit(config, root, accepted, schemas)["circuit"]
            self.assertAlmostEqual(float(fitted.probabilities.sum()), 1.0)
            self.assertEqual(fitted.probability({"domain": "webnlg", "triples": []}), 0.0)
            projected = project_distribution(config, root, fitted)
            self.assertEqual(projected["report"]["status"], "complete")
            sampled = sample_plans(config, root, projected["circuit"], schemas)
            self.assertEqual(len(sampled["plans"]), 12)
            self.assertEqual(sampled["report"]["schema_validity"], 1.0)

    def test_visfd_label_parser_and_schema(self) -> None:
        pairs, others = _parse_visfd_label("{CAMERA#Positive};{BATTERY#Negative};{OTHERS};")
        self.assertTrue(others)
        self.assertEqual({(row["aspect"], row["sentiment"]) for row in pairs}, {("CAMERA", "Positive"), ("BATTERY", "Negative")})
        raw = {
            "index": 7, "type": "train", "comment": "Pin yếu nhưng camera ổn.", "n_star": 3,
            "label": "{CAMERA#Positive};{BATTERY#Negative};",
        }
        record = normalize_record("visfd", "train", 7, raw)
        schema = schema_from_record(record)
        self.assertEqual(schema.validate_plan(record["reference_plan"]), [])
        invalid = {"domain": "visfd", "pairs": [{"aspect": "BATTERY", "sentiment": "Positive"}], "others": False}
        self.assertTrue(schema.validate_plan(invalid))
        self.assertIn("BATTERY", serialize_plan(record["reference_plan"]))

    def test_visfd_cnf_and_controlled_realization(self) -> None:
        raw = {
            "index": 8, "comment": "Giá tốt.", "n_star": 5, "label": "{PRICE#Positive};",
        }
        record = normalize_record("visfd", "train", 8, raw)
        schema = schema_from_record(record)
        cnf = schema.to_cnf()
        self.assertGreater(cnf.num_variables, 0)
        text = controlled_realizations(record["reference_plan"])[0]
        self.assertIn("giá", text.casefold())
        self.assertIn("tích cực", text.casefold())

    def test_visfd_circuit_projection_and_sampling(self) -> None:
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        config["sampling"]["total_plans"] = 4
        raws = [
            {"index": 1, "comment": "Pin yếu.", "n_star": 2, "label": "{BATTERY#Negative};"},
            {"index": 2, "comment": "Camera tốt.", "n_star": 5, "label": "{CAMERA#Positive};"},
        ]
        records = [normalize_record("visfd", "train", i, raw) for i, raw in enumerate(raws)]
        schemas = {row["source_id"]: schema_from_record(row) for row in records}
        accepted = [
            {"source_id": row["source_id"], "plan": row["reference_plan"]}
            for row in records
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted = compile_and_fit(config, root, accepted, schemas)["circuit"]
            projected = project_distribution(config, root, fitted)["circuit"]
            sampled = sample_plans(config, root, projected, schemas)
            self.assertEqual(len(sampled["plans"]), 4)
            self.assertEqual(sampled["report"]["schema_validity"], 1.0)
            self.assertEqual({row["domain"] for row in sampled["plans"]}, {"visfd"})


if __name__ == "__main__":
    unittest.main()


def test_teacher_normalization_rejects_malformed_nested_list_without_crash():
    from src.teacher import normalize_teacher_output
    from src.schema import SchemaSpec

    schema = SchemaSpec(
        schema_id="web:test", version="1", domain="webnlg",
        constraints={"allowed_predicates": [], "cardinality": [1, 9]},
    )
    assert normalize_teacher_output('{"triples": [["s", "p", "o"]]}', schema) == "null"


def test_teacher_normalization_rejects_malformed_zebra_assignment_without_crash():
    from src.teacher import normalize_teacher_output
    from src.schema import SchemaSpec

    schema = SchemaSpec(
        schema_id="z:test", version="1", domain="zebra_easy",
        constraints={"grid_size": 2, "categories": {"person": ["a", "b"]}, "clues": []},
    )
    raw = '{"plan":{"domain":"zebra_easy","assignment":{"person":[0,1]}},"text":"x"}'
    assert normalize_teacher_output(raw, schema) == "null"


def test_marginal_js_tolerates_tiny_probability_roundoff():
    observed = np.array([0.0, 0.5, 1.0])
    target = np.array([-1e-12, 0.5, 1.0 + 1e-12])
    value = marginal_jensen_shannon(observed, target)
    assert np.isfinite(value)
    assert value >= 0.0


def test_marginal_js_rejects_materially_invalid_probability():
    observed = np.array([0.5])
    target = np.array([1.01])
    try:
        marginal_jensen_shannon(observed, target)
    except ValueError as error:
        assert "marginals must lie in [0, 1]" in str(error)
    else:
        raise AssertionError("expected materially invalid marginal to be rejected")


def test_visfd_rejects_foreign_script_text():
    from src.schema import SchemaSpec
    from src.verifier import verify_candidate

    plan = {"domain": "visfd", "pairs": [{"aspect": "BATTERY", "sentiment": "Positive"}], "others": False}
    schema = SchemaSpec(
        schema_id="visfd:test", version="1", domain="visfd",
        constraints={
            "allowed_aspects": ["BATTERY"],
            "allowed_sentiments": ["Positive", "Neutral", "Negative"],
            "required_pairs": [["BATTERY", "Positive"]],
            "required_others": False,
        },
    )
    good = verify_candidate({"plan": plan, "text": "pin có đánh giá tích cực."}, schema)
    assert good.factual_consistency
    bad = verify_candidate({"plan": plan, "text": "pin có đánh giá tích cực. 请继续。"}, schema)
    assert not bad.factual_consistency
    assert bad.reject_reason == "language_violation"


def test_visfd_controlled_realizations_are_vietnamese_script_safe():
    from src.fsa import controlled_realizations
    from src.verifier import visfd_language_valid

    plan = {
        "domain": "visfd",
        "pairs": [
            {"aspect": "BATTERY", "sentiment": "Positive"},
            {"aspect": "SCREEN", "sentiment": "Negative"},
        ],
        "others": True,
    }
    outputs = controlled_realizations(plan)
    assert outputs
    assert all(visfd_language_valid(text) for text in outputs)

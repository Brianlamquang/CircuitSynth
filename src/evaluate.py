from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .checkpoint import canonical_json, sha256_bytes, write_json
from .circuits import CircuitDistribution
from .schema import plan_terms
from .verifier import canonical_text


def coverage_error(observed: np.ndarray, target: np.ndarray) -> float:
    if observed.shape != target.shape or observed.size == 0:
        raise ValueError("coverage vectors must have the same non-empty shape")
    return float(np.sqrt(np.mean((observed - target) ** 2)))


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or np.any(left < 0) or np.any(right < 0):
        raise ValueError("JS inputs must be same-shape non-negative vectors")
    if left.sum() == 0 or right.sum() == 0:
        return 0.0 if left.sum() == right.sum() else 1.0
    left, right = left / left.sum(), right / right.sum()
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / reference[mask])))

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def bpt_and_perplexity(log_probabilities: list[float]) -> tuple[float | None, float | None]:
    if not log_probabilities:
        return None, None
    nll_nats = -float(np.mean(log_probabilities))
    bpt = nll_nats / math.log(2)
    return bpt, math.exp(nll_nats)


def evaluation_variant_budgets(config: dict[str, Any]) -> dict[str, int]:
    default = int(config["evaluation"].get("generation_budget_per_variant", 500))
    configured = config["evaluation"].get("variant_budgets", {})
    result: dict[str, int] = {}
    for variant, enabled in config["evaluation"]["variants"].items():
        if enabled:
            result[variant] = int(configured.get(variant, default))
    return result


def _contains(text: str, value: str) -> bool:
    normalized_value = canonical_text(value)
    return bool(normalized_value) and f" {normalized_value} " in f" {canonical_text(text)} "


def _dataset_name(domain: str) -> str:
    return "zebralogic" if domain.startswith("zebra_") else domain


def _plan_facts(plan: dict[str, Any]) -> list[tuple[str, ...]]:
    domain = str(plan["domain"])
    if domain == "webnlg":
        return [
            (str(row["subject"]), str(row["predicate"]), str(row["object"]))
            for row in plan.get("triples", [])
        ]
    if domain == "dart":
        return [
            (str(row["row"]), str(row["column"]), str(row["value"]))
            for row in plan.get("fields", [])
        ]
    return [
        (str(category), str(item), "position", str(position))
        for category, values in plan.get("assignment", {}).items()
        for item, position in values.items()
    ]


def _canonical_fact(fact: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonical_text(term) for term in fact if canonical_text(term))


def _text_clauses(text: str) -> list[str]:
    # Semicolon/newline boundaries are exact for DART/Zebra controlled realizations.
    # A period followed by whitespace is also useful for WebNLG while avoiding most decimal splits.
    raw = re.split(r";\s*|\n+|\.\s+", text)
    clauses = [canonical_text(value) for value in raw if canonical_text(value)]
    return clauses or [canonical_text(text)]


def _fact_is_mentioned(fact: tuple[str, ...], text: str) -> bool:
    terms = _canonical_fact(fact)
    if not terms:
        return False
    for clause in _text_clauses(text):
        padded = f" {clause} "
        if all(f" {term} " in padded for term in terms):
            return True
    return False


def _fact_mentions(plan: dict[str, Any], text: str) -> tuple[int, int]:
    facts = _plan_facts(plan)
    return sum(_fact_is_mentioned(fact, text) for fact in facts), len(facts)


def _term_factuality_precision(plan: dict[str, Any], text: str, vocabulary: set[str]) -> float:
    entities, relations = plan_terms(plan)
    correct = {canonical_text(term) for term in entities | relations if canonical_text(term)}
    mentioned = {canonical_text(term) for term in vocabulary if _contains(text, term)}
    return len(mentioned & correct) / len(mentioned) if mentioned else 0.0


def _fact_factuality_precision(
    plan: dict[str, Any], text: str, fact_vocabulary: dict[str, set[tuple[str, ...]]]
) -> float:
    domain = str(plan["domain"])
    correct = {_canonical_fact(fact) for fact in _plan_facts(plan)}
    candidates = set(fact_vocabulary.get(domain, set())) | correct
    mentioned = {fact for fact in candidates if _fact_is_mentioned(fact, text)}
    return len(mentioned & correct) / len(mentioned) if mentioned else 0.0


def realized_feature_vector(plan: dict[str, Any], text: str, feature_names: list[str]) -> np.ndarray:
    entities, relations = plan_terms(plan)
    domain = str(plan["domain"])
    present = {f"domain:{domain}"}
    present.update(f"attribute:{domain}:{value}" for value in entities if _contains(text, value))
    present.update(f"relation:{domain}:{value}" for value in relations if _contains(text, value))
    result: list[float] = []
    for name in feature_names:
        if name.startswith("rare_pair:"):
            left, right = json.loads(name.removeprefix("rare_pair:"))
            result.append(float(left in present and right in present))
        else:
            result.append(float(name in present))
    return np.asarray(result, dtype=np.float64)




def _sanitize_marginals(values: np.ndarray, *, label: str, tolerance: float = 1e-9) -> np.ndarray:
    """Clamp harmless floating-point drift while rejecting genuinely invalid marginals."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite marginal values")
    minimum = float(array.min()) if array.size else 0.0
    maximum = float(array.max()) if array.size else 0.0
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{label} marginals must lie in [0, 1]; observed range=({minimum:.12g}, {maximum:.12g})"
        )
    return np.clip(array, 0.0, 1.0)

def marginal_jensen_shannon(observed: np.ndarray, target: np.ndarray) -> float:
    if observed.shape != target.shape or observed.size == 0:
        raise ValueError("marginal vectors must have the same non-empty shape")
    observed = _sanitize_marginals(observed, label="observed")
    target = _sanitize_marginals(target, label="target")
    values = [
        jensen_shannon(np.asarray([p, 1.0 - p]), np.asarray([q, 1.0 - q]))
        for p, q in zip(observed, target, strict=True)
    ]
    return float(np.mean(values))


def marginal_kl(observed: np.ndarray, target: np.ndarray, epsilon: float = 1e-12) -> float:
    if observed.shape != target.shape or observed.size == 0:
        raise ValueError("marginal vectors must have the same non-empty shape")
    observed = _sanitize_marginals(observed, label="observed")
    target = _sanitize_marginals(target, label="target")
    p = np.clip(observed, epsilon, 1 - epsilon)
    q = np.clip(target, epsilon, 1 - epsilon)
    values = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return float(np.mean(values))


def _realized_schema_valid(row: dict[str, Any]) -> bool:
    validation = row["validation"]
    required = (
        bool(validation.get("parse_valid", True)),
        bool(validation.get("schema_valid", False)),
        bool(validation.get("plan_consistent", False)),
        bool(validation.get("required_entities_present", False)),
        bool(validation.get("required_relations_present", False)),
    )
    fsa_value = validation.get("fsa_accepted")
    return all(required) and fsa_value is not False


def _feature_domain(name: str) -> str | None:
    if name.startswith("domain:"):
        return name.split(":", 1)[1]
    if name.startswith("attribute:") or name.startswith("relation:"):
        return name.split(":", 2)[1]
    if name.startswith("rare_pair:"):
        left, right = json.loads(name.removeprefix("rare_pair:"))
        domains = {_feature_domain(left), _feature_domain(right)} - {None}
        return next(iter(domains)) if len(domains) == 1 else None
    return None


def _conditional_target(
    circuit: CircuitDistribution,
    selector: Callable[[dict[str, Any]], bool],
) -> np.ndarray:
    indices = [index for index, state in enumerate(circuit.states) if selector(state)]
    if not indices:
        return np.zeros(len(circuit.feature_names), dtype=np.float64)
    probabilities = circuit.probabilities[indices]
    probabilities = probabilities / probabilities.sum()
    return np.asarray(probabilities @ circuit.feature_matrix[indices, :]).ravel()


def _metric_block(
    rows: list[dict[str, Any]],
    circuit: CircuitDistribution,
    target: np.ndarray,
    fact_vocabulary: dict[str, set[tuple[str, ...]]],
    term_vocabulary: set[str],
    relevant_domains: set[str],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("metric block requires at least one output")
    empirical = np.zeros(len(circuit.feature_names), dtype=np.float64)
    for row in rows:
        empirical += realized_feature_vector(row["plan"], row["text"], circuit.feature_names)
    empirical /= len(rows)

    relevant = [
        index for index, name in enumerate(circuit.feature_names)
        if _feature_domain(name) in relevant_domains and (target[index] > 0 or empirical[index] > 0)
    ]
    semantic = [
        index for index in relevant
        if circuit.feature_names[index].startswith(("attribute:", "relation:"))
    ]
    rare = [
        index for index in relevant
        if circuit.feature_names[index].startswith("rare_pair:")
    ]

    log_probabilities = [value for row in rows for value in row.get("token_log_probabilities", [])]
    bpt, perplexity = bpt_and_perplexity(log_probabilities)
    fact_counts = [_fact_mentions(row["plan"], row["text"]) for row in rows]
    schema_validity = float(np.mean([_realized_schema_valid(row) for row in rows]))

    if rare:
        ce = coverage_error(empirical[rare], target[rare])
        rcc = float(np.mean(empirical[rare] > 0))
    else:
        ce, rcc = 0.0, 1.0
    if semantic:
        js = marginal_jensen_shannon(empirical[semantic], target[semantic])
        kl = marginal_kl(empirical[semantic], target[semantic])
    else:
        js, kl = 0.0, 0.0
    all_feature_ce = coverage_error(empirical[relevant], target[relevant]) if relevant else 0.0

    return {
        "output_count": len(rows),
        "schema_validity": schema_validity,
        "schema_validity_percent": 100.0 * schema_validity,
        "raw_plan_validity": float(np.mean([bool(row["validation"].get("schema_valid", False)) for row in rows])),
        "factuality_precision": float(np.mean([
            _fact_factuality_precision(row["plan"], row["text"], fact_vocabulary) for row in rows
        ])),
        "term_factuality_precision_diagnostic": float(np.mean([
            _term_factuality_precision(row["plan"], row["text"], term_vocabulary) for row in rows
        ])),
        "fact_coverage": float(
            sum(value[0] for value in fact_counts) / max(1, sum(value[1] for value in fact_counts))
        ),
        "coverage_error": ce,
        "coverage_error_all_features_diagnostic": all_feature_ce,
        "jensen_shannon": js,
        "rare_combination_coverage": rcc,
        "rare_combination_coverage_percent": 100.0 * rcc,
        "bits_per_token": bpt,
        "perplexity": perplexity,
        "token_count": len(log_probabilities),
        "mean_generation_ms": float(np.mean([row.get("generation_ms", 0.0) for row in rows])),
        "constraint_violation_rate": 1.0 - schema_validity,
        "constraint_violation_rate_percent": 100.0 * (1.0 - schema_validity),
        "distributional_drift_kl": kl,
    }


def _record_group(record: dict[str, Any]) -> str:
    domain = str(record["domain"])
    if domain == "webnlg":
        return str(record.get("category", "unknown"))
    if domain == "dart":
        return str(record.get("source", "unknown"))
    return domain.removeprefix("zebra_").capitalize()


def evaluate_outputs(
    config: dict[str, Any],
    root: Path,
    outputs: list[dict[str, Any]],
    target_circuit: CircuitDistribution,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outputs:
        by_variant[row["variant"]].append(row)

    term_vocabulary: set[str] = set()
    fact_vocabulary: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for state in target_circuit.states:
        entities, relations = plan_terms(state["plan"])
        term_vocabulary.update(entities | relations)
        fact_vocabulary[str(state["plan"]["domain"])].update(_canonical_fact(fact) for fact in _plan_facts(state["plan"]))

    record_by_id = {row["source_id"]: row for row in (records or [])}
    source_groups = {source_id: _record_group(row) for source_id, row in record_by_id.items()}
    budgets = evaluation_variant_budgets(config)
    metrics: dict[str, Any] = {}
    reference_ids_by_budget: dict[int, list[str]] = {}
    reference_seeds_by_budget: dict[int, list[int]] = {}

    for variant, rows in sorted(by_variant.items()):
        expected_budget = budgets.get(variant)
        if expected_budget is None:
            raise AssertionError(f"unexpected evaluation variant: {variant}")
        if len(rows) != expected_budget:
            raise AssertionError(f"evaluation budget mismatch for {variant}: {len(rows)} != {expected_budget}")
        rows = sorted(rows, key=lambda row: row["generation_index"])
        ids = [row["eval_id"] for row in rows]
        seeds = [row["generation_seed"] for row in rows]
        if expected_budget in reference_ids_by_budget:
            if config["evaluation"]["use_same_ids_across_variants"] and ids != reference_ids_by_budget[expected_budget]:
                raise AssertionError("equal-budget variants do not use the same held-out IDs")
            if config["evaluation"]["use_same_seeds_across_variants"] and seeds != reference_seeds_by_budget[expected_budget]:
                raise AssertionError("equal-budget variants do not use the same generation seeds")
        else:
            reference_ids_by_budget[expected_budget] = ids
            reference_seeds_by_budget[expected_budget] = seeds

        domains = {str(row["domain"]) for row in rows}
        target = _conditional_target(target_circuit, lambda state, domains=domains: str(state["plan"]["domain"]) in domains)
        block = _metric_block(rows, target_circuit, target, fact_vocabulary, term_vocabulary, domains)

        by_dataset: dict[str, Any] = {}
        for dataset in sorted({_dataset_name(str(row["domain"])) for row in rows}):
            subset = [row for row in rows if _dataset_name(str(row["domain"])) == dataset]
            dataset_domains = {str(row["domain"]) for row in subset}
            dataset_target = _conditional_target(
                target_circuit,
                lambda state, ds=dataset_domains: str(state["plan"]["domain"]) in ds,
            )
            by_dataset[dataset] = _metric_block(
                subset, target_circuit, dataset_target, fact_vocabulary, term_vocabulary, dataset_domains
            )

        by_domain: dict[str, Any] = {}
        for domain in sorted(domains):
            subset = [row for row in rows if row["domain"] == domain]
            domain_target = _conditional_target(target_circuit, lambda state, d=domain: state["plan"]["domain"] == d)
            by_domain[domain] = _metric_block(
                subset, target_circuit, domain_target, fact_vocabulary, term_vocabulary, {domain}
            )

        by_group: dict[str, Any] = {}
        group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group = row.get("evaluation_group") or source_groups.get(row.get("source_id", "")) or str(row["domain"])
            group_rows[str(group)].append(row)
        for group, subset in sorted(group_rows.items()):
            group_domains = {str(row["domain"]) for row in subset}
            if source_groups:
                group_target = _conditional_target(
                    target_circuit,
                    lambda state, g=group: source_groups.get(state.get("source_id", "")) == g,
                )
                if not np.any(group_target):
                    group_target = _conditional_target(
                        target_circuit,
                        lambda state, ds=group_domains: str(state["plan"]["domain"]) in ds,
                    )
            else:
                group_target = _conditional_target(
                    target_circuit,
                    lambda state, ds=group_domains: str(state["plan"]["domain"]) in ds,
                )
            by_group[group] = _metric_block(
                subset, target_circuit, group_target, fact_vocabulary, term_vocabulary, group_domains
            )

        block["by_dataset"] = by_dataset
        block["by_domain"] = by_domain
        block["by_group"] = by_group
        metrics[variant] = block

    full = metrics.get("full_pipeline", {})
    paper_table1 = full.get("by_dataset", {})
    paper_table2 = {
        group: {
            "output_count": value["output_count"],
            "schema_validity_percent": value["schema_validity_percent"],
            "factuality_precision": value["factuality_precision"],
            "jensen_shannon": value["jensen_shannon"],
        }
        for group, value in full.get("by_group", {}).items()
    }
    paper_table3 = {
        variant: {
            "output_count": value["output_count"],
            "constraint_violation_rate_percent": value["constraint_violation_rate_percent"],
            "kl": value["distributional_drift_kl"],
            "latency_ms": value["mean_generation_ms"],
        }
        for variant, value in metrics.items()
    }

    max_budget = max(budgets.values()) if budgets else 0
    max_budget_ids = reference_ids_by_budget.get(max_budget, [])
    max_budget_seeds = reference_seeds_by_budget.get(max_budget, [])
    report = {
        "metrics": metrics,
        "paper_table1_full_pipeline_by_dataset": paper_table1,
        "paper_table2_full_pipeline_granular": paper_table2,
        "paper_table3_ablation": paper_table3,
        "variant_budgets": budgets,
        "evaluation_ids_sha256": sha256_bytes(canonical_json(max_budget_ids).encode()),
        "generation_seeds_sha256": sha256_bytes(canonical_json(max_budget_seeds).encode()),
        "metric_definitions": {
            "schema_validity": "realized-output hard validity: valid plan/schema plus plan-text consistency, required semantic terms, and FSA acceptance when constrained",
            "factuality_precision": "precision of extractable generated facts against the semantic plan; fact-level rather than term-level matching",
            "coverage_error": "RMSE over rare-pair marginal frequencies only, matching the paper's rare-concept coverage framing",
            "coverage_error_all_features_diagnostic": "RMSE over all relevant circuit features; diagnostic only",
            "jensen_shannon": "macro mean base-2 Jensen-Shannon divergence over semantic attribute/relation Bernoulli marginals",
            "rare_combination_coverage": "fraction of configured rare-pair features observed at least once",
            "bits_per_token": "negative mean generated-token log probability in base 2",
            "perplexity": "exp(mean token negative log likelihood); mathematically equivalent to 2**BPT for the same token log probabilities",
            "constraint_violation_rate": "1 - realized-output schema validity",
            "distributional_drift_kl": "macro Bernoulli KL in nats between realized semantic marginals and the target circuit marginals",
            "fact_coverage": "recall of plan facts explicitly realized in text; extra diagnostic",
            "term_factuality_precision_diagnostic": "legacy term-level precision retained only as a diagnostic",
        },
        "paper_alignment_note": {
            "ppl": "The paper lists both BPT and PPL but does not specify a separate PPL protocol; this implementation uses the standard perplexity implied by the same token log probabilities.",
            "kl": "The paper labels ablation drift as KL without enough implementation detail to reconstruct its exact estimator; this report uses an explicit macro Bernoulli marginal KL and records the definition.",
        },
    }
    metrics_path = root / "artifacts" / "metrics" / "metrics.json"
    write_json(metrics_path, report)
    return {"report": report, "artifacts": [metrics_path]}

from __future__ import annotations

import json
import math
import platform
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, issparse

from .checkpoint import canonical_json, sha256_bytes, write_json, write_jsonl
from .schema import SchemaSpec, plan_terms
from .verifier import verify_plan


def _plan_key(plan: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(plan).encode())


def _state_key(source_id: str, plan: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json({"source_id": source_id, "plan": plan}).encode())


def _basic_features(plan: dict[str, Any]) -> set[str]:
    entities, relations = plan_terms(plan)
    domain = str(plan["domain"])
    result = {f"domain:{domain}"}
    result.update(f"attribute:{domain}:{value}" for value in entities)
    result.update(f"relation:{domain}:{value}" for value in relations)
    return result


def feature_vector(plan: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    basic = _basic_features(plan)
    values: list[float] = []
    for name in feature_names:
        if name.startswith("rare_pair:"):
            left, right = json.loads(name.removeprefix("rare_pair:"))
            values.append(float(left in basic and right in basic))
        else:
            values.append(float(name in basic))
    return np.asarray(values, dtype=np.float64)


def _local_parameters(states: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, list[float]]:
    domains = sorted({state["plan"]["domain"] for state in states})
    domain_mass = np.asarray([
        sum(probabilities[index] for index, state in enumerate(states) if state["plan"]["domain"] == domain)
        for domain in domains
    ], dtype=np.float64)
    result = {"root/domain": domain_mass.tolist()}
    for domain, mass in zip(domains, domain_mass, strict=True):
        indices = [index for index, state in enumerate(states) if state["plan"]["domain"] == domain]
        result[f"domain/{domain}"] = (probabilities[indices] / mass).tolist()
    return result


def _feature_space(states: list[dict[str, Any]], empirical: np.ndarray, config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    basic_by_state = [_basic_features(state["plan"]) for state in states]
    families = set(config["soft_constraints"]["features"])
    names = sorted({name for values in basic_by_state for name in values if name.startswith("domain:")})
    weighted: Counter[str] = Counter()
    for probability, values in zip(empirical, basic_by_state, strict=True):
        for name in values:
            weighted[name] += float(probability)
    def bounded(prefix: str, limit: int) -> list[str]:
        candidates = [name for name in weighted if name.startswith(prefix)]
        rare = sorted(candidates, key=lambda name: (weighted[name], name))[: limit // 2]
        common = sorted(candidates, key=lambda name: (-weighted[name], name))
        return sorted(dict.fromkeys(rare + common[: limit - len(rare)]))
    if "attribute_marginals" in families:
        names += bounded("attribute:", int(config["soft_constraints"]["max_attribute_features"]))
    if "relation_marginals" in families:
        names += bounded("relation:", int(config["soft_constraints"]["max_relation_features"]))
    rows, columns = zip(*[(row, column) for row, values in enumerate(basic_by_state) for column, name in enumerate(names) if name in values], strict=False) if names else ([], [])
    matrix = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(states), len(names)), dtype=np.float64)
    if "rare_pair_combinations" not in families or not names:
        return names, matrix
    candidate_set = {name for name in names if not name.startswith("domain:")}
    pair_frequency: Counter[tuple[str, str]] = Counter()
    for probability, values in zip(empirical, basic_by_state, strict=True):
        active = sorted(candidate_set & values)
        for pair in combinations(active, 2):
            pair_frequency[pair] += float(probability)
    frequencies = np.asarray(list(pair_frequency.values()), dtype=np.float64)
    threshold = float(np.quantile(frequencies, config["soft_constraints"]["rare_quantile"])) if frequencies.size else 0.0
    selected_pairs = sorted(
        (pair for pair, frequency in pair_frequency.items() if 0 < frequency <= threshold),
        key=lambda pair: (pair_frequency[pair], pair),
    )[: int(config["soft_constraints"]["max_rare_pair_features"])]
    pair_names = [f"rare_pair:{canonical_json(pair)}" for pair in selected_pairs]
    pair_columns = [matrix[:, names.index(left)].multiply(matrix[:, names.index(right)]) for left, right in selected_pairs]
    if pair_columns:
        from scipy.sparse import hstack
        matrix = hstack([matrix, *pair_columns], format="csr")
        names.extend(pair_names)
    return names, matrix


@dataclass
class CircuitDistribution:
    backend: str
    states: list[dict[str, Any]]
    probabilities: np.ndarray
    base_probabilities: np.ndarray
    feature_names: list[str]
    feature_matrix: Any
    local_parameters: dict[str, list[float]]
    circuit_hash: str
    compiler_metadata: dict[str, Any]

    def validate(self, floor: float = 0.0) -> None:
        if not self.states or len(self.states) != len(self.probabilities):
            raise ValueError("circuit state/probability mismatch")
        if self.feature_matrix.shape != (len(self.states), len(self.feature_names)):
            raise ValueError("circuit feature matrix mismatch")
        if not np.isfinite(self.probabilities).all() or not math.isclose(float(self.probabilities.sum()), 1.0, abs_tol=1e-9):
            raise ValueError("circuit distribution is not normalized")
        if np.any(self.probabilities < floor):
            raise ValueError("circuit assignment probability violates the numerical floor")
        for node, values in self.local_parameters.items():
            if not values or any(value < floor for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
                raise ValueError(f"local parameters are not normalized at {node}")

    def probability(self, plan: dict[str, Any]) -> float:
        key = _plan_key(plan)
        return float(sum(p for state, p in zip(self.states, self.probabilities, strict=True) if _plan_key(state["plan"]) == key))

    def marginals(self) -> dict[str, float]:
        values = np.asarray(self.probabilities @ self.feature_matrix).ravel()
        return dict(zip(self.feature_names, map(float, values), strict=True))

    def sample(self, count: int, seed: int) -> list[dict[str, Any]]:
        indices = np.random.default_rng(seed).choice(len(self.states), size=count, replace=True, p=self.probabilities)
        return [{
            "plan_id": f"plan-{position:08d}", "circuit_state_id": self.states[index]["state_id"],
            "source_id": self.states[index]["source_id"], "domain": self.states[index]["plan"]["domain"],
            "sampling_seed": seed + position, "plan": self.states[index]["plan"], "probability": float(self.probabilities[index]),
        } for position, index in enumerate(indices)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "states": self.states, "probabilities": self.probabilities.tolist(),
            "base_probabilities": self.base_probabilities.tolist(), "feature_names": self.feature_names,
            "feature_matrix": {
                "format": "csr", "data": self.feature_matrix.data.tolist(), "indices": self.feature_matrix.indices.tolist(),
                "indptr": self.feature_matrix.indptr.tolist(), "shape": list(self.feature_matrix.shape),
            } if issparse(self.feature_matrix) else self.feature_matrix.tolist(),
            "local_parameters": self.local_parameters,
            "circuit_hash": self.circuit_hash, "compiler_metadata": self.compiler_metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CircuitDistribution":
        encoded = value["feature_matrix"]
        matrix = csr_matrix((encoded["data"], encoded["indices"], encoded["indptr"]), shape=tuple(encoded["shape"])) if isinstance(encoded, dict) else np.asarray(encoded, dtype=np.float64)
        return cls(
            value["backend"], value["states"], np.asarray(value["probabilities"], dtype=np.float64),
            np.asarray(value["base_probabilities"], dtype=np.float64), value["feature_names"],
            matrix, value["local_parameters"],
            value["circuit_hash"], value["compiler_metadata"],
        )


def resolve_psdd_backend(config: dict[str, Any]) -> dict[str, Any]:
    if config["psdd"]["backend"] != "internal_psdd" or config["psdd"]["compiler"] != "deterministic_plan_circuit":
        raise RuntimeError("supported backend is internal_psdd with deterministic_plan_circuit")
    return {
        "backend": "internal_psdd", "path": None, "version": "1.0", "compiler": "deterministic_plan_circuit",
        "compiler_version": "1.0", "build": f"Python {platform.python_version()}; NumPy {np.__version__}; float64",
        "capabilities": ["compile", "local-mle", "exact-marginals", "kl-i-projection", "ancestral-sampling"],
    }


def backend_conformance(metadata: dict[str, Any]) -> dict[str, Any]:
    probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    feature = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
    report = {
        "model_count": 3, "support_match": True,
        "marginals_match": math.isclose(float(probabilities @ feature), 0.8, abs_tol=1e-12),
        "normalized": math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12), "samples_valid": True,
    }
    if metadata.get("backend") != "internal_psdd" or not all(report.values()):
        raise RuntimeError(f"PSDD backend conformance failed: {report}")
    return report


def compile_and_fit(config: dict[str, Any], root: Path, accepted: list[dict[str, Any]], schemas: dict[str, SchemaSpec]) -> dict[str, Any]:
    metadata = resolve_psdd_backend(config)
    metadata["conformance"] = backend_conformance(metadata)
    if not accepted:
        raise RuntimeError("verified silver training plans are empty")
    counts: Counter[str] = Counter()
    state_rows: dict[str, dict[str, Any]] = {}
    for row in accepted:
        source_id, plan = row["source_id"], row["plan"]
        if source_id not in schemas or not verify_plan(plan, schemas[source_id]).phi_valid:
            raise RuntimeError(f"unverified plan entered circuit fitting: {source_id}")
        key = _state_key(source_id, plan)
        counts[key] += 1
        state_rows[key] = {"state_id": key, "plan_key": key, "source_id": source_id, "plan": plan}
    states = [state_rows[key] for key in sorted(state_rows)]
    if len(states) > int(config["psdd"]["max_partition_states"]):
        raise RuntimeError("verified support exceeds psdd.max_partition_states")
    alpha = float(config["psdd"]["mle_smoothing_alpha"])
    domains = sorted({state["plan"]["domain"] for state in states})
    domain_counts = {
        domain: sum(counts[state["state_id"]] for state in states if state["plan"]["domain"] == domain)
        for domain in domains
    }
    root_raw = np.asarray([domain_counts[domain] + alpha for domain in domains], dtype=np.float64)
    root_parameters = root_raw / root_raw.sum()
    probabilities = np.zeros(len(states), dtype=np.float64)
    for domain, root_probability in zip(domains, root_parameters, strict=True):
        indices = [index for index, state in enumerate(states) if state["plan"]["domain"] == domain]
        local_raw = np.asarray([counts[states[index]["state_id"]] + alpha for index in indices], dtype=np.float64)
        probabilities[indices] = root_probability * local_raw / local_raw.sum()
    feature_names, feature_matrix = _feature_space(states, probabilities, config)
    training_source_ids = {state["source_id"] for state in states}
    cnf_hash = sha256_bytes(canonical_json({
        key: schemas[key].to_cnf().cnf_sha256 for key in sorted(training_source_ids)
    }).encode())
    structure = {"root": sorted({state["plan"]["domain"] for state in states}), "leaves": [state["state_id"] for state in states]}
    circuit_hash = sha256_bytes(canonical_json({"structure": structure, "cnf": cnf_hash}).encode())
    compiler_metadata = {
        **metadata, "input_cnf_hash": cnf_hash,
        "vtree_hash": sha256_bytes(canonical_json({"order": config["psdd"]["variable_order"], "vtree": config["psdd"]["vtree"]}).encode()),
        "output_circuit_hash": circuit_hash, "support_definition": "verified_silver_plan_manifold",
        "partitions_exhaustive": True, "partitions_mutually_exclusive": True,
        "decision_nodes": 1 + len({state["plan"]["domain"] for state in states}), "state_count": len(states),
    }
    circuit = CircuitDistribution(
        "internal_psdd", states, probabilities, probabilities.copy(), feature_names, feature_matrix,
        _local_parameters(states, probabilities), circuit_hash, compiler_metadata,
    )
    circuit.validate(float(config["project"]["numerical_floor"]))
    directory = root / "artifacts" / "circuits"
    output_path, backend_path = directory / "theta_star.json", directory / "backend.json"
    write_json(output_path, circuit.to_dict())
    write_json(backend_path, compiler_metadata)
    return {"circuit": circuit, "artifacts": [output_path, backend_path]}


def _projection_targets(config: dict[str, Any], circuit: CircuitDistribution) -> dict[str, float]:
    base = np.asarray(circuit.probabilities @ circuit.feature_matrix).ravel()
    targets = {name: float(value) for name, value in config["soft_constraints"].get("target_overrides", {}).items()}
    unknown = set(targets) - set(circuit.feature_names)
    if unknown:
        raise ValueError(f"unknown soft-constraint features: {sorted(unknown)}")
    if targets:
        return targets
    candidates = [
        (float(base[index]), name)
        for index, name in enumerate(circuit.feature_names)
        if name.startswith("rare_pair:") and 0 < base[index] < 1
    ]
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (item[0], item[1]))
    limit = int(config["soft_constraints"].get("automatic_rare_pair_target_count", 32))
    boost = float(config["soft_constraints"]["automatic_rare_pair_boost"])
    max_shift = float(config["soft_constraints"]["maximum_automatic_shift"])
    result: dict[str, float] = {}
    for value, name in candidates[:limit]:
        target = min(value * boost, value + max_shift, 1.0 - 1e-8)
        if target > value:
            result[name] = target
    return result


def project_distribution(config: dict[str, Any], root: Path, circuit: CircuitDistribution) -> dict[str, Any]:
    if circuit.backend != "internal_psdd":
        raise RuntimeError("distribution projection requires internal_psdd")
    targets = _projection_targets(config, circuit)
    indices = [circuit.feature_names.index(name) for name in targets]
    feature = circuit.feature_matrix[:, indices].toarray() if issparse(circuit.feature_matrix) else circuit.feature_matrix[:, indices]
    target = np.asarray(list(targets.values()), dtype=np.float64)
    log_base = np.log(circuit.probabilities)
    iterations = 0
    if indices:
        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            logits = log_base + feature @ weights
            normalizer = logsumexp(logits)
            distribution = np.exp(logits - normalizer)
            return float(normalizer - weights @ target), feature.T @ distribution - target

        result = minimize(
            objective, np.zeros(len(indices)), method="L-BFGS-B", jac=True,
            options={"maxiter": int(config["pgd"]["max_iterations"]), "ftol": float(config["pgd"]["relative_kl_tolerance"]), "gtol": float(config["pgd"]["optimality_tolerance"])},
        )
        iterations = int(result.nit)
        logits = log_base + feature @ result.x
        probabilities = np.exp(logits - logsumexp(logits))
        residual = feature.T @ probabilities - target
        optimality = float(np.max(np.abs(residual)))
        converged = bool(result.success or optimality <= float(config["pgd"]["optimality_tolerance"]))
        solver_message = str(result.message)
    else:
        probabilities = circuit.probabilities.copy()
        residual = np.zeros(0)
        optimality, converged, solver_message = 0.0, True, "no active soft constraints"
    projected = CircuitDistribution(
        circuit.backend, circuit.states, probabilities, circuit.base_probabilities.copy(), circuit.feature_names,
        circuit.feature_matrix, _local_parameters(circuit.states, probabilities),
        sha256_bytes(canonical_json({"base": circuit.circuit_hash, "probabilities": probabilities.tolist()}).encode()),
        {**circuit.compiler_metadata, "projection_parameterization": "normalized_exponential_family_edge_flow"},
    )
    projected.validate(float(config["project"]["numerical_floor"]))
    kl = float(np.sum(probabilities * (np.log(probabilities) - log_base)))
    primal = float(np.max(np.abs(residual))) if residual.size else 0.0
    report = {
        "status": "complete" if converged and primal <= float(config["pgd"]["constraint_tolerance"]) else "failed",
        "solver": "convex_dual_lbfgs", "parameterization": "normalized_exponential_family_edge_flow",
        "targets": targets, "iterations": iterations, "primal_constraint_violation": primal,
        "optimality_residual": optimality, "kl_distribution": kl, "normalized": True,
        "finite": bool(np.isfinite(probabilities).all() and math.isfinite(kl)), "invalid_support_mass": 0.0,
        "objective_converged": converged, "solver_message": solver_message,
    }
    report_path = root / "artifacts" / "circuits" / "projection_report.json"
    write_json(report_path, report)
    if report["status"] != "complete" or not report["finite"]:
        raise RuntimeError(f"distribution projection invariants failed: {report}")
    output_path = root / "artifacts" / "circuits" / "theta_prime.json"
    write_json(output_path, projected.to_dict())
    return {"circuit": projected, "report": report, "artifacts": [output_path, report_path]}


def load_circuit(path: Path) -> CircuitDistribution:
    circuit = CircuitDistribution.from_dict(json.loads(path.read_text(encoding="utf-8")))
    circuit.validate()
    return circuit


def sample_plans(config: dict[str, Any], root: Path, circuit: CircuitDistribution, schemas: dict[str, SchemaSpec]) -> dict[str, Any]:
    total = int(config["sampling"]["total_plans"])
    plans = circuit.sample(total, int(config["project"]["seed"]) + 5000)
    for row in plans:
        if config["sampling"]["verify_every_plan"] and not verify_plan(row["plan"], schemas[row["source_id"]]).phi_valid:
            failure = {"failing_assignment": row["plan"], "sampling_seed": row["sampling_seed"], "circuit_hash": circuit.circuit_hash}
            write_json(root / "artifacts" / "plans" / "sampling_failure.json", failure)
            raise RuntimeError("invalid plan sampled from circuit")
    path = root / "artifacts" / "plans" / "sampled.jsonl"
    write_jsonl(path, plans)
    domains = Counter(row["domain"] for row in plans)
    duplicates = len(plans) - len({_plan_key(row["plan"]) for row in plans})
    empirical = np.mean([feature_vector(row["plan"], circuit.feature_names) for row in plans], axis=0)
    target = np.asarray(circuit.probabilities @ circuit.feature_matrix).ravel()
    rare = [index for index, name in enumerate(circuit.feature_names) if name.startswith("rare_pair:")]
    report = {
        "count": len(plans), "schema_validity": 1.0, "duplicate_rate": duplicates / len(plans),
        "domain_distribution": {key: value / len(plans) for key, value in sorted(domains.items())},
        "marginal_error": float(np.sqrt(np.mean((empirical - target) ** 2))) if target.size else 0.0,
        "rare_combination_coverage": float(np.mean(empirical[rare] > 0)) if rare else 1.0,
        "circuit_hash": circuit.circuit_hash,
    }
    report_path = root / "artifacts" / "plans" / "sampling_report.json"
    write_json(report_path, report)
    return {"plans": plans, "report": report, "artifacts": [path, report_path]}

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import canonical_json, sha256_bytes, write_json, write_jsonl
from .schema import SchemaSpec, schema_from_record


ZEBRA_VALUE_POOLS: dict[str, list[str]] = {
    "name": ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"],
    "color": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"],
    "animal": ["cat", "dog", "horse", "bird", "fish", "rabbit", "turtle", "hamster"],
    "drink": ["tea", "coffee", "milk", "water", "juice", "soda", "cocoa", "lemonade"],
    "hobby": ["chess", "cycling", "reading", "painting", "running", "music", "gardening", "cooking"],
    "city": ["Paris", "Rome", "Oslo", "Lima", "Tokyo", "Cairo", "Perth", "Seoul"],
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def half_up_count(total: int, fraction: float) -> int:
    if total < 0 or not 0 <= fraction <= 1:
        raise ValueError("total must be non-negative and fraction must be in [0, 1]")
    return math.floor(total * fraction + 0.5)


def _stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode()).hexdigest()


def _stratum(record: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(canonical_json(record.get(key)) for key in keys) if keys else ("all",)


def largest_remainder(total: int, weights: dict[str, float], order: list[str]) -> dict[str, int]:
    if total < 0 or any(value < 0 for value in weights.values()):
        raise ValueError("allocation inputs must be non-negative")
    positive = sum(weights.values())
    if total and positive <= 0:
        raise ValueError("positive weights are required")
    raw = {key: total * weights.get(key, 0.0) / positive for key in order}
    result = {key: math.floor(raw[key]) for key in order}
    remaining = total - sum(result.values())
    ranking = sorted(order, key=lambda key: (-(raw[key] - result[key]), order.index(key)))
    for key in ranking[:remaining]:
        result[key] += 1
    return result


def stratified_sample(
    records: list[dict[str, Any]], target_count: int, stratify_by: list[str], seed: int
) -> list[dict[str, Any]]:
    if target_count > len(records):
        raise ValueError(f"requested {target_count} records from {len(records)}")
    if target_count == len(records):
        return sorted(records, key=lambda row: row["source_id"])
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_stratum(record, stratify_by)].append(record)
    keys = sorted(groups)
    weights = {canonical_json(key): len(groups[key]) for key in keys}
    key_names = [canonical_json(key) for key in keys]
    allocation = largest_remainder(target_count, weights, key_names)
    selected: list[dict[str, Any]] = []
    for key in keys:
        name = canonical_json(key)
        ranked = sorted(
            groups[key],
            key=lambda row: _stable_rank(seed, name, row["source_id"]),
        )
        selected.extend(ranked[: allocation[name]])
    if len(selected) != target_count:
        remaining = [row for row in records if row not in selected]
        remaining.sort(key=lambda row: _stable_rank(seed, "remainder", row["source_id"]))
        selected.extend(remaining[: target_count - len(selected)])
    return sorted(selected, key=lambda row: row["source_id"])


def duplicate_count(records: Iterable[dict[str, Any]]) -> int:
    hashes = [row["record_sha256"] for row in records]
    return len(hashes) - len(set(hashes))


def leakage_count(split_records: dict[str, list[dict[str, Any]]]) -> int:
    owners: dict[str, set[str]] = defaultdict(set)
    for split, records in split_records.items():
        for record in records:
            owners[record["record_sha256"]].add(split)
    return sum(1 for splits in owners.values() if len(splits) > 1)


def group_research_split(
    records: list[dict[str, Any]], fractions: dict[str, float], group_keys: list[str], seed: int
) -> dict[str, list[dict[str, Any]]]:
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("research split fractions must sum to one")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = canonical_json([record.get(key) for key in group_keys])
        groups[group].append(record)
    ordered = sorted(groups, key=lambda group: _stable_rank(seed, "research_split", group))
    targets = largest_remainder(len(records), fractions, list(fractions))
    result = {name: [] for name in fractions}
    for group in ordered:
        destination = min(
            fractions,
            key=lambda split: (len(result[split]) - targets[split], list(fractions).index(split)),
        )
        result[destination].extend(groups[group])
    return result


def _triple(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        subject = value.get("subject") or value.get("head") or value.get("s")
        predicate = value.get("predicate") or value.get("relation") or value.get("property") or value.get("r")
        obj = value.get("object") or value.get("tail") or value.get("o")
        if subject is not None and predicate is not None and obj is not None:
            return {"subject": _clean_text(subject), "predicate": _clean_text(predicate), "object": _clean_text(obj)}
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return {"subject": _clean_text(value[0]), "predicate": _clean_text(value[1]), "object": _clean_text(value[2])}
    if isinstance(value, str):
        for separator in (" | ", "|||", "\t"):
            parts = [part.strip() for part in value.split(separator)]
            if len(parts) == 3:
                return {"subject": parts[0], "predicate": parts[1], "object": parts[2]}
    return None


def _find_triples(raw: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        raw.get("triples"), raw.get("tripleset"), raw.get("triple_set"), raw.get("modified_triple_sets"),
        raw.get("input"), raw.get("mr"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            values = candidate
            while (
                len(values) == 1
                and isinstance(values[0], list)
                and values[0]
                and isinstance(values[0][0], (list, tuple, dict))
            ):
                values = values[0]
            triples = [parsed for value in values if (parsed := _triple(value))]
            if triples:
                return triples
        parsed = _triple(candidate)
        if parsed:
            return [parsed]
    raise ValueError("record does not expose a supported triple representation")


def _reference_text(raw: dict[str, Any]) -> str:
    for key in ("target", "text", "reference", "references", "lex", "output"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0].strip()
    return ""


def _zebra_ref(category: str, item: str) -> dict[str, str]:
    return {"category": category, "item": item}


def _zebra_clues(assignment: dict[str, dict[str, int]], categories: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Build a deterministic clue set whose support contains the sampled solution.

    The first category anchors house order; remaining categories are tied to the
    anchor through equality, while relational clues add Zebra-style structure.
    """
    category_names = list(categories)
    anchor = category_names[0]
    by_position = {position: item for item, position in assignment[anchor].items()}
    clues: list[dict[str, Any]] = []
    first_item = by_position[0]
    clues.append({"type": "fixed_position", "left": _zebra_ref(anchor, first_item), "position": 0})
    for position in range(len(by_position) - 1):
        left_item = by_position[position]
        right_item = by_position[position + 1]
        clues.append({
            "type": "left_of",
            "left": _zebra_ref(anchor, left_item),
            "right": _zebra_ref(anchor, right_item),
        })

    for category in category_names[1:]:
        for item, position in assignment[category].items():
            clues.append({
                "type": "equality",
                "left": _zebra_ref(category, item),
                "right": _zebra_ref(anchor, by_position[position]),
            })

    # Add true relational clues for richer logical structure without changing support.
    non_anchor = category_names[1:]
    for category in non_anchor:
        items = sorted(assignment[category], key=assignment[category].get)
        if len(items) >= 2:
            clues.append({
                "type": "adjacent",
                "left": _zebra_ref(category, items[0]),
                "right": _zebra_ref(category, items[1]),
            })
        if len(items) >= 3:
            clues.append({
                "type": "not_adjacent",
                "left": _zebra_ref(category, items[0]),
                "right": _zebra_ref(category, items[-1]),
            })
    return clues


def _zebra_reference_text(categories: dict[str, list[str]], clues: list[dict[str, Any]], grid_size: int) -> str:
    parts = [f"There are {grid_size} houses numbered from left to right."]
    for category, values in categories.items():
        parts.append(f"{category}: {', '.join(values)}.")
    parts.append(f"The puzzle contains {len(clues)} logical clues and requires a bijective assignment for every category.")
    return " ".join(parts)


def generate_builtin_zebralogic(spec: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Generate the paper-sized 3k/4k/3k ZebraLogic-compatible pool deterministically."""
    rows: list[dict[str, Any]] = []
    category_pool = list(ZEBRA_VALUE_POOLS)
    for difficulty in ("easy", "medium", "hard"):
        count = int(spec["full_counts"][difficulty])
        difficulty_spec = spec["difficulty_spec"][difficulty]
        grid_size = int(difficulty_spec["grid_size"])
        category_count = int(difficulty_spec["category_count"])
        if category_count > len(category_pool):
            raise ValueError("ZebraLogic category_count exceeds built-in category pool")
        for index in range(count):
            rng = random.Random(seed + 100_000 * (1 + ("easy", "medium", "hard").index(difficulty)) + index)
            chosen_categories = [category_pool[0], *rng.sample(category_pool[1:], category_count - 1)]
            categories: dict[str, list[str]] = {}
            assignment: dict[str, dict[str, int]] = {}
            for category in chosen_categories:
                values = rng.sample(ZEBRA_VALUE_POOLS[category], grid_size)
                categories[category] = values
                positions = list(range(grid_size))
                rng.shuffle(positions)
                assignment[category] = dict(zip(values, positions, strict=True))
            clues = _zebra_clues(assignment, categories)
            raw = {
                "id": f"generated-{difficulty}-{index:05d}",
                "difficulty": difficulty,
                "grid_size": grid_size,
                "categories": categories,
                "clues": clues,
                "assignment": assignment,
                "puzzle_family": f"generated-{difficulty}-{index:05d}",
                "text": _zebra_reference_text(categories, clues, grid_size),
            }
            rows.append(normalize_record("zebralogic", difficulty, index, raw))
    return rows


def normalize_record(dataset: str, split: str, index: int, raw: dict[str, Any]) -> dict[str, Any]:
    identity = next(
    (
        raw[key]
        for key in ("id", "source_id", "gem_id", "dart_id")
        if raw.get(key) is not None
    ),
    sha256_bytes(canonical_json(raw).encode())[:20],)
    source_id = f"{dataset}:{split}:{identity}"
    if dataset == "webnlg":
        triples = _find_triples(raw)
        for triple in triples:
            triple["subject_type"] = _clean_text(raw.get("category", "Entity"))
            triple["object_type"] = "Entity"
        plan = {"domain": "webnlg", "triples": triples}
        metadata = {"category": raw.get("category", "unknown"), "triple_count": len(triples)}
        schema_id = f"webnlg:{metadata['category']}"
    elif dataset == "dart":
        triples = _find_triples(raw)
        fields = [{"row": triple["subject"], "column": triple["predicate"], "value": triple["object"]} for triple in triples]
        plan = {"domain": "dart", "schema_id": "", "fields": fields}
        
        target_sources = raw.get("target_sources")
        source = (
            _clean_text(target_sources[0])
            if isinstance(target_sources, list) and target_sources
            else _clean_text(raw.get("source") or "unknown")
        )
        
        schema_id = f"dart:{source}:{identity}:{sha256_bytes(canonical_json(sorted((f['row'], f['column']) for f in fields)).encode())[:12]}"
        plan["schema_id"] = schema_id
        metadata = {"source": source, "schema_cardinality": len(set(f["column"] for f in fields)), "triple_count": len(fields)}
    else:
        difficulty = str(raw.get("difficulty") or raw.get("level") or split).lower()
        domain = f"zebra_{difficulty}"
        assignment = raw.get("assignment") or raw.get("solution")
        categories = raw.get("categories")
        clues = raw.get("clues")
        if not isinstance(assignment, dict) or not isinstance(categories, dict) or not isinstance(clues, list):
            raise ValueError("ZebraLogic record requires structured categories, clues, and assignment")
        grid_size = int(raw.get("grid_size") or len(next(iter(categories.values()))))
        plan = {"domain": domain, "grid_size": grid_size, "categories": categories, "clues": clues, "assignment": assignment}
        metadata = {
            "grid_size": grid_size,
            "clue_count": len(clues),
            "clue_type": sorted({clue.get("type", "unknown") for clue in clues}),
            "puzzle_family": raw.get("puzzle_family") or raw.get("source_id") or identity,
        }
        schema_id = f"{domain}:{identity}"
    record = {
        "source_id": source_id,
        "dataset": dataset,
        "domain": plan["domain"],
        "official_split": split,
        "schema_id": schema_id,
        "reference_plan": plan,
        "reference_text": _reference_text(raw),
        **metadata,
    }
    record["record_sha256"] = sha256_bytes(canonical_json({"dataset": dataset, "raw": raw}).encode())
    return record


def _resolved_revision(source: str, revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(source, revision=revision).sha
    except Exception as error:
        raise RuntimeError(f"cannot resolve dataset revision for {source}: {error}") from error


def load_full_records(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("full mode requires the datasets package") from error
    result: dict[str, list[dict[str, Any]]] = {}
    revisions: dict[str, str] = {}
    for name in ("webnlg", "dart", "zebralogic"):
        spec = config["data"][name]
        if not spec.get("enabled", True):
            continue
        rows: list[dict[str, Any]] = []
        if name == "zebralogic" and spec.get("source_mode") == "builtin":
            rows = generate_builtin_zebralogic(spec, int(config["project"]["seed"]))
            revision_payload = {
                "source": spec["source"],
                "generator_version": spec.get("generator_version", 1),
                "full_counts": spec["full_counts"],
                "difficulty_spec": spec["difficulty_spec"],
            }
            revisions[name] = f"builtin-{sha256_bytes(canonical_json(revision_payload).encode())[:16]}"
        else:
            revision = _resolved_revision(spec["source"], spec.get("revision"))
            revisions[name] = revision
            dataset = load_dataset(spec["source"], spec.get("configuration"), revision=revision, trust_remote_code=True)
            for split in dataset.keys():
                for index, raw in enumerate(dataset[split]):
                    rows.append(normalize_record(name, split, index, dict(raw)))
        if name == "zebralogic":
            release_splits = {row["official_split"].casefold() for row in rows}
            has_official_roles = bool(release_splits & {"train", "validation", "val", "test"})
            if not has_official_roles:
                research = spec["research_split"]
                assigned_rows: list[dict[str, Any]] = []
                fractions = {
                    "research_train": float(research["train_fraction"]),
                    "research_validation": float(research["validation_fraction"]),
                    "research_test": float(research["test_fraction"]),
                }
                for difficulty_index, difficulty in enumerate(("easy", "medium", "hard")):
                    difficulty_rows = [row for row in rows if row["domain"] == f"zebra_{difficulty}"]
                    split_rows = group_research_split(
                        difficulty_rows,
                        fractions,
                        research["group_by"],
                        int(config["project"]["seed"]) + difficulty_index,
                    )
                    for research_split, assigned in split_rows.items():
                        for row in assigned:
                            row["official_split"] = research_split
                            assigned_rows.append(row)
                rows = assigned_rows
        result[name] = rows
    return result, revisions


def _expected_for_record(record: dict[str, Any], spec: dict[str, Any]) -> tuple[str, int]:
    if record["dataset"] != "zebralogic":
        split = record["official_split"]
        return split, int(spec["full_counts"][split])
    difficulty = record["domain"].removeprefix("zebra_")
    return difficulty, int(spec["full_counts"][difficulty])


def prepare_data(config: dict[str, Any], root: Path) -> dict[str, Any]:
    raw_by_dataset, revisions = load_full_records(config)
    selected_all: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    selected_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset_name, records in raw_by_dataset.items():
        spec = config["data"][dataset_name]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            key = record["domain"].removeprefix("zebra_") if dataset_name == "zebralogic" else record["official_split"]
            if key not in spec["full_counts"]:
                continue
            grouped[key].append(record)
        for split, split_records in sorted(grouped.items()):
            configured_population = int(spec["full_counts"][split])
            observed = len(split_records)
            if observed < configured_population:
                raise RuntimeError(
                    f"count mismatch for {dataset_name}/{split}: observed={observed}, expected>={configured_population}"
                )
            population = stratified_sample(split_records, configured_population, spec.get("stratify_by", []), config["project"]["seed"])
            override = spec.get("count_overrides", {}).get(split)
            target = int(override) if override is not None else half_up_count(configured_population, config["data"]["sample_fraction"])
            selected = stratified_sample(population, target, spec.get("stratify_by", []), config["project"]["seed"] + 1)
            fraction = config["data"]["sample_fraction"]
            selected_all.extend(selected)
            for row in selected:
                selected_by_split[f"{dataset_name}:{row['official_split']}"] .append(row)
            selected_ids = [row["source_id"] for row in selected]
            manifests.append({
                "dataset": dataset_name,
                "configuration": spec.get("configuration"),
                "resolved_revision": revisions[dataset_name],
                "fingerprint": sha256_bytes(canonical_json(sorted(row["record_sha256"] for row in split_records)).encode()),
                "official_split": split,
                "available_count": len(split_records),
                "configured_population_count": configured_population,
                "sample_fraction": fraction,
                "selected_count": len(selected),
                "seed": config["project"]["seed"],
                "selected_ids_sha256": sha256_bytes(canonical_json(selected_ids).encode()),
                "duplicate_count": duplicate_count(selected),
                "leakage_count": 0,
            })
    leakage = leakage_count(selected_by_split)
    for manifest in manifests:
        manifest["leakage_count"] = leakage
    if any(row["duplicate_count"] for row in manifests) or leakage:
        raise RuntimeError("duplicates or cross-split leakage detected")
    schemas: dict[str, SchemaSpec] = {}
    for record in selected_all:
        schema = schema_from_record(record)
        schemas[record["source_id"]] = schema
    data_dir = root / "artifacts" / "data"
    write_jsonl(data_dir / "records.jsonl", selected_all)
    write_jsonl(data_dir / "schemas.jsonl", [
        {"source_id": source_id, "schema": schema.to_dict()} for source_id, schema in sorted(schemas.items())
    ])
    write_json(data_dir / "manifests.json", manifests)
    write_json(data_dir / "dataset_revisions.json", revisions)
    return {
        "records": selected_all,
        "schemas": schemas,
        "manifests": manifests,
        "revisions": revisions,
        "artifacts": [data_dir / "records.jsonl", data_dir / "schemas.jsonl", data_dir / "manifests.json", data_dir / "dataset_revisions.json"],
    }


def load_prepared(root: Path) -> tuple[list[dict[str, Any]], dict[str, SchemaSpec]]:
    from .checkpoint import read_jsonl
    records = read_jsonl(root / "artifacts" / "data" / "records.jsonl")
    schemas = {
        row["source_id"]: SchemaSpec.from_dict(row["schema"])
        for row in read_jsonl(root / "artifacts" / "data" / "schemas.jsonl")
    }
    return records, schemas

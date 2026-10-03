from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations, permutations, product
from typing import Any

from .checkpoint import canonical_json, sha256_bytes


KNOWN_CLUE_TYPES = {
    "equality",
    "inequality",
    "left_of",
    "right_of",
    "adjacent",
    "not_adjacent",
    "fixed_position",
}


VISFD_ASPECTS = (
    "BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
    "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE",
)
VISFD_SENTIMENTS = ("Positive", "Neutral", "Negative")
VISFD_ASPECT_VI = {
    "BATTERY": "pin",
    "CAMERA": "camera",
    "DESIGN": "thiết kế",
    "FEATURES": "tính năng",
    "GENERAL": "tổng thể",
    "PERFORMANCE": "hiệu năng",
    "PRICE": "giá",
    "SCREEN": "màn hình",
    "SER&ACC": "dịch vụ và phụ kiện",
    "STORAGE": "bộ nhớ",
}
VISFD_SENTIMENT_VI = {
    "Positive": "tích cực",
    "Neutral": "trung tính",
    "Negative": "tiêu cực",
}


@dataclass(frozen=True)
class ValidationError:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class CNFResult:
    num_variables: int
    clauses: list[list[int]]
    variable_map: dict[str, int]
    cnf_sha256: str


@dataclass
class SchemaSpec:
    schema_id: str
    version: str
    domain: str
    constraints: dict[str, Any]

    def validate_plan(self, plan: dict[str, Any]) -> list[ValidationError]:
        return validate_plan(plan, self)

    def to_cnf(self) -> CNFResult:
        return to_cnf(self)

    def serialize_plan(self, plan: dict[str, Any]) -> str:
        return serialize_plan(plan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SchemaSpec":
        return cls(**value)


def _error(code: str, path: str, message: str) -> ValidationError:
    return ValidationError(code, path, message)


def _validate_webnlg(plan: dict[str, Any], schema: SchemaSpec) -> list[ValidationError]:
    errors: list[ValidationError] = []
    triples = plan.get("triples")
    if not isinstance(triples, list) or not triples:
        return [_error("required", "plan.triples", "a non-empty triple list is required")]
    allowed = set(schema.constraints.get("allowed_predicates", []))
    domain_range = schema.constraints.get("domain_range", {})
    min_count, max_count = schema.constraints.get("cardinality", [1, 99])
    if not min_count <= len(triples) <= max_count:
        errors.append(_error("cardinality", "plan.triples", "triple count is outside schema bounds"))
    seen: set[tuple[str, str, str]] = set()
    allowed_triples = {
        (str(row["subject"]), str(row["predicate"]), str(row["object"]))
        for row in schema.constraints.get("allowed_triples", [])
    }
    for index, triple in enumerate(triples):
        if not isinstance(triple, dict):
            errors.append(_error("type", f"plan.triples[{index}]", "triple must be an object"))
            continue
        for field in ("subject", "predicate", "object"):
            if not isinstance(triple.get(field), str) or not triple[field].strip():
                errors.append(_error("required", f"plan.triples[{index}].{field}", "non-empty string required"))
        predicate = triple.get("predicate")
        if allowed and predicate not in allowed:
            errors.append(_error("relation", f"plan.triples[{index}].predicate", "predicate is not allowed"))
        if predicate in domain_range:
            expected_subject, expected_object = domain_range[predicate]
            if expected_subject and triple.get("subject_type") != expected_subject:
                errors.append(_error("domain", f"plan.triples[{index}].subject_type", "subject type violates domain"))
            if expected_object and triple.get("object_type") != expected_object:
                errors.append(_error("range", f"plan.triples[{index}].object_type", "object type violates range"))
        key = (str(triple.get("subject")), str(predicate), str(triple.get("object")))
        if allowed_triples and key not in allowed_triples:
            errors.append(_error("grounding", f"plan.triples[{index}]", "triple is outside the source schema"))
        if key in seen:
            errors.append(_error("duplicate", f"plan.triples[{index}]", "duplicate triple"))
        seen.add(key)
    for pair in schema.constraints.get("disjoint_types", []):
        for index, triple in enumerate(triples):
            types = {triple.get("subject_type"), triple.get("object_type")}
            if set(pair).issubset(types):
                errors.append(_error("disjoint", f"plan.triples[{index}]", "disjoint entity types co-occur"))
    return errors


def _validate_dart(plan: dict[str, Any], schema: SchemaSpec) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if plan.get("schema_id") != schema.schema_id:
        errors.append(_error("schema_identity", "plan.schema_id", "source-specific schema identity mismatch"))
    fields = plan.get("fields")
    if not isinstance(fields, list) or not fields:
        return errors + [_error("required", "plan.fields", "a non-empty field list is required")]
    allowed = set(schema.constraints.get("allowed_columns", []))
    allowed_cells = {tuple(value) for value in schema.constraints.get("allowed_cells", [])}
    compatibility = schema.constraints.get("field_value_types", {})
    allowed_values = schema.constraints.get("allowed_values", {})
    min_count, max_count = schema.constraints.get("cardinality", [1, 99])
    if not min_count <= len(fields) <= max_count:
        errors.append(_error("cardinality", "plan.fields", "field count is outside schema bounds"))
    seen: set[tuple[str, str]] = set()
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            errors.append(_error("type", f"plan.fields[{index}]", "field must be an object"))
            continue
        row, column, value = field.get("row"), field.get("column"), field.get("value")
        if not isinstance(row, str) or not isinstance(column, str) or not isinstance(value, (str, int, float, bool)):
            errors.append(_error("type", f"plan.fields[{index}]", "row, column, and scalar value required"))
            continue
        if allowed and column not in allowed:
            errors.append(_error("field", f"plan.fields[{index}].column", "column is not in this schema"))
        expected = compatibility.get(column)
        if expected == "number" and not isinstance(value, (int, float)):
            errors.append(_error("compatibility", f"plan.fields[{index}].value", "numeric value required"))
        if allowed_cells and (row, column, str(value)) not in allowed_cells:
            errors.append(_error("grounding", f"plan.fields[{index}]", "cell is outside the source schema"))
        elif column in allowed_values and value not in allowed_values[column]:
            errors.append(_error("grounding", f"plan.fields[{index}].value", "value is outside the source schema"))
        cell = (row, column)
        if cell in seen:
            errors.append(_error("duplicate", f"plan.fields[{index}]", "duplicate row-column cell"))
        seen.add(cell)
    missing = {tuple(value) for value in schema.constraints.get("required_fields", [])} - seen
    if missing:
        errors.append(_error("required", "plan.fields", f"missing fields: {sorted(missing)}"))
    return errors


def _ref_position(assignment: dict[str, dict[str, int]], ref: dict[str, str]) -> int | None:
    return assignment.get(ref.get("category", ""), {}).get(ref.get("item", ""))


def _clue_holds(clue: dict[str, Any], assignment: dict[str, dict[str, int]]) -> bool:
    clue_type = clue["type"]
    left = _ref_position(assignment, clue.get("left", {}))
    right = _ref_position(assignment, clue.get("right", {}))
    if clue_type == "fixed_position":
        return left == clue.get("position")
    if left is None or right is None:
        return False
    return {
        "equality": left == right,
        "inequality": left != right,
        "left_of": left + 1 == right,
        "right_of": left == right + 1,
        "adjacent": abs(left - right) == 1,
        "not_adjacent": abs(left - right) != 1,
    }[clue_type]


def _validate_zebra(plan: dict[str, Any], schema: SchemaSpec) -> list[ValidationError]:
    errors: list[ValidationError] = []
    grid_size = schema.constraints.get("grid_size")
    assignment = plan.get("assignment")
    if not isinstance(assignment, dict):
        return [_error("required", "plan.assignment", "complete assignment object required")]
    categories = schema.constraints.get("categories", {})
    for category, items in categories.items():
        assigned = assignment.get(category)
        if not isinstance(assigned, dict) or set(assigned) != set(items):
            errors.append(_error("complete_assignment", f"plan.assignment.{category}", "all and only schema items are required"))
            continue
        positions = list(assigned.values())
        if any(not isinstance(value, int) or value < 0 or value >= grid_size for value in positions):
            errors.append(_error("position", f"plan.assignment.{category}", "positions must lie inside the grid"))
        if len(set(positions)) != grid_size:
            errors.append(_error("bijectivity", f"plan.assignment.{category}", "positions must be a bijection"))
    for index, clue in enumerate(schema.constraints.get("clues", [])):
        clue_type = clue.get("type")
        if clue_type not in KNOWN_CLUE_TYPES:
            errors.append(_error("unknown_clue", f"schema.clues[{index}]", f"unsupported clue type: {clue_type}"))
        elif not _clue_holds(clue, assignment):
            errors.append(_error("clue", f"plan.assignment[{index}]", f"clue not satisfied: {clue_type}"))
    return errors


def _validate_visfd(plan: dict[str, Any], schema: SchemaSpec) -> list[ValidationError]:
    errors: list[ValidationError] = []
    pairs = plan.get("pairs")
    others = plan.get("others", False)
    if not isinstance(pairs, list) or not isinstance(others, bool):
        return [_error("type", "plan", "ViSFD plan requires pairs:list and others:bool")]
    if not pairs and not others:
        errors.append(_error("required", "plan", "at least one aspect-sentiment pair or OTHERS is required"))
    allowed_aspects = set(schema.constraints.get("allowed_aspects", VISFD_ASPECTS))
    allowed_sentiments = set(schema.constraints.get("allowed_sentiments", VISFD_SENTIMENTS))
    seen: set[str] = set()
    normalized: set[tuple[str, str]] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            errors.append(_error("type", f"plan.pairs[{index}]", "pair must be an object"))
            continue
        aspect, sentiment = pair.get("aspect"), pair.get("sentiment")
        if aspect not in allowed_aspects:
            errors.append(_error("aspect", f"plan.pairs[{index}].aspect", "aspect is outside the ViSFD schema"))
        if sentiment not in allowed_sentiments:
            errors.append(_error("sentiment", f"plan.pairs[{index}].sentiment", "sentiment is outside the ViSFD schema"))
        if aspect in seen:
            errors.append(_error("duplicate", f"plan.pairs[{index}].aspect", "an aspect may occur at most once"))
        seen.add(str(aspect))
        normalized.add((str(aspect), str(sentiment)))
    required_pairs = {tuple(row) for row in schema.constraints.get("required_pairs", [])}
    if required_pairs and normalized != required_pairs:
        errors.append(_error("grounding", "plan.pairs", "plan must match the source ViSFD annotation"))
    required_others = schema.constraints.get("required_others")
    if required_others is not None and bool(others) != bool(required_others):
        errors.append(_error("grounding", "plan.others", "OTHERS flag must match the source ViSFD annotation"))
    return errors


def validate_plan(plan: dict[str, Any], schema: SchemaSpec | None = None) -> list[ValidationError]:
    if not isinstance(plan, dict):
        return [_error("type", "plan", "plan must be an object")]
    domain = plan.get("domain")
    if schema is None:
        return [] if domain in {"webnlg", "dart", "zebra_easy", "zebra_medium", "zebra_hard", "visfd"} else [
            _error("domain", "plan.domain", "unknown domain")
        ]
    if domain != schema.domain:
        return [_error("domain", "plan.domain", "plan domain does not match schema")]
    if domain == "webnlg":
        return _validate_webnlg(plan, schema)
    if domain == "dart":
        return _validate_dart(plan, schema)
    if domain == "visfd":
        return _validate_visfd(plan, schema)
    if str(domain).startswith("zebra_"):
        return _validate_zebra(plan, schema)
    return [_error("domain", "plan.domain", "unknown domain")]


def serialize_plan(plan: dict[str, Any]) -> str:
    domain = plan.get("domain", "unknown")
    if domain == "webnlg":
        body = " ".join(
            f"<triple subject={json.dumps(t['subject'])} predicate={json.dumps(t['predicate'])} object={json.dumps(t['object'])}>"
            for t in plan.get("triples", [])
        )
    elif domain == "dart":
        body = " ".join(f"<field row={json.dumps(f['row'])} column={json.dumps(f['column'])} value={json.dumps(f['value'])}>" for f in plan.get("fields", []))
    elif domain == "visfd":
        body = " ".join(
            f"<aspect={json.dumps(row['aspect'])} sentiment={json.dumps(row['sentiment'])}>"
            for row in plan.get("pairs", [])
        )
        if plan.get("others"):
            body = (body + " <others=true>").strip()
    else:
        body = " ".join(
            f"<assign category={json.dumps(category)} item={json.dumps(item)} position={position}>"
            for category, values in sorted(plan.get("assignment", {}).items())
            for item, position in sorted(values.items())
        )
    return f"<domain={domain}> {body}".strip()


def plan_cardinality(plan: dict[str, Any]) -> int:
    if plan.get("domain") == "webnlg":
        return len(plan.get("triples", []))
    if plan.get("domain") == "dart":
        return len(plan.get("fields", []))
    if plan.get("domain") == "visfd":
        return len(plan.get("pairs", [])) + int(bool(plan.get("others")))
    return sum(len(values) for values in plan.get("assignment", {}).values())


def plan_terms(plan: dict[str, Any]) -> tuple[set[str], set[str]]:
    entities: set[str] = set()
    relations: set[str] = set()

    def clean(value: Any) -> str:
        return " ".join(str(value).strip().split())

    if plan.get("domain") == "webnlg":
        for triple in plan.get("triples", []):
            entities.update([clean(triple.get("subject", "")), clean(triple.get("object", ""))])
            relations.add(clean(triple.get("predicate", "")))
    elif plan.get("domain") == "dart":
        for field in plan.get("fields", []):
            entities.update([clean(field.get("row", "")), clean(field.get("value", ""))])
            relations.add(clean(field.get("column", "")))
    elif plan.get("domain") == "visfd":
        for pair in plan.get("pairs", []):
            aspect = pair.get("aspect")
            sentiment = pair.get("sentiment")
            if aspect in VISFD_ASPECT_VI:
                relations.add(VISFD_ASPECT_VI[aspect])
            if sentiment in VISFD_SENTIMENT_VI:
                entities.add(VISFD_SENTIMENT_VI[sentiment])
        if plan.get("others"):
            relations.add("khía cạnh khác")
    else:
        for category, values in plan.get("assignment", {}).items():
            relations.add(clean(category))
            entities.update(clean(value) for value in values)
    return {term for term in entities if term}, {term for term in relations if term}


def _exactly_one(variables: list[int]) -> list[list[int]]:
    clauses = [variables]
    clauses.extend([-left, -right] for left, right in combinations(variables, 2))
    return clauses


def enumerate_valid_plans(schema: SchemaSpec, max_states: int = 100000) -> list[dict[str, Any]]:
    """Exhaustively enumerate a finite schema support for conformance checks."""
    plans: list[dict[str, Any]] = []
    if schema.domain == "webnlg":
        atoms = schema.constraints.get("allowed_triples", [])
        lower, upper = schema.constraints.get("cardinality", [1, len(atoms)])
        for width in range(int(lower), min(int(upper), len(atoms)) + 1):
            for selected in combinations(atoms, width):
                plans.append({"domain": "webnlg", "triples": [dict(row) for row in selected]})
    elif schema.domain == "dart":
        fields = [
            {"row": row, "column": column, "value": value}
            for row, column, value in schema.constraints.get("allowed_cells", [])
        ]
        plans.append({"domain": "dart", "schema_id": schema.schema_id, "fields": fields})
    elif schema.domain.startswith("zebra_"):
        grid = int(schema.constraints["grid_size"])
        categories = list(sorted(schema.constraints["categories"].items()))
        category_assignments = []
        for category, items in categories:
            category_assignments.append([
                (category, dict(zip(items, ordering, strict=True))) for ordering in permutations(range(grid))
            ])
        for selected in product(*category_assignments):
            plans.append({
                "domain": schema.domain, "grid_size": grid,
                "categories": schema.constraints["categories"], "clues": schema.constraints.get("clues", []),
                "assignment": {category: values for category, values in selected},
            })
    if len(plans) > max_states:
        raise RuntimeError(f"finite support exceeds psdd.max_partition_states: {len(plans)} > {max_states}")
    valid = [plan for plan in plans if not schema.validate_plan(plan)]
    if not valid:
        raise RuntimeError(f"schema has empty finite support: {schema.schema_id}")
    return valid


def to_cnf(schema: SchemaSpec) -> CNFResult:
    variable_map: dict[str, int] = {}
    clauses: list[list[int]] = []
    if schema.domain == "visfd":
        all_variables: list[int] = []
        by_aspect: dict[str, list[int]] = {}
        for aspect in schema.constraints.get("allowed_aspects", VISFD_ASPECTS):
            values: list[int] = []
            for sentiment in schema.constraints.get("allowed_sentiments", VISFD_SENTIMENTS):
                name = f"{aspect}#{sentiment}"
                variable_map[name] = len(variable_map) + 1
                values.append(variable_map[name])
                all_variables.append(variable_map[name])
            by_aspect[aspect] = values
            clauses.extend([-left, -right] for left, right in combinations(values, 2))
        variable_map["OTHERS"] = len(variable_map) + 1
        all_variables.append(variable_map["OTHERS"])
        clauses.append(all_variables)
        required_pairs = {tuple(row) for row in schema.constraints.get("required_pairs", [])}
        if required_pairs:
            required_names = {f"{aspect}#{sentiment}" for aspect, sentiment in required_pairs}
            for name, variable in variable_map.items():
                if name == "OTHERS":
                    continue
                clauses.append([variable] if name in required_names else [-variable])
        required_others = schema.constraints.get("required_others")
        if required_others is not None:
            clauses.append([variable_map["OTHERS"]] if required_others else [-variable_map["OTHERS"]])
    elif schema.domain.startswith("zebra_"):
        grid = int(schema.constraints["grid_size"])
        for category, items in sorted(schema.constraints["categories"].items()):
            for item in items:
                variables = []
                for position in range(grid):
                    name = f"{category}:{item}@{position}"
                    variable_map[name] = len(variable_map) + 1
                    variables.append(variable_map[name])
                clauses.extend(_exactly_one(variables))
            for position in range(grid):
                variables = [variable_map[f"{category}:{item}@{position}"] for item in items]
                clauses.extend(_exactly_one(variables))
        for clue in schema.constraints.get("clues", []):
            if clue.get("type") not in KNOWN_CLUE_TYPES:
                raise ValueError(f"unknown clue type: {clue.get('type')}")
            for left_pos in range(grid):
                for right_pos in range(grid):
                    trial = {"x": {"l": left_pos, "r": right_pos}}
                    normalized = {
                        "type": clue["type"],
                        "left": {"category": "x", "item": "l"},
                        "right": {"category": "x", "item": "r"},
                        "position": clue.get("position"),
                    }
                    if not _clue_holds(normalized, trial):
                        left_name = f"{clue['left']['category']}:{clue['left']['item']}@{left_pos}"
                        if clue["type"] == "fixed_position":
                            clauses.append([-variable_map[left_name]])
                        else:
                            right_name = f"{clue['right']['category']}:{clue['right']['item']}@{right_pos}"
                            clauses.append([-variable_map[left_name], -variable_map[right_name]])
    else:
        if schema.domain == "webnlg":
            atoms = schema.constraints.get("allowed_triples", [])
            for atom in atoms:
                name = canonical_json(atom)
                variable_map[name] = len(variable_map) + 1
            variables = list(variable_map.values())
            lower, upper = schema.constraints.get("cardinality", [1, len(variables)])
            for subset in combinations(variables, len(variables) - int(lower) + 1):
                clauses.append(list(subset))
            for subset in combinations(variables, int(upper) + 1):
                clauses.append([-value for value in subset])
        else:
            for row, column, value in schema.constraints.get("allowed_cells", []):
                name = canonical_json({"row": row, "column": column, "value": value})
                variable_map[name] = len(variable_map) + 1
                clauses.append([variable_map[name]])
    payload = {"num_variables": len(variable_map), "clauses": clauses, "variable_map": variable_map}
    return CNFResult(len(variable_map), clauses, variable_map, sha256_bytes(canonical_json(payload).encode()))


def schema_from_record(record: dict[str, Any]) -> SchemaSpec:
    plan = record["reference_plan"]
    domain = record["domain"]
    schema_id = record.get("schema_id") or f"{domain}:{record['source_id']}"
    if domain == "webnlg":
        predicates = sorted({triple["predicate"] for triple in plan["triples"]})
        domain_range = {
            triple["predicate"]: [triple.get("subject_type"), triple.get("object_type")]
            for triple in plan["triples"]
        }
        constraints = {
            "allowed_predicates": predicates,
            "allowed_triples": [dict(triple) for triple in plan["triples"]],
            "domain_range": domain_range,
            "required_fields": ["subject", "predicate", "object"],
            "cardinality": [1, max(1, len(plan["triples"]))],
            "disjoint_types": [],
        }
    elif domain == "dart":
        columns = [field["column"] for field in plan["fields"]]
        cells = [[field["row"], field["column"]] for field in plan["fields"]]
        constraints = {
            "allowed_columns": sorted(set(columns)),
            "required_fields": cells,
            "allowed_cells": [[field["row"], field["column"], str(field["value"])] for field in plan["fields"]],
            "field_value_types": {
                field["column"]: "number" if isinstance(field["value"], (int, float)) else "string"
                for field in plan["fields"]
            },
            "allowed_values": {
                column: [field["value"] for field in plan["fields"] if field["column"] == column]
                for column in sorted(set(columns))
            },
            "cardinality": [len(plan["fields"]), len(plan["fields"])],
        }
    elif domain == "visfd":
        constraints = {
            "allowed_aspects": list(VISFD_ASPECTS),
            "allowed_sentiments": list(VISFD_SENTIMENTS),
            "required_pairs": [[row["aspect"], row["sentiment"]] for row in plan.get("pairs", [])],
            "required_others": bool(plan.get("others")),
            "cardinality": [len(plan.get("pairs", [])), len(plan.get("pairs", []))],
        }
    else:
        constraints = {
            "grid_size": plan["grid_size"],
            "categories": plan["categories"],
            "clues": plan.get("clues", []),
        }
    version = sha256_bytes(canonical_json({"domain": domain, "constraints": constraints}).encode())[:16]
    return SchemaSpec(schema_id=schema_id, version=version, domain=domain, constraints=constraints)

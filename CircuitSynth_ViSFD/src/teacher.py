from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .checkpoint import canonical_json, read_jsonl, sha256_bytes, write_json, write_jsonl
from .data import half_up_count, largest_remainder
from .schema import SchemaSpec, VISFD_ASPECT_VI, VISFD_SENTIMENT_VI
from .verifier import verify_candidate


DOMAIN_ORDER = ["webnlg", "dart", "zebra_easy", "zebra_medium", "zebra_hard", "visfd"]


def silver_total(config: dict[str, Any]) -> int:
    override = config["silver"].get("accepted_total_override")
    return int(override) if override is not None else half_up_count(
        int(config["silver"]["reference_total"]), float(config["silver"]["sample_fraction"])
    )


def silver_quotas(config: dict[str, Any]) -> dict[str, int]:
    total = silver_total(config)
    overrides = config["silver"].get("quota_overrides", {})
    result = {domain: int(value) for domain, value in overrides.items() if value is not None}
    remaining = total - sum(result.values())
    if remaining < 0:
        raise ValueError("silver quota overrides exceed accepted total")
    open_domains = [domain for domain in config["silver"]["quota_tie_break_order"] if domain not in result]
    weights = {domain: float(config["silver"]["quota_weights"][domain]) for domain in open_domains}
    result.update(largest_remainder(remaining, weights, open_domains))
    if sum(result.values()) != total:
        raise AssertionError("silver quota allocation does not match total")
    return {domain: result.get(domain, 0) for domain in DOMAIN_ORDER}


def training_record(record: dict[str, Any]) -> bool:
    return record["official_split"] in {"train", "research_train"}


def capacity_report(records: list[dict[str, Any]], quotas: dict[str, int], max_candidates: int) -> dict[str, Any]:
    counts = Counter(row["domain"] for row in records if training_record(row))
    report = {
        domain: {"sources": counts[domain], "capacity": counts[domain] * max_candidates, "required": quota}
        for domain, quota in quotas.items()
    }
    insufficient = {domain: row for domain, row in report.items() if row["capacity"] < row["required"]}
    if insufficient:
        raise RuntimeError(f"insufficient silver capacity: {canonical_json(insufficient)}")
    return report


def build_prompt(record: dict[str, Any], schema: SchemaSpec) -> str:
    reference = record["reference_plan"]
    if schema.domain == "webnlg":
        plan_template: dict[str, Any] = {
            "domain": "webnlg",
            "triples": [
                {
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": triple["object"],
                    "subject_type": triple.get("subject_type", "entity"),
                    "object_type": triple.get("object_type", "entity"),
                }
                for triple in reference.get("triples", [])
            ],
        }
    elif schema.domain == "dart":
        plan_template = {
            "domain": "dart",
            "schema_id": schema.schema_id,
            "fields": reference.get("fields", []),
        }
    elif schema.domain == "visfd":
        plan_template = {
            "domain": "visfd",
            "pairs": reference.get("pairs", []),
            "others": bool(reference.get("others")),
        }
    else:
        plan_template = reference

    instruction = {
        "task": "Return one verified synthetic data record.",
        "required_output": {
            "plan": plan_template,
            "text": "one concise sentence or short paragraph that explicitly mentions every subject, relation/field, and object/value in the plan",
        },
        "schema_id": schema.schema_id,
        "schema_domain": schema.domain,
        "source_id": record["source_id"],
        "hard_rules": [
            "Return exactly one JSON object.",
            "The first character must be { and the last character must be }.",
            "Do not use markdown, code fences, bullets, explanations, comments, or extra text.",
            "Keep the plan grounded in the provided plan template.",
            "Do not invent entities, predicates, columns, rows, or values.",
            "The text must be faithful to the plan and must mention every required plan term.",
        ],
    }
    if schema.domain == "visfd":
        required_phrases = [
            f"{VISFD_ASPECT_VI[pair['aspect']]} có đánh giá {VISFD_SENTIMENT_VI[pair['sentiment']]}"
            for pair in plan_template.get("pairs", [])
            if pair.get("aspect") in VISFD_ASPECT_VI and pair.get("sentiment") in VISFD_SENTIMENT_VI
        ]
        if plan_template.get("others"):
            required_phrases.append("khía cạnh khác")
        instruction["task"] = "Generate one short Vietnamese smartphone review for the supplied ViSFD semantic annotation."
        # ViSFD already provides a human annotation plan. The Teacher only realizes it into text;
        # the pipeline attaches the fixed plan after generation so a small model does not waste
        # capacity copying nested JSON.
        instruction["required_output"] = {"text": "one concise Vietnamese smartphone review"}
        instruction["source_vietnamese_comment"] = record.get("reference_text", "")
        instruction["semantic_plan"] = plan_template
        instruction["language"] = "Vietnamese only"
        instruction["required_exact_phrases"] = required_phrases
        instruction["hard_rules"] = [
            "Return exactly one JSON object with exactly one key named text.",
            "The first character must be { and the last character must be }.",
            "Do not return the semantic plan, labels, markdown, code fences, explanations, or extra keys.",
            "Write the review in Vietnamese only.",
            "The text must be faithful to the supplied semantic plan.",
            "Include every string in required_exact_phrases exactly as written so symbolic verification can check it.",
            "Do not copy the source comment verbatim; create a short new realization.",
        ]
    return canonical_json(instruction)


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.replace("```json", "```", 1)
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1].strip()

    decoder = json.JSONDecoder()
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return stripped

def visfd_candidate_from_teacher(decoded: str, record: dict[str, Any]) -> str:
    """Attach the fixed ViSFD annotation plan to a Teacher-generated realization.

    ViSFD supplies the semantic plan as a human annotation, so the Teacher need only
    produce text. This keeps the stage structure unchanged while avoiding failures
    caused by a small model copying nested plan JSON incorrectly.
    """
    extracted = extract_json_object(decoded)
    text: str | None = None
    try:
        value = json.loads(extracted)
    except (json.JSONDecodeError, TypeError):
        value = None
    if isinstance(value, dict):
        for key in ("text", "review", "comment", "response", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break
    elif decoded.strip() and not decoded.lstrip().startswith("{"):
        text = decoded.strip()

    reference = record["reference_plan"]
    plan = {
        "domain": "visfd",
        "pairs": reference.get("pairs", []),
        "others": bool(reference.get("others")),
    }
    return json.dumps({"plan": plan, "text": text or ""}, ensure_ascii=False, sort_keys=True)

def plan_text_from_plan(plan: dict[str, Any]) -> str:
    """Deterministic fallback realization aligned with constrained decoding.

    Teacher output is untrusted: malformed nested lists must be rejected, never
    crash long-running silver generation.
    """
    if not isinstance(plan, dict):
        return ""
    domain = plan.get("domain")
    if domain == "webnlg":
        triples = plan.get("triples", [])
        if not isinstance(triples, list):
            return ""
        parts = []
        for triple in triples:
            if not isinstance(triple, dict):
                return ""
            parts.append(f"{triple.get('subject')} {triple.get('predicate')} {triple.get('object')}.")
        return " ".join(parts)
    if domain == "dart":
        fields = plan.get("fields", [])
        if not isinstance(fields, list):
            return ""
        parts = []
        for field in fields:
            if not isinstance(field, dict):
                return ""
            parts.append(f"{field.get('row')} {field.get('column')} {field.get('value')}")
        return ("; ".join(parts) + ".") if parts else ""
    if domain == "visfd":
        pairs = plan.get("pairs", [])
        if not isinstance(pairs, list):
            return ""
        parts = []
        for pair in pairs:
            if not isinstance(pair, dict):
                return ""
            aspect = VISFD_ASPECT_VI.get(pair.get("aspect"))
            sentiment = VISFD_SENTIMENT_VI.get(pair.get("sentiment"))
            if not aspect or not sentiment:
                return ""
            parts.append(f"{aspect} có đánh giá {sentiment}")
        if plan.get("others"):
            parts.append("khía cạnh khác cũng được đề cập")
        return "; ".join(parts) + "." if parts else ""

    assignment = plan.get("assignment", {})
    if not isinstance(assignment, dict):
        return ""
    parts = []
    for category, values in sorted(assignment.items(), key=lambda item: str(item[0])):
        if not isinstance(values, dict):
            return ""
        for item, position in sorted(values.items(), key=lambda item: str(item[0])):
            parts.append(f"{category} {item} position {position}")
    return "; ".join(parts) + "." if parts else ""


def _teacher_plan_shape_valid(plan: dict[str, Any], schema: SchemaSpec) -> bool:
    if not isinstance(plan, dict):
        return False
    if schema.domain == "webnlg":
        triples = plan.get("triples")
        return isinstance(triples, list) and all(isinstance(row, dict) for row in triples)
    if schema.domain == "dart":
        fields = plan.get("fields")
        return isinstance(fields, list) and all(isinstance(row, dict) for row in fields)
    if schema.domain == "visfd":
        pairs = plan.get("pairs")
        return isinstance(pairs, list) and all(isinstance(row, dict) for row in pairs) and isinstance(plan.get("others", False), bool)
    assignment = plan.get("assignment")
    return isinstance(assignment, dict) and all(isinstance(values, dict) for values in assignment.values())


def normalize_teacher_output(raw: str, schema: SchemaSpec) -> str:
    extracted = extract_json_object(raw)
    try:
        value = json.loads(extracted)
    except (json.JSONDecodeError, TypeError):
        return extracted

    if not isinstance(value, dict):
        return extracted

    if isinstance(value.get("plan"), dict) and isinstance(value.get("text"), str):
        # A syntactically valid JSON object may still contain arrays where the
        # schema expects objects. Reject it safely before schema validation.
        if not _teacher_plan_shape_valid(value["plan"], schema):
            return "null"
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    if schema.domain == "webnlg":
        if "triples" in value:
            triples = value["triples"]
            if not isinstance(triples, list) or not all(isinstance(row, dict) for row in triples):
                return "null"
            plan = {"domain": "webnlg", "triples": triples}
        elif {"subject", "predicate", "object"}.issubset(value):
            plan = {"domain": "webnlg", "triples": [value]}
        else:
            return extracted
        return json.dumps({"plan": plan, "text": plan_text_from_plan(plan)}, ensure_ascii=False, sort_keys=True)

    if schema.domain == "dart":
        if "fields" in value:
            fields = value["fields"]
            if not isinstance(fields, list) or not all(isinstance(row, dict) for row in fields):
                return "null"
            plan = {"domain": "dart", "schema_id": schema.schema_id, "fields": fields}
            return json.dumps({"plan": plan, "text": plan_text_from_plan(plan)}, ensure_ascii=False, sort_keys=True)
    if schema.domain == "visfd" and "pairs" in value:
        pairs = value.get("pairs")
        if not isinstance(pairs, list) or not all(isinstance(row, dict) for row in pairs):
            return "null"
        plan = {"domain": "visfd", "pairs": pairs, "others": bool(value.get("others", False))}
        return json.dumps({"plan": plan, "text": plan_text_from_plan(plan)}, ensure_ascii=False, sort_keys=True)

    return extracted


class TransformersTeacher:
    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("full Teacher requires torch, transformers, accelerate, and bitsandbytes") from error
        teacher = config["teacher"]
        self.config = teacher
        self.model_name = teacher["model"]
        self.revision = teacher.get("revision") or "resolved-by-huggingface"
        quantization = None
        if teacher["load_in_4bit"]:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=teacher["quantization_type"],
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=teacher.get("revision"))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=teacher.get("revision"), quantization_config=quantization, device_map="auto"
        )
        self.revision = getattr(self.model.config, "_commit_hash", None) or self.revision
        self.tokenizer_revision = self.tokenizer.init_kwargs.get("_commit_hash") or self.revision
        self.last_attempt_count = 0
        self.last_fallback_used = False
        self.last_attempt_reasons: list[str] = []

    def generate(
        self, record: dict[str, Any], schema: SchemaSpec, seed: int,
        *, allow_fallback: bool = True, max_attempts: int | None = None,
    ) -> str:
        import torch
        prompt = build_prompt(record, schema)
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=self.config["max_input_tokens"])
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        attempts = max(1, int(max_attempts if max_attempts is not None else self.config.get("retries", 1)))
        last_candidate = ""
        self.last_fallback_used = False
        self.last_attempt_reasons = []
        for attempt in range(attempts):
            effective_seed = int(seed) + attempt * 1_000_003
            torch.manual_seed(effective_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(effective_seed)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"],
                    repetition_penalty=self.config["repetition_penalty"],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.decode(
                output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            self.last_attempt_count = attempt + 1
            try:
                candidate = (
                    visfd_candidate_from_teacher(decoded, record)
                    if schema.domain == "visfd"
                    else normalize_teacher_output(decoded, schema)
                )
                last_candidate = candidate
                verified = verify_candidate(candidate, schema)
            except (AttributeError, TypeError, KeyError, ValueError) as error:
                # Malformed model JSON is a rejected attempt, not a pipeline failure.
                last_candidate = decoded
                self.last_attempt_reasons.append(f"normalization_error:{type(error).__name__}")
                continue
            if verified.parse_valid and verified.phi_valid and verified.factual_consistency:
                return candidate
            self.last_attempt_reasons.append(verified.reject_reason or "verification_failed")

        if not allow_fallback:
            return last_candidate

        reference_plan = record["reference_plan"]
        if schema.domain == "webnlg":
            plan = {"domain": "webnlg", "triples": reference_plan.get("triples", [])}
        elif schema.domain == "dart":
            plan = {"domain": "dart", "schema_id": schema.schema_id, "fields": reference_plan.get("fields", [])}
        elif schema.domain == "visfd":
            plan = {"domain": "visfd", "pairs": reference_plan.get("pairs", []), "others": bool(reference_plan.get("others"))}
        else:
            plan = reference_plan
        self.last_fallback_used = True
        return json.dumps(
            {"plan": plan, "text": plan_text_from_plan(plan)},
            ensure_ascii=False, sort_keys=True,
        )



def generate_silver(
    config: dict[str, Any],
    root: Path,
    records: list[dict[str, Any]],
    schemas: dict[str, SchemaSpec],
    *,
    resume: bool = True,
    progress: Callable[[list[dict[str, Any]], list[dict[str, Any]], int], None] | None = None,
) -> dict[str, Any]:
    quotas = silver_quotas(config)
    capacity = capacity_report(records, quotas, int(config["silver"]["max_candidates_per_source"]))
    silver_dir = root / "artifacts" / "silver"
    accepted_path, rejected_path = silver_dir / "accepted.jsonl", silver_dir / "rejected.jsonl"
    accepted = read_jsonl(accepted_path) if config["checkpoint"]["resume"] and resume else []
    rejected = read_jsonl(rejected_path) if config["checkpoint"]["resume"] and resume else []
    counts = Counter(row["domain"] for row in accepted)
    attempts_per_source = Counter(row["source_id"] for row in accepted + rejected)
    attempts = len(accepted) + len(rejected)
    teacher = TransformersTeacher(config)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if training_record(record):
            by_domain[record["domain"]].append(record)
    for domain in by_domain:
        by_domain[domain].sort(key=lambda row: row["source_id"])
    maximum_attempts = int(config["silver"]["max_total_attempts"])
    checkpoint_every = int(config["silver"]["checkpoint_every_accepted"])
    base_seed = int(config["project"]["seed"])
    while any(counts[domain] < quota for domain, quota in quotas.items()):
        made_progress = False
        for domain in DOMAIN_ORDER:
            if counts[domain] >= quotas[domain]:
                continue
            candidates = [
                row for row in by_domain[domain]
                if attempts_per_source[row["source_id"]] < config["silver"]["max_candidates_per_source"]
            ]
            if not candidates or attempts >= maximum_attempts:
                shortfall = {key: quotas[key] - counts[key] for key in DOMAIN_ORDER if quotas[key] > counts[key]}
                write_jsonl(accepted_path, accepted)
                write_jsonl(rejected_path, rejected)
                write_json(silver_dir / "quota_report.json", {"quotas": quotas, "accepted": dict(counts), "shortfall": shortfall, "capacity": capacity})
                raise RuntimeError(f"silver generation stopped with shortfall: {shortfall}")
            record = candidates[(counts[domain] + len(rejected)) % len(candidates)]
            generation_index = attempts_per_source[record["source_id"]]
            generation_round = generation_index
            candidate_seed = base_seed + int(sha256_bytes(f"{record['source_id']}|{generation_index}".encode())[:8], 16)
            prompt_hash = sha256_bytes(build_prompt(record, schemas[record["source_id"]]).encode())
            raw = teacher.generate(record, schemas[record["source_id"]], candidate_seed, allow_fallback=True)
            teacher_attempt_count = int(teacher.last_attempt_count)
            teacher_fallback_used = bool(teacher.last_fallback_used)
            teacher_attempt_reasons = list(teacher.last_attempt_reasons)
            verified = verify_candidate(raw, schemas[record["source_id"]])
            sample_id = sha256_bytes(f"{record['source_id']}|{candidate_seed}".encode())[:24]
            row = {
                "sample_id": sample_id,
                "source_id": record["source_id"],
                "domain": domain,
                "official_split": record["official_split"],
                "generation_round": generation_round,
                "generation_index": generation_index,
                "candidate_seed": candidate_seed,
                "schema_version": schemas[record["source_id"]].version,
                "teacher_model": teacher.model_name,
                "teacher_revision": teacher.revision,
                "tokenizer_revision": teacher.tokenizer_revision,
                "prompt_hash": prompt_hash,
                "teacher_attempt_count": teacher_attempt_count,
                "teacher_fallback_used": teacher_fallback_used,
                "teacher_attempt_reasons": teacher_attempt_reasons,
                "semantic_plan_source": "human_annotation" if domain == "visfd" else "teacher_projected",
                "plan": verified.plan,
                "text": verified.text,
                "parse_valid": verified.parse_valid,
                "phi_valid": verified.phi_valid,
                "factual_consistency": verified.factual_consistency,
                "reject_reason": verified.reject_reason,
            }
            attempts += 1
            attempts_per_source[record["source_id"]] += 1
            if verified.parse_valid and verified.phi_valid and verified.factual_consistency:
                accepted.append(row)
                counts[domain] += 1
                made_progress = True
                if progress and len(accepted) % checkpoint_every == 0:
                    write_jsonl(accepted_path, accepted)
                    write_jsonl(rejected_path, rejected)
                    progress(accepted, rejected, attempts)
            else:
                rejected.append(row)
        if not made_progress and attempts >= maximum_attempts:
            raise RuntimeError("silver generation exhausted attempts")
    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected)
    report = {
        "required_total": sum(quotas.values()), "accepted_total": len(accepted), "rejected_total": len(rejected),
        "quotas": quotas, "accepted_by_domain": {domain: counts[domain] for domain in DOMAIN_ORDER},
        "rejected_by_domain": dict(Counter(row["domain"] for row in rejected)), "attempts": attempts, "capacity": capacity,
        "teacher_fallback_count": sum(bool(row.get("teacher_fallback_used")) for row in accepted),
        "teacher_retry_count": sum(max(0, int(row.get("teacher_attempt_count", 1)) - 1) for row in accepted),
        "semantic_plan_sources": dict(Counter(row.get("semantic_plan_source", "unknown") for row in accepted)),
    }
    if report["accepted_total"] != report["required_total"] or report["accepted_by_domain"] != quotas:
        raise AssertionError("silver quota invariant failed")
    write_json(silver_dir / "quota_report.json", report)
    return {"accepted": accepted, "rejected": rejected, "report": report, "artifacts": [accepted_path, rejected_path, silver_dir / "quota_report.json"]}

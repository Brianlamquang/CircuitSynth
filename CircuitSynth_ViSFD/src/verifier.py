from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .schema import SchemaSpec, plan_terms


@dataclass(frozen=True)
class VerificationResult:
    parse_valid: bool
    phi_valid: bool
    factual_consistency: bool
    reject_reason: str | None
    plan: dict[str, Any] | None
    text: str | None
    errors: list[dict[str, str]]


def canonical_text(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


# ViSFD is a Vietnamese-only experiment. We allow Latin text because Vietnamese
# smartphone reviews naturally contain product names/loanwords (e.g. iPhone, camera),
# but reject scripts that clearly indicate a switch to another language family.
_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # CJK
    r"\u3040-\u30ff"                                  # Japanese kana
    r"\uac00-\ud7af"                                  # Hangul
    r"\u0400-\u052f"                                  # Cyrillic
    r"\u0600-\u06ff"                                  # Arabic
    r"\u0e00-\u0e7f]"                                 # Thai
)


def visfd_language_valid(text: str) -> bool:
    return bool(text.strip()) and _FOREIGN_SCRIPT_RE.search(text) is None


def text_consistent(plan: dict[str, Any], text: str) -> bool:
    normalized = canonical_text(text)
    entities, relations = plan_terms(plan)
    required = {canonical_text(term) for term in entities | relations if canonical_text(term)}
    padded = f" {normalized} "
    return all(f" {term} " in padded for term in required)


def verify_candidate(candidate: str | dict[str, Any], schema: SchemaSpec) -> VerificationResult:
    try:
        parsed = json.loads(candidate) if isinstance(candidate, str) else candidate
    except (json.JSONDecodeError, TypeError) as error:
        return VerificationResult(False, False, False, f"json_parse:{error}", None, None, [])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("plan"), dict) or not isinstance(parsed.get("text"), str):
        return VerificationResult(False, False, False, "required_fields", None, None, [])
    plan, text = parsed["plan"], parsed["text"]
    validation = schema.validate_plan(plan)
    errors = [asdict(error) for error in validation]
    if validation:
        return VerificationResult(True, False, False, validation[0].code, plan, text, errors)
    if schema.domain == "visfd" and not visfd_language_valid(text):
        return VerificationResult(True, True, False, "language_violation", plan, text, errors)
    factual = text_consistent(plan, text)
    return VerificationResult(True, True, factual, None if factual else "plan_text_inconsistent", plan, text, errors)


def verify_plan(plan: dict[str, Any], schema: SchemaSpec) -> VerificationResult:
    validation = schema.validate_plan(plan)
    errors = [asdict(error) for error in validation]
    return VerificationResult(True, not validation, True, validation[0].code if validation else None, plan, None, errors)

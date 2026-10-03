from __future__ import annotations

import gc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .checkpoint import canonical_json, read_jsonl, sha256_bytes, write_json, write_jsonl
from .circuits import CircuitDistribution
from .data import largest_remainder
from .evaluate import evaluation_variant_budgets
from .schema import SchemaSpec, plan_terms
from .student import load_full_student
from .teacher import TransformersTeacher
from .verifier import canonical_text, verify_candidate


class DeadEndError(RuntimeError):
    def __init__(self, trace: dict[str, Any]):
        super().__init__("token automaton reached a state with no legal continuation")
        self.trace = trace


@dataclass
class TokenFSA:
    transitions: dict[int, dict[int, int]]
    start_state: int
    accepting_states: set[int]
    eos_ready_states: set[int]
    eos_token_id: int

    @classmethod
    def from_sequences(cls, sequences: Iterable[list[int]], eos_token_id: int) -> "TokenFSA":
        transitions: dict[int, dict[int, int]] = {0: {}}
        eos_ready: set[int] = set()
        next_state = 1
        for sequence in sequences:
            state = 0
            for token in sequence:
                if token not in transitions[state]:
                    transitions[state][token] = next_state
                    transitions[next_state] = {}
                    next_state += 1
                state = transitions[state][token]
            eos_ready.add(state)
        accepting: set[int] = set()
        for state in eos_ready:
            if eos_token_id not in transitions[state]:
                transitions[state][eos_token_id] = next_state
                transitions[next_state] = {}
                accepting.add(next_state)
                next_state += 1
            else:
                accepting.add(transitions[state][eos_token_id])
        return cls(transitions, 0, accepting, eos_ready, eos_token_id)

    def next_tokens(self, state: int) -> set[int]:
        return set(self.transitions.get(state, {}))

    def step(self, state: int, token: int) -> int:
        if token not in self.transitions.get(state, {}):
            raise ValueError(f"illegal token {token} from state {state}")
        if token == self.eos_token_id and state not in self.eos_ready_states:
            raise ValueError("EOS is legal only after a complete controlled realization")
        return self.transitions[state][token]

    def accepts(self, tokens: list[int]) -> bool:
        state = self.start_state
        try:
            for token in tokens:
                state = self.step(state, token)
        except ValueError:
            return False
        return state in self.accepting_states


def controlled_realizations(plan: dict[str, Any]) -> list[str]:
    domain = plan["domain"]
    if domain == "webnlg":
        rows = plan["triples"]
        values = [" ".join(f"{row['subject']} {row['predicate']} {row['object']}." for row in order) for order in (rows, list(reversed(rows)))]
        return list(dict.fromkeys(values))
    if domain == "dart":
        rows = plan["fields"]
        values = ["; ".join(f"{row['row']} {row['column']} {row['value']}" for row in order) + "." for order in (rows, list(reversed(rows)))]
        return list(dict.fromkeys(values))
    fields = [
        f"{category} {item} position {position}"
        for category, values in sorted(plan["assignment"].items())
        for item, position in sorted(values.items())
    ]
    return list(dict.fromkeys(["; ".join(fields) + ".", "; ".join(reversed(fields)) + "."]))


def _encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_plan_fsa(plan: dict[str, Any], tokenizer: Any) -> TokenFSA:
    sequences = [_encode(tokenizer, text) for text in controlled_realizations(plan)]
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define EOS")
    return TokenFSA.from_sequences(sequences, eos)


def constrained_full_generate(model: Any, tokenizer: Any, prompt: str, automaton: TokenFSA, max_tokens: int, seed: int) -> tuple[str, list[int], list[float]]:
    import torch
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    state = automaton.start_state
    generated: list[int] = []
    log_probabilities: list[float] = []
    generator = torch.Generator(device=model.device).manual_seed(seed)
    for position in range(max_tokens):
        legal = sorted(automaton.next_tokens(state))
        if not legal:
            raise DeadEndError({"position": position, "state": state, "tokens": generated})
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0, -1]
        log_probs = torch.log_softmax(logits, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask[legal] = logits[legal]
        probabilities = torch.softmax(mask, dim=-1)
        token = int(torch.multinomial(probabilities, 1, generator=generator))
        generated.append(token)
        log_probabilities.append(float(log_probs[token]))
        state = automaton.step(state, token)
        input_ids = torch.cat([input_ids, torch.tensor([[token]], device=model.device)], dim=1)
        if state in automaton.accepting_states:
            return tokenizer.decode(generated, skip_special_tokens=True), generated, log_probabilities
    raise DeadEndError({"position": max_tokens, "state": state, "tokens": generated, "reason": "max_tokens"})


def unconstrained_full_generate(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int, seed: int,
    temperature: float, top_p: float, top_k: int,
) -> tuple[str, list[int], list[float]]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.inference_mode():
        result = model.generate(
            **encoded,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated = result.sequences[0, encoded["input_ids"].shape[1]:]
    tokens = generated.tolist()
    log_probabilities: list[float] = []

    for token, scores in zip(tokens, result.scores):
        log_probs = torch.log_softmax(scores[0], dim=-1)
        log_probabilities.append(float(log_probs[token].detach().cpu()))

    text = tokenizer.decode(tokens, skip_special_tokens=True)

    del result, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text, tokens, log_probabilities


def _post_validate(plan: dict[str, Any], text: str, schema: SchemaSpec, fsa: TokenFSA | None, tokens: list[int]) -> dict[str, Any]:
    verified = verify_candidate({"plan": plan, "text": text}, schema)
    entities, relations = plan_terms(plan)
    normalized = f" {canonical_text(text)} "
    contains = lambda term: f" {canonical_text(term)} " in normalized
    return {
        "parse_valid": verified.parse_valid,
        "plan_consistent": verified.factual_consistency,
        "schema_valid": verified.phi_valid,
        "required_entities_present": all(contains(term) for term in entities),
        "required_relations_present": all(contains(term) for term in relations),
        "forbidden_entities": [],
        "forbidden_relations": [],
        "fsa_accepted": fsa.accepts(tokens) if fsa else None,
    }


def _evaluation_records(config: dict[str, Any], records: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    held_out = [row for row in records if row["official_split"] in {"test", "research_test"}]
    if not held_out:
        raise RuntimeError("no held-out IDs available for generation")
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in held_out:
        by_domain[row["domain"]].append(row)
    for rows in by_domain.values():
        rows.sort(key=lambda row: row["source_id"])
    order = list(config["evaluation"].get("domain_tie_break_order", sorted(by_domain)))
    weights = {
        domain: float(config["evaluation"].get("domain_weights", {}).get(domain, len(by_domain.get(domain, []))))
        for domain in order
    }
    allocation = largest_remainder(budget, weights, order)
    for domain, count in allocation.items():
        if count and not by_domain.get(domain):
            raise RuntimeError(f"evaluation domain {domain} has no held-out records")

    selected: list[dict[str, Any]] = []
    cursors = Counter()
    total_weight = sum(weights.values())
    order_index = {domain: index for index, domain in enumerate(order)}
    for step in range(budget):
        candidates = [domain for domain in order if cursors[domain] < allocation[domain]]
        if not candidates:
            break
        domain = max(
            candidates,
            key=lambda value: (
                (step + 1) * weights[value] / total_weight - cursors[value],
                -order_index[value],
            ),
        )
        selected.append(by_domain[domain][cursors[domain] % len(by_domain[domain])])
        cursors[domain] += 1
    if len(selected) != budget or any(cursors[domain] != allocation[domain] for domain in order):
        raise AssertionError(f"evaluation allocation mismatch: {dict(cursors)} != {allocation}")
    return selected


def _sample_circuit_for_domains(circuit: CircuitDistribution, domains: list[str], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(circuit.states):
        by_domain[state["plan"]["domain"]].append(index)
    sampled: list[dict[str, Any]] = []
    for position, domain in enumerate(domains):
        indices = by_domain.get(domain, [])
        if not indices:
            raise RuntimeError(f"circuit has no support for evaluation domain {domain}")
        probabilities = circuit.probabilities[indices]
        probabilities = probabilities / probabilities.sum()
        local = int(rng.choice(len(indices), p=probabilities))
        index = indices[local]
        state = circuit.states[index]
        sampled.append({
            "plan_id": f"eval-plan-{position:08d}",
            "circuit_state_id": state["state_id"],
            "source_id": state["source_id"],
            "domain": domain,
            "sampling_seed": seed + position,
            "plan": state["plan"],
            "probability": float(circuit.probabilities[index]),
        })
    return sampled


def generate_outputs(
    config: dict[str, Any], root: Path, records: list[dict[str, Any]], schemas: dict[str, SchemaSpec],
    accepted: list[dict[str, Any]], base: CircuitDistribution, projected: CircuitDistribution,
    model: Any, tokenizer: Any, *, resume: bool = True, progress: Any | None = None,
) -> dict[str, Any]:
    budgets = evaluation_variant_budgets(config)
    if not budgets:
        raise RuntimeError("no enabled evaluation variants")
    max_budget = max(budgets.values())
    eval_records = _evaluation_records(config, records, max_budget)
    eval_ids = [row["source_id"] for row in eval_records]
    eval_domains = [row["domain"] for row in eval_records]
    seeds = [int(config["project"]["seed"]) + 9000 + index for index in range(max_budget)]
    base_plans = _sample_circuit_for_domains(base, eval_domains, int(config["project"]["seed"]) + 7000)
    projected_plans = _sample_circuit_for_domains(projected, eval_domains, int(config["project"]["seed"]) + 7000)
    variants = config["evaluation"]["variants"]
    records_by_id = {row["source_id"]: row for row in records}

    def evaluation_group(source_id: str, domain: str) -> str:
        record = records_by_id.get(source_id, {})
        if domain == "webnlg":
            return str(record.get("category", "unknown"))
        if domain == "dart":
            return str(record.get("source", "unknown"))
        return domain.removeprefix("zebra_").capitalize()

    output_path = root / "artifacts" / "outputs" / "generated.jsonl"
    outputs = read_jsonl(output_path) if resume else []
    completed = {(row["variant"], row["generation_index"]) for row in outputs}
    teacher_budget = budgets.get("teacher_direct", 0)
    teacher = TransformersTeacher(config) if variants.get("teacher_direct") and any(
        ("teacher_direct", index) not in completed for index in range(teacher_budget)
    ) else None
    dead_end_path = root / "artifacts" / "outputs" / "dead_end.json"
    for variant, enabled in variants.items():
        if not enabled:
            continue
        budget = budgets[variant]
        for index in range(budget):
            if (variant, index) in completed:
                continue
            started = perf_counter()
            held_out_record = eval_records[index]
            if variant in {"teacher_direct", "distill_only"}:
                plan_row = {
                    "plan_id": f"heldout-{index:08d}", "source_id": held_out_record["source_id"],
                    "plan": held_out_record["reference_plan"],
                }
            elif variant in {"psdd_pgd", "full_pipeline"}:
                plan_row = projected_plans[index]
            else:
                plan_row = base_plans[index]
            plan, source_id = plan_row["plan"], plan_row["source_id"]
            schema = schemas[source_id]
            prompt = f"Realize the semantic plan faithfully.\nPlan: {canonical_json(plan)}\nText:"
            fsa = None
            if variant == "teacher_direct":
                assert teacher is not None
                candidate = teacher.generate(held_out_record, schema, seeds[index], allow_fallback=False, max_attempts=1)
                verified_teacher = verify_candidate(candidate, schema)
                if verified_teacher.plan and verified_teacher.plan.get("domain") == held_out_record["domain"]:
                    plan = verified_teacher.plan
                text = verified_teacher.text if verified_teacher.text is not None else candidate
                tokens: list[int] = []
                log_probabilities: list[float] = []
            else:
                if model is None or tokenizer is None:
                    model, tokenizer = load_full_student(config, root / "artifacts" / "student")
                constrained = variant in {"psdd_fsa", "full_pipeline"}
                try:
                    if constrained:
                        fsa = build_plan_fsa(plan, tokenizer)
                        text, tokens, log_probabilities = constrained_full_generate(
                            model, tokenizer, prompt, fsa, config["generation"]["max_new_tokens"], seeds[index]
                        )
                    else:
                        text, tokens, log_probabilities = unconstrained_full_generate(
                            model, tokenizer, prompt, config["generation"]["max_new_tokens"], seeds[index],
                            config["generation"]["temperature"], config["generation"]["top_p"], config["generation"]["top_k"],
                        )
                except DeadEndError as error:
                    write_json(dead_end_path, {"variant": variant, "eval_id": eval_ids[index], "trace": error.trace})
                    raise
            validation = _post_validate(plan, text, schema, fsa, tokens)
            if variant == "teacher_direct":
                validation.update({
                    "parse_valid": verified_teacher.parse_valid, "schema_valid": verified_teacher.phi_valid,
                    "plan_consistent": verified_teacher.factual_consistency,
                })
            output_domain = held_out_record["domain"] if variant == "teacher_direct" else plan["domain"]
            outputs.append({
                "output_id": sha256_bytes(f"{variant}|{eval_ids[index]}|{seeds[index]}".encode())[:24],
                "variant": variant, "eval_id": eval_ids[index], "generation_seed": seeds[index],
                "generation_index": index,
                "plan_id": plan_row["plan_id"], "source_id": source_id, "domain": output_domain,
                "dataset": "zebralogic" if str(output_domain).startswith("zebra_") else str(output_domain),
                "evaluation_group": evaluation_group(source_id, str(output_domain)),
                "plan": plan, "text": text, "tokens": tokens, "token_log_probabilities": log_probabilities,
                "validation": validation, "generation_ms": (perf_counter() - started) * 1000.0,
            })
            if len(outputs) % int(config["generation"]["checkpoint_every_outputs"]) == 0:
                write_jsonl(output_path, outputs)
                if progress:
                    progress(outputs)
        if variant == "teacher_direct":
            if teacher is not None:
                try:
                    del teacher.model
                except AttributeError:
                    pass
            teacher = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    manifest_path = root / "artifacts" / "outputs" / "generation_manifest.json"
    write_jsonl(output_path, outputs)
    expected_outputs = sum(budgets.values())
    write_json(manifest_path, {
        "variant_budgets": budgets, "max_budget": max_budget,
        "variants": [name for name, enabled in variants.items() if enabled],
        "evaluation_domain_counts": dict(Counter(eval_domains)),
        "eval_ids_sha256": sha256_bytes(canonical_json(eval_ids).encode()),
        "seeds_sha256": sha256_bytes(canonical_json(seeds).encode()),
        "output_count": len(outputs), "expected_output_count": expected_outputs,
    })
    if len(outputs) != expected_outputs:
        raise AssertionError(f"generation output count mismatch: {len(outputs)} != {expected_outputs}")
    if model is not None:
        del model
    tokenizer = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {"outputs": outputs, "artifacts": [output_path, manifest_path]}

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .checkpoint import write_json, write_jsonl
from .data import half_up_count
from .schema import plan_cardinality, serialize_plan


def dynamic_optimizer_steps(train_count: int, epochs: int, micro_batch_size: int, accumulation: int) -> int:
    if min(train_count, epochs, micro_batch_size, accumulation) <= 0:
        raise ValueError("training counts and batch parameters must be positive")
    return math.ceil(train_count * epochs / (micro_batch_size * accumulation))




def deterministic_training_rows(rows: list[dict[str, Any]], seed: int, enabled: bool = True) -> list[dict[str, Any]]:
    """Return a reproducibly shuffled training order.

    Silver rows are otherwise grouped by domain/cardinality after the group-aware
    split. A seeded shuffle prevents long domain blocks from biasing the single
    local epoch while retaining exact reproducibility and resume compatibility.
    """
    ordered = list(rows)
    if enabled:
        random.Random(seed).shuffle(ordered)
    return ordered

def checkpoint_steps(config: dict[str, Any], total_steps: int) -> list[int]:
    checkpoint = config["student"]["checkpoint"]
    values = set(range(int(checkpoint["every_steps"]), total_steps + 1, int(checkpoint["every_steps"])))
    for step in checkpoint.get("explicit_steps", []):
        if step <= total_steps:
            values.add(int(step))
        elif checkpoint.get("strict_explicit_steps"):
            raise ValueError(f"explicit checkpoint {step} exceeds final step {total_steps}")
    if checkpoint.get("save_final"):
        values.add(total_steps)
    return sorted(values)


def _group_subset(groups: list[tuple[str, list[dict[str, Any]]]], target: int) -> set[str]:
    by_size_and_stratum: dict[int, dict[tuple[str, int], list[str]]] = defaultdict(lambda: defaultdict(list))
    for group_id, rows in groups:
        stratum = (rows[0]["domain"], rows[0]["plan_cardinality"])
        by_size_and_stratum[len(rows)][stratum].append(group_id)
    ordered_by_size: dict[int, list[str]] = {}
    for size, strata in by_size_and_stratum.items():
        ordered: list[str] = []
        keys = sorted(strata)
        offset = 0
        while any(offset < len(strata[key]) for key in keys):
            for key in keys:
                if offset < len(strata[key]):
                    ordered.append(strata[key][offset])
            offset += 1
        ordered_by_size[size] = ordered
    chunks: list[tuple[int, list[str]]] = []
    for size, group_ids in sorted(ordered_by_size.items()):
        offset, width = 0, 1
        while offset < len(group_ids):
            selected = group_ids[offset: offset + width]
            chunks.append((size * len(selected), selected))
            offset += len(selected)
            width *= 2
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for chunk_index, (weight, _) in enumerate(chunks):
        for subtotal in sorted(list(parents), reverse=True):
            candidate = subtotal + weight
            if candidate <= target and candidate not in parents:
                parents[candidate] = (subtotal, chunk_index)
    best = max(parents)
    chosen: set[str] = set()
    cursor = best
    while cursor:
        parent = parents[cursor]
        if parent is None:
            break
        cursor, chunk_index = parent
        chosen.update(chunks[chunk_index][1])
    return chosen


def split_silver(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = config["student"]["split"]
    if not math.isclose(split["train_fraction"] + split["validation_fraction"], 1.0, abs_tol=1e-9):
        raise ValueError("Student split fractions must sum to one")
    override = split.get("count_overrides", {})
    validation_target = override.get("validation")
    validation_target = int(validation_target) if validation_target is not None else half_up_count(len(rows), split["validation_fraction"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row = dict(row)
        row["plan_cardinality"] = plan_cardinality(row["plan"])
        groups[row["source_id"]].append(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: (item[1][0]["domain"], item[1][0]["plan_cardinality"], item[0]),
    )
    validation_groups = _group_subset(ordered, validation_target)
    validation = [row for group_id, group_rows in ordered if group_id in validation_groups for row in group_rows]
    train = [row for group_id, group_rows in ordered if group_id not in validation_groups for row in group_rows]
    if override.get("train") is not None and len(train) != int(override["train"]):
        raise ValueError("group-aware split cannot satisfy train count override")
    if len(validation) != validation_target:
        raise ValueError(f"group-aware split cannot satisfy validation count {validation_target}; nearest={len(validation)}")
    if {row["source_id"] for row in train} & {row["source_id"] for row in validation}:
        raise AssertionError("source group leakage in Student split")
    return train, validation


def build_student_prompt(plan: dict[str, Any]) -> str:
    serialized = serialize_plan(plan)
    if plan.get("domain") == "visfd":
        return (
            "Hãy hiện thực hóa kế hoạch ngữ nghĩa sau thành một đánh giá điện thoại bằng tiếng Việt. "
            "Chỉ trả về nội dung đánh giá bằng tiếng Việt, không giải thích và không dùng ngôn ngữ khác.\n"
            f"Kế hoạch: {serialized}\n"
            "Đánh giá:"
        )
    return f"Realize the semantic plan faithfully.\nPlan: {serialized}\nText:"


def format_example(row: dict[str, Any]) -> tuple[str, str]:
    return build_student_prompt(row["plan"]), row["text"]


def render_generation_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _full_labels(tokenizer: Any, prompt: str, target: str, max_length: int) -> tuple[Any, Any]:
    import torch
    rendered_prompt = render_generation_prompt(tokenizer, prompt)
    target_ids = tokenizer(
        target + tokenizer.eos_token, add_special_tokens=False, truncation=True, max_length=max_length - 1
    )["input_ids"]
    if not target_ids:
        raise ValueError("target tokenization is empty")
    prompt_budget = max(1, max_length - len(target_ids))
    prompt_ids = tokenizer(
        rendered_prompt, add_special_tokens=False, truncation=True, max_length=prompt_budget
    )["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return torch.tensor([input_ids]), torch.tensor([labels])


def train_full(rows: list[dict[str, Any]], config: dict[str, Any], output: Path, total_steps: int, resume: bool = True) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
    except ImportError as error:
        raise RuntimeError("full Student requires torch, transformers, peft, accelerate, and bitsandbytes") from error
    student = config["student"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=student["load_in_4bit"], bnb_4bit_quant_type=student["quantization_type"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(student["model"], revision=student.get("revision"))
    base_model = AutoModelForCausalLM.from_pretrained(
        student["model"], revision=student.get("revision"), quantization_config=quantization, device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)
    candidates = sorted(
        [path for path in (output / "checkpoints").glob("step-*") if (path / "trainer.pt").exists()],
        key=lambda path: int(path.name.split("-")[-1]),
    ) if resume and (output / "checkpoints").exists() else []
    resume_dir = candidates[-1] if candidates else None
    if resume_dir:
        model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    else:
        model = get_peft_model(base_model, LoraConfig(
            r=student["lora_rank"], lora_alpha=student["lora_alpha"], lora_dropout=student["lora_dropout"],
            task_type="CAUSAL_LM", target_modules="all-linear",
        ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=student["learning_rate"], weight_decay=student["weight_decay"])
    warmup = int(total_steps * student["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    accumulation = student["gradient_accumulation_steps"]
    start_step = 0
    losses: list[float] = []
    if resume_dir:
        state = torch.load(resume_dir / "trainer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        losses = list(state.get("losses", []))
        torch.random.set_rng_state(state["rng_cpu"])
        if torch.cuda.is_available() and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    training_rows = deterministic_training_rows(
        rows,
        int(config["project"]["seed"]) + 6000,
        bool(student.get("shuffle_training", True)),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, total_steps + 1):
        step_loss = 0.0
        for micro in range(accumulation):
            row = training_rows[((step - 1) * accumulation + micro) % len(training_rows)]
            input_ids, labels = _full_labels(tokenizer, *format_example(row), student["max_sequence_length"])
            input_ids, labels = input_ids.to(model.device), labels.to(model.device)
            loss = model(input_ids=input_ids, labels=labels).loss / accumulation
            loss.backward()
            step_loss += float(loss.detach())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(step_loss)
        if step in checkpoint_steps(config, total_steps):
            target = output / "checkpoints" / f"step-{step}"
            model.save_pretrained(target)
            torch.save({
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step,
                "losses": losses, "rng_cpu": torch.random.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }, target / "trainer.pt")
    final = output / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    write_json(final / "trainer_state.json", {"optimizer_step": total_steps, "losses": losses})
    return {"model": model, "tokenizer": tokenizer, "current_step": total_steps, "losses": losses, "artifacts": [final / "adapter_model.safetensors", final / "trainer_state.json"]}


def load_full_student(config: dict[str, Any], output: Path) -> tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("full Student loading requires torch, transformers, peft, accelerate, and bitsandbytes") from error
    student = config["student"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=student["load_in_4bit"], bnb_4bit_quant_type=student["quantization_type"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(output / "final")
    base = AutoModelForCausalLM.from_pretrained(
        student["model"], revision=student.get("revision"), quantization_config=quantization, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, output / "final")
    model.eval()
    return model, tokenizer


def train_student(
    config: dict[str, Any], root: Path, accepted: list[dict[str, Any]], *, resume: bool = True
) -> dict[str, Any]:
    train, validation = split_silver(accepted, config)
    output = root / "artifacts" / "student"
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    calculated = dynamic_optimizer_steps(
        len(train), config["student"]["epochs"], config["student"]["micro_batch_size"],
        config["student"]["gradient_accumulation_steps"],
    )
    total = calculated
    result = train_full(train, config, output, total, resume=resume)
    report = {
        "train_count": len(train), "validation_count": len(validation), "calculated_optimizer_steps": calculated,
        "executed_optimizer_steps": total, "current_step": result["current_step"],
        "checkpoints": checkpoint_steps(config, calculated),
        "target_only_masking": True,
    }
    write_json(output / "training_report.json", report)
    result.update({"train": train, "validation": validation, "report": report})
    result["artifacts"] = sorted(
        {path for path in result["artifacts"] + list((output / "final").glob("*")) if path.is_file()},
        key=str,
    ) + [output / "train.jsonl", output / "validation.jsonl", output / "training_report.json"]
    return result

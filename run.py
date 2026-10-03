from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.checkpoint import (
    STAGES,
    CheckpointStore,
    JsonlLogger,
    read_jsonl,
    scientific_hash,
    atomic_write_bytes,
    write_json,
)
from src.circuits import backend_conformance, compile_and_fit, load_circuit, project_distribution, resolve_psdd_backend, sample_plans
from src.data import load_prepared, prepare_data
from src.evaluate import evaluate_outputs, evaluation_variant_budgets
from src.fsa import generate_outputs
from src.student import dynamic_optimizer_steps, train_student
from src.teacher import generate_silver, silver_quotas, silver_total


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = {
    0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [1], 6: [4, 5], 7: [1, 2, 3, 4, 5, 6],
}


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use dotted.path=value: {override}")
        dotted, raw = override.split("=", 1)
        value = yaml.safe_load(raw)
        target = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"unknown config path: {dotted}")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"unknown config path: {dotted}")
        target[keys[-1]] = value
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    supported = {
        "project.numerical_dtype": (config["project"]["numerical_dtype"], "float64"),
        "project.count_policy": (config["project"]["count_policy"], "validate"),
        "data.count_rounding": (config["data"]["count_rounding"], "half_up"),
        "data.preserve_official_splits": (config["data"]["preserve_official_splits"], True),
        "data.sampling_without_replacement": (config["data"]["sampling_without_replacement"], True),
        "psdd.backend": (config["psdd"]["backend"], "internal_psdd"),
        "psdd.compiler": (config["psdd"]["compiler"], "deterministic_plan_circuit"),
        "psdd.require_positive_parameters": (config["psdd"]["require_positive_parameters"], True),
        "pgd.parameterization": (config["pgd"]["parameterization"], "edge_flow"),
        "soft_constraints.target_source": (config["soft_constraints"]["target_source"], "silver_train"),
        "checkpoint.atomic_write": (config["checkpoint"]["atomic_write"], True),
        "checkpoint.checksum": (config["checkpoint"]["checksum"], "sha256"),
    }
    incompatible = [name for name, (value, expected) in supported.items() if value != expected]
    if incompatible:
        raise ValueError(f"unsupported invariant configuration: {incompatible}")
    if not 0 < float(config["data"]["sample_fraction"]) <= 1:
        raise ValueError("data.sample_fraction must be in (0, 1]")
    if not 0 < float(config["silver"]["sample_fraction"]) <= 1:
        raise ValueError("silver.sample_fraction must be in (0, 1]")
    split = config["student"]["split"]
    train_fraction = float(split["train_fraction"])
    validation_fraction = float(split["validation_fraction"])
    if abs(train_fraction + validation_fraction - 1.0) > 1e-9:
        raise ValueError("Student split fractions must sum to one")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Student split fractions must both lie in (0, 1)")
    positive = {
        "teacher.retries": config["teacher"]["retries"],
        "teacher.batch_size": config["teacher"]["batch_size"],
        "silver.max_candidates_per_source": config["silver"]["max_candidates_per_source"],
        "silver.max_total_attempts": config["silver"]["max_total_attempts"],
        "psdd.mle_smoothing_alpha": config["psdd"]["mle_smoothing_alpha"],
        "pgd.max_iterations": config["pgd"]["max_iterations"],
        "sampling.total_plans": config["sampling"]["total_plans"],
        "student.micro_batch_size": config["student"]["micro_batch_size"],
        "student.gradient_accumulation_steps": config["student"]["gradient_accumulation_steps"],
        "student.epochs": config["student"]["epochs"],
        "student.learning_rate": config["student"]["learning_rate"],
        "evaluation.generation_budget_per_variant": config["evaluation"]["generation_budget_per_variant"],
    }
    bad = [name for name, value in positive.items() if float(value) <= 0]
    if bad:
        raise ValueError(f"positive configuration values required: {bad}")
    variant_budgets = evaluation_variant_budgets(config)
    bad_budgets = [name for name, value in variant_budgets.items() if int(value) <= 0]
    if bad_budgets:
        raise ValueError(f"positive evaluation variant budgets required: {bad_budgets}")
    max_evaluation_budget = max(variant_budgets.values()) if variant_budgets else 0
    if int(config["sampling"]["total_plans"]) < max_evaluation_budget:
        raise ValueError("sampling.total_plans must cover the largest enabled evaluation variant budget")
    if int(config["student"]["micro_batch_size"]) != 1 or int(config["teacher"]["batch_size"]) != 1 or int(config["generation"]["batch_size"]) != 1:
        raise ValueError("the current Trainer, Teacher, and generator require batch_size=1")
    quotas = silver_quotas(config)
    enabled = {name for name in ("webnlg", "dart", "zebralogic") if config["data"][name].get("enabled", True)}
    for domain, quota in quotas.items():
        dataset = "zebralogic" if domain.startswith("zebra_") else domain
        if dataset not in enabled and quota:
            raise ValueError(f"disabled dataset {dataset} has non-zero silver quota for {domain}")


def save_resolved_config(config: dict[str, Any], root: Path, overrides: list[str]) -> None:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    resolved = deepcopy(config)
    resolved["resolved"] = {"cli_overrides": overrides, "scientific_config_hash": scientific_hash(config)}
    target = artifacts / "resolved_config.yaml"
    atomic_write_bytes(target, yaml.safe_dump(resolved, sort_keys=False).encode())


def _system_ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except (ImportError, AttributeError):
        return None


def _cuda_report() -> dict[str, Any]:
    report = {"available": False, "gpu": None, "usable_vram_bytes": 0, "cuda_version": None, "driver_version": None}
    if not shutil.which("nvidia-smi"):
        return report
    try:
        fields = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()[0].split(",")
        report.update({
            "available": True, "gpu": fields[0].strip(), "usable_vram_bytes": int(fields[1]) * 1024**2,
            "total_vram_bytes": int(fields[2]) * 1024**2,
        })
        report["driver_version"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10
        ).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        import torch
        report["cuda_version"] = torch.version.cuda
        report["available"] = bool(report["available"] and torch.cuda.is_available())
    except ImportError:
        report["available"] = False
    return report


def _hub_access(config: dict[str, Any]) -> tuple[dict[str, bool], dict[str, bool]]:
    dataset_access = {
        name: False for name in ("webnlg", "dart", "zebralogic")
        if config["data"][name].get("enabled", True)
    }
    model_access = {"teacher": False, "student": False}
    if importlib.util.find_spec("huggingface_hub") is None:
        return dataset_access, model_access
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for name in dataset_access:
            spec = config["data"][name]
            if spec.get("source_mode") == "builtin":
                dataset_access[name] = True
                continue
            try:
                api.dataset_info(spec["source"], revision=spec.get("revision"))
                dataset_access[name] = True
            except Exception:
                pass
        for role in model_access:
            spec = config[role]
            try:
                api.model_info(spec["model"], revision=spec.get("revision"))
                model_access[role] = True
            except Exception:
                pass
    except Exception:
        pass
    return dataset_access, model_access


def preflight(config: dict[str, Any], root: Path) -> dict[str, Any]:
    cuda = _cuda_report()
    dataset_access, model_access = _hub_access(config)
    psdd: dict[str, Any]
    try:
        backend = resolve_psdd_backend(config)
        conformance = backend_conformance(backend)
        psdd = {"available": True, **backend, "conformance": conformance}
    except RuntimeError as error:
        psdd = {"available": False, "error": str(error)}
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("datasets", "huggingface_hub", "torch", "transformers", "peft", "bitsandbytes")
    }
    train_total = silver_total(config)
    train_count = int(train_total * config["student"]["split"]["train_fraction"] + 0.5)
    optimizer_steps = dynamic_optimizer_steps(
        train_count, config["student"]["epochs"], config["student"]["micro_batch_size"],
        config["student"]["gradient_accumulation_steps"],
    )
    free_disk = shutil.disk_usage(root).free
    estimated_checkpoint = int(0.9 * 1024**3 * max(1, len(config["student"]["checkpoint"]["explicit_steps"]) + 1))
    checks = {
        "cuda": cuda["available"],
        "required_packages": all(packages.values()),
        "dataset_access": all(dataset_access.values()),
        "model_access": all(model_access.values()),
        "quantization_backend": packages["bitsandbytes"] and cuda["available"],
        "psdd_compiler": psdd["available"],
        "disk": free_disk > estimated_checkpoint,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "python_version": platform.python_version(),
        "cuda": cuda,
        "system_ram_bytes": _system_ram_bytes(),
        "free_disk_bytes": free_disk,
        "packages": packages,
        "dataset_access": dataset_access,
        "model_access": model_access,
        "psdd": psdd,
        "estimated_teacher_generation_count": {"minimum": train_total, "maximum": int(config["silver"]["max_total_attempts"])},
        "estimated_student_optimizer_steps": optimizer_steps,
        "evaluation_variant_budgets": evaluation_variant_budgets(config),
        "estimated_generation_output_count": sum(evaluation_variant_budgets(config).values()),
        "estimated_checkpoint_disk_bytes": estimated_checkpoint,
    }
    path = root / "artifacts" / "metrics" / "preflight.json"
    write_json(path, report)
    return report


def _load_silver(root: Path) -> list[dict[str, Any]]:
    return read_jsonl(root / "artifacts" / "silver" / "accepted.jsonl")


def run_stage(index: int, config: dict[str, Any], root: Path, store: CheckpointStore, logger: JsonlLogger) -> None:
    for dependency in DEPENDENCIES[index]:
        valid, reason = store.validate(dependency)
        if not valid:
            raise RuntimeError(f"{STAGES[index]} requires {STAGES[dependency]}: {reason}")
    resume_compatible, _ = store.validate(index, require_complete=False)
    store.save(index, "in_progress", upstream=DEPENDENCIES[index])
    logger.log("stage_start", stage=STAGES[index])
    try:
        records, schemas = load_prepared(root) if index else ([], {})
        if index == 0:
            result = prepare_data(config, root)
            metadata = {
                "dataset_revisions": result["revisions"],
                "schema_versions": sorted({schema.version for schema in result["schemas"].values()}),
                "selected_counts": {f"{row['dataset']}:{row['official_split']}": row["selected_count"] for row in result["manifests"]},
            }
            count, current_step = len(result["records"]), None
        elif index == 1:
            def progress(accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], attempts: int) -> None:
                store.save(
                    1, "in_progress",
                    [root / "artifacts" / "silver" / "accepted.jsonl", root / "artifacts" / "silver" / "rejected.jsonl"],
                    record_count=len(accepted), resume={"attempts": attempts, "rejected": len(rejected)}, upstream=DEPENDENCIES[1],
                )
            result = generate_silver(
                config, root, records, schemas, resume=resume_compatible, progress=progress
            )
            metadata = result["report"]
            count, current_step = len(result["accepted"]), None
        elif index == 2:
            result = compile_and_fit(config, root, _load_silver(root), schemas)
            metadata = result["circuit"].compiler_metadata
            count, current_step = len(result["circuit"].states), None
        elif index == 3:
            circuit = load_circuit(root / "artifacts" / "circuits" / "theta_star.json")
            result = project_distribution(config, root, circuit)
            metadata = result["report"]
            count, current_step = len(result["circuit"].states), result["report"]["iterations"]
        elif index == 4:
            circuit = load_circuit(root / "artifacts" / "circuits" / "theta_prime.json")
            result = sample_plans(config, root, circuit, schemas)
            metadata = result["report"]
            count, current_step = len(result["plans"]), None
        elif index == 5:
            result = train_student(config, root, _load_silver(root), resume=resume_compatible)
            metadata = result["report"]
            count, current_step = result["report"]["train_count"], result["current_step"]
        elif index == 6:
            base = load_circuit(root / "artifacts" / "circuits" / "theta_star.json")
            projected = load_circuit(root / "artifacts" / "circuits" / "theta_prime.json")
            model, tokenizer = None, None
            def generation_progress(outputs: list[dict[str, Any]]) -> None:
                store.save(
                    6, "in_progress", [root / "artifacts" / "outputs" / "generated.jsonl"],
                    record_count=len(outputs), resume={"outputs": len(outputs)}, upstream=DEPENDENCIES[6],
                )
            result = generate_outputs(
                config, root, records, schemas, _load_silver(root), base, projected, model, tokenizer,
                resume=resume_compatible, progress=generation_progress,
            )
            metadata = {"variants": list(config["evaluation"]["variants"]), "output_count": len(result["outputs"])}
            count, current_step = len(result["outputs"]), None
        else:
            outputs = read_jsonl(root / "artifacts" / "outputs" / "generated.jsonl")
            projected = load_circuit(root / "artifacts" / "circuits" / "theta_prime.json")
            result = evaluate_outputs(config, root, outputs, projected, records)
            metadata = {"variants": list(result["report"]["metrics"])}
            count, current_step = len(result["report"]["metrics"]), None
        store.save(
            index, "complete", result["artifacts"], record_count=count, current_step=current_step,
            metadata=metadata, upstream=DEPENDENCIES[index],
        )
        valid, reason = store.validate(index)
        if not valid:
            raise RuntimeError(f"final checkpoint validation failed: {reason}")
        logger.log("stage_complete", stage=STAGES[index], record_count=count, current_step=current_step)
    except KeyboardInterrupt:
        store.save(index, "interrupted", error="keyboard interrupt", upstream=DEPENDENCIES[index])
        logger.log("stage_interrupted", stage=STAGES[index])
        raise
    except Exception as error:
        store.save(index, "failed", error=str(error), upstream=DEPENDENCIES[index])
        logger.log("stage_failed", stage=STAGES[index], error=str(error))
        raise


def execute(config: dict[str, Any], root: Path, start: int = 0, stop: int = 7) -> None:
    store = CheckpointStore(root, config)
    logger = JsonlLogger(root / "artifacts" / "pipeline.jsonl")
    for index in range(start, stop + 1):
        valid, _ = store.validate(index)
        if valid:
            logger.log("stage_skip", stage=STAGES[index], reason="compatible complete checkpoint")
            continue
        run_stage(index, config, root, store, logger)


def status(config: dict[str, Any], root: Path) -> dict[str, Any]:
    store = CheckpointStore(root, config)
    rows = []
    for index, name in enumerate(STAGES):
        checkpoint = store.load(index)
        valid, reason = store.validate(index)
        rows.append({
            "stage": name, "status": checkpoint.get("status") if checkpoint else "missing",
            "compatible_complete": valid, "reason": reason,
            "record_count": checkpoint.get("record_count") if checkpoint else None,
            "current_step": checkpoint.get("current_step") if checkpoint else None,
        })
    return {"scientific_config_hash": scientific_hash(config), "stages": rows}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="CircuitSynth data pipeline")
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("run", "resume", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    stage = subparsers.add_parser("stage")
    stage.add_argument("number", type=int, choices=range(1, 9))
    stage.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    config = load_config(ROOT / "config.yaml", arguments.set)
    if arguments.command == "status":
        print(json.dumps(status(config, ROOT), indent=2))
        return 0
    save_resolved_config(config, ROOT, arguments.set)
    if arguments.command in {"run", "resume", "stage"}:
        report = preflight(config, ROOT)
        print(json.dumps(report, indent=2))
        if report["status"] != "pass":
            return 2
    store = CheckpointStore(ROOT, config)
    if arguments.command == "resume":
        start = store.first_incomplete()
        if start == len(STAGES):
            print("All stages are complete and compatible.")
            return 0
        execute(config, ROOT, start=start)
    elif arguments.command == "stage":
        execute(config, ROOT, start=arguments.number - 1, stop=arguments.number - 1)
    else:
        execute(config, ROOT)
    print(json.dumps(status(config, ROOT), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

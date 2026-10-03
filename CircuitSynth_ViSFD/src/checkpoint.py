from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGES = [
    "C0_data_schema",
    "C1_silver",
    "C2_psdd_theta_star",
    "C3_psdd_theta_prime",
    "C4_sampled_plans",
    "C5_student",
    "C6_generated_outputs",
    "C7_metrics",
]
VALID_STATUS = {"in_progress", "complete", "failed", "interrupted"}
STAGE_CONFIG = {
    0: ["project", "data"],
    1: ["project", "silver", "teacher", "data"],
    2: ["project", "psdd"],
    3: ["project", "soft_constraints", "pgd"],
    4: ["project", "sampling"],
    5: ["project", "student"],
    6: ["project", "generation", "evaluation"],
    7: ["project", "evaluation"],
}
STAGE_CODE = {
    0: ["src/data.py", "src/schema.py", "src/verifier.py", "src/checkpoint.py"],
    1: ["src/teacher.py", "src/schema.py", "src/verifier.py", "src/checkpoint.py"],
    2: ["src/circuits.py", "src/schema.py", "src/verifier.py", "src/checkpoint.py"],
    3: ["src/circuits.py", "src/checkpoint.py"],
    4: ["src/circuits.py", "src/verifier.py", "src/checkpoint.py"],
    5: ["src/student.py", "src/schema.py", "src/checkpoint.py"],
    6: ["src/fsa.py", "src/student.py", "src/verifier.py", "src/checkpoint.py"],
    7: ["src/evaluate.py", "src/verifier.py", "src/checkpoint.py"],
}
COUNT_ARTIFACT = {
    0: ("artifacts/data/records.jsonl", "jsonl"),
    1: ("artifacts/silver/accepted.jsonl", "jsonl"),
    2: ("artifacts/circuits/theta_star.json", "states"),
    3: ("artifacts/circuits/theta_prime.json", "states"),
    4: ("artifacts/plans/sampled.jsonl", "jsonl"),
    5: ("artifacts/student/train.jsonl", "jsonl"),
    6: ("artifacts/outputs/generated.jsonl", "jsonl"),
    7: ("artifacts/metrics/metrics.json", "metrics"),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"runtime", "checkpoint"}}


def scientific_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(scientific_config(config)).encode())


def stage_config_hash(config: dict[str, Any], stage_index: int) -> str:
    view = json.loads(canonical_json({key: config[key] for key in STAGE_CONFIG[stage_index] if key in config}))
    if "silver" in view:
        view["silver"].pop("checkpoint_every_accepted", None)
    if "pgd" in view:
        view["pgd"].pop("checkpoint_every_iterations", None)
    if "sampling" in view:
        view["sampling"].pop("checkpoint_every_plans", None)
    if "student" in view:
        view["student"].pop("checkpoint", None)
    if "generation" in view:
        view["generation"].pop("checkpoint_every_outputs", None)
    return sha256_bytes(canonical_json(view).encode())


def stage_code_hash(root: Path, stage_index: int) -> str:
    values = []
    for relative in STAGE_CODE[stage_index]:
        path = root / relative
        values.append({"path": relative, "sha256": sha256_file(path) if path.exists() else None})
    values.append({"path": "run.py", "sha256": sha256_file(root / "run.py") if (root / "run.py").exists() else None})
    return sha256_bytes(canonical_json(values).encode())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode()
    atomic_write_bytes(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def code_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        files = sorted(root.glob("src/*.py")) + [root / "run.py", root / "config.yaml"]
        joined = "".join(sha256_file(path) for path in files if path.exists())
        return sha256_bytes(joined.encode())


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(row) + "\n")


class CheckpointStore:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.directory = root / "artifacts" / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config

    def path(self, index: int) -> Path:
        return self.directory / f"{STAGES[index]}.json"

    def load(self, index: int) -> dict[str, Any] | None:
        path = self.path(index)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(
        self,
        index: int,
        status: str,
        artifacts: Iterable[Path] = (),
        *,
        record_count: int = 0,
        current_step: int | None = None,
        error: str | None = None,
        resume: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        upstream: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid checkpoint status: {status}")
        artifact_rows = []
        for artifact in artifacts:
            if artifact.exists():
                artifact_rows.append({
                    "path": str(artifact.relative_to(self.root)),
                    "sha256": sha256_file(artifact),
                    "size": artifact.stat().st_size,
                })
        upstream_ids = list(upstream if upstream is not None else ([index - 1] if index else []))
        upstream_hashes = {}
        for upstream_index in upstream_ids:
            checkpoint = self.load(upstream_index)
            if checkpoint:
                upstream_hashes[STAGES[upstream_index]] = checkpoint["checkpoint_sha256"]
        revisions_path = self.root / "artifacts" / "data" / "dataset_revisions.json"
        try:
            dataset_revisions = json.loads(revisions_path.read_text(encoding="utf-8")) if revisions_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            dataset_revisions = {}
        schema_path = self.root / "artifacts" / "data" / "schemas.jsonl"
        try:
            schema_versions = sorted({row["schema"]["version"] for row in read_jsonl(schema_path)}) if schema_path.exists() else []
        except (OSError, KeyError, json.JSONDecodeError):
            schema_versions = []
        model_revisions = {
            role: {
                "model": self.config.get(role, {}).get("model"),
                "revision": self.config.get(role, {}).get("revision"),
            }
            for role in ("teacher", "student")
        }
        row: dict[str, Any] = {
            "stage": STAGES[index],
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scientific_config_hash": scientific_hash(self.config),
            "stage_config_hash": stage_config_hash(self.config, index),
            "stage_code_hash": stage_code_hash(self.root, index),
            "upstream_checkpoint_hashes": upstream_hashes,
            "code_revision": code_revision(self.root),
            "dataset_revisions": dataset_revisions,
            "model_revisions": model_revisions,
            "schema_versions": schema_versions,
            "record_count": record_count,
            "current_step": current_step,
            "artifacts": artifact_rows,
            "error": error,
            "resume": resume or {},
            "metadata": metadata or {},
        }
        row["checkpoint_sha256"] = sha256_bytes(canonical_json(row).encode())
        write_json(self.path(index), row)
        return row

    def validate(self, index: int, require_complete: bool = True) -> tuple[bool, str]:
        row = self.load(index)
        if not row:
            return False, "missing"
        stored_hash = row.get("checkpoint_sha256")
        check = dict(row)
        check.pop("checkpoint_sha256", None)
        if stored_hash != sha256_bytes(canonical_json(check).encode()):
            return False, "checkpoint checksum mismatch"
        if require_complete and row.get("status") != "complete":
            return False, f"status={row.get('status')}"
        if row.get("stage_config_hash") != stage_config_hash(self.config, index):
            return False, "stale scientific configuration"
        if row.get("stage_code_hash") != stage_code_hash(self.root, index):
            return False, "stale stage implementation"
        for item in row.get("artifacts", []):
            path = self.root / item["path"]
            if not path.exists() or sha256_file(path) != item["sha256"]:
                return False, f"invalid artifact: {item['path']}"
        if require_complete:
            relative, kind = COUNT_ARTIFACT[index]
            target = self.root / relative
            if not target.exists():
                return False, f"missing count artifact: {relative}"
            if kind == "jsonl":
                observed = len(read_jsonl(target))
            else:
                payload = json.loads(target.read_text(encoding="utf-8"))
                observed = len(payload[kind])
            if observed != int(row.get("record_count", -1)):
                return False, f"record count mismatch: {observed} != {row.get('record_count')}"
        for name, expected in row.get("upstream_checkpoint_hashes", {}).items():
            upstream = self.load(STAGES.index(name))
            if not upstream or upstream.get("checkpoint_sha256") != expected:
                return False, f"stale upstream: {name}"
        return True, "complete" if require_complete else row.get("status", "unknown")

    def first_incomplete(self) -> int:
        for index in range(len(STAGES)):
            valid, _ = self.validate(index)
            if not valid:
                return index
        return len(STAGES)

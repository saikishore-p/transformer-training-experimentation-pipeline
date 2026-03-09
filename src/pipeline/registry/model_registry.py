"""
Model Registry — tracks every completed run and maintains the best model.

Backed by a single `registry.json` file so it persists across runs and
is human-readable without needing a running server.

Format of registry.json:
{
  "best_run_id": "abc123",
  "entries": [
    {
      "run_id": "abc123",
      "experiment_name": "baseline",
      "model_name": "distilbert-base-uncased",
      "checkpoint_path": "checkpoints/baseline/best",
      "test_accuracy": 0.92,
      "val_accuracy": 0.91,
      "config_snapshot": { ... },
      "registered_at": "2026-04-01T12:00:00"
    },
    ...
  ]
}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path


@dataclass
class RegistryEntry:
    """Immutable record of one completed training + evaluation run."""

    run_id: str  # MLflow run ID
    experiment_name: str
    model_name: str  # HuggingFace model ID or "custom"
    checkpoint_path: str  # Path to saved model weights
    test_accuracy: float
    val_accuracy: float
    config_snapshot: dict  # serialised PipelineConfig.to_flat_dict()
    registered_at: str  # ISO-8601 timestamp (UTC)
    # Optional rich fields populated when available
    test_f1: float = 0.0
    total_training_time_seconds: float = 0.0
    num_parameters: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RegistryEntry:
        # Pop unknown keys gracefully (forward compatibility)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class ModelRegistry:
    """
    JSON-backed registry of all completed pipeline runs.

    Thread safety: not thread-safe — each Ray worker process maintains
    its own registry file in a separate directory (keyed by experiment name).
    The sweep runner merges entries after all workers finish.

    Args:
        registry_path: Path to the registry.json file.
                       Parent directory is created if it doesn't exist.
    """

    def __init__(self, registry_path: str = "checkpoints/registry.json") -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()
        # Always write to disk on init so the file exists after construction
        if not self._path.exists():
            self._save()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        run_id: str,
        experiment_name: str,
        model_name: str,
        checkpoint_path: str,
        test_accuracy: float,
        val_accuracy: float,
        config_snapshot: dict,
        test_f1: float = 0.0,
        total_training_time_seconds: float = 0.0,
        num_parameters: int = 0,
    ) -> RegistryEntry:
        """
        Add a completed run to the registry and persist to disk.

        Updates best_run_id if this run has the highest test_accuracy so far.
        Returns the created RegistryEntry.
        """
        entry = RegistryEntry(
            run_id=run_id,
            experiment_name=experiment_name,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            test_accuracy=test_accuracy,
            val_accuracy=val_accuracy,
            config_snapshot=config_snapshot,
            registered_at=datetime.now(UTC).isoformat(),
            test_f1=test_f1,
            total_training_time_seconds=total_training_time_seconds,
            num_parameters=num_parameters,
        )

        # Remove any prior entry with the same run_id (re-registration)
        self._data["entries"] = [e for e in self._data["entries"] if e["run_id"] != run_id]
        self._data["entries"].append(entry.to_dict())

        # Update best if this run beats current best
        best = self.get_best_model()
        if best is None or test_accuracy > best.test_accuracy:
            self._data["best_run_id"] = run_id

        self._save()
        return entry

    def get_best_model(self) -> RegistryEntry | None:
        """Return the entry with the highest test_accuracy, or None if empty."""
        best_id = self._data.get("best_run_id")
        if not best_id:
            return None
        for entry_dict in self._data["entries"]:
            if entry_dict["run_id"] == best_id:
                return RegistryEntry.from_dict(entry_dict)
        # best_run_id stale (entry was removed) — recompute
        return self._recompute_best()

    def get_all_entries(self) -> list[RegistryEntry]:
        """Return all registered entries, sorted best-first by test_accuracy."""
        entries = [RegistryEntry.from_dict(e) for e in self._data["entries"]]
        return sorted(entries, key=lambda e: e.test_accuracy, reverse=True)

    def get_entry(self, run_id: str) -> RegistryEntry | None:
        """Look up a specific entry by MLflow run ID."""
        for entry_dict in self._data["entries"]:
            if entry_dict["run_id"] == run_id:
                return RegistryEntry.from_dict(entry_dict)
        return None

    def remove(self, run_id: str) -> bool:
        """Remove an entry by run_id. Returns True if found and removed."""
        before = len(self._data["entries"])
        self._data["entries"] = [e for e in self._data["entries"] if e["run_id"] != run_id]
        removed = len(self._data["entries"]) < before
        if removed:
            if self._data.get("best_run_id") == run_id:
                best = self._recompute_best()
                self._data["best_run_id"] = best.run_id if best else None
            self._save()
        return removed

    def __len__(self) -> int:
        return len(self._data["entries"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return {"best_run_id": None, "entries": []}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def _recompute_best(self) -> RegistryEntry | None:
        entries = self.get_all_entries()
        if not entries:
            self._data["best_run_id"] = None
            self._save()
            return None
        best = entries[0]  # sorted best-first
        self._data["best_run_id"] = best.run_id
        self._save()
        return best

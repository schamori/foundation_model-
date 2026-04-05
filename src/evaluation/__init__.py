"""
Evaluation framework for foundation model pretraining.

All evaluators accept a Config and operate on its checkpoint_dir / eval_dir.
Designed for use both during training (after each epoch) and standalone.

Usage (standalone):
    python -m src.evaluation --config Full --checkpoint output/full/checkpoints/dino_epoch50.pt

Usage (from training):
    from src.evaluation import run_all_evaluations
    metrics = run_all_evaluations(cfg, checkpoint_path, epoch)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


class BaseEvaluator(ABC):
    """Interface for all evaluation methods."""

    name: str = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.results: list[dict] = []

    @abstractmethod
    def evaluate_checkpoint(self, checkpoint_path: Path, epoch: int) -> dict:
        """Run evaluation on one checkpoint. Returns metrics dict."""
        ...

    def summarize(self) -> dict:
        """Aggregate results across all evaluated checkpoints."""
        return {"name": self.name, "n_evaluations": len(self.results), "results": self.results}


# Registry of evaluators
_EVALUATORS: list[type[BaseEvaluator]] = []


def register_evaluator(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
    """Decorator to register an evaluator class."""
    _EVALUATORS.append(cls)
    return cls


def run_all_evaluations(cfg: Config, checkpoint_path: Path, epoch: int) -> dict:
    """
    Run all registered evaluators on a checkpoint.
    Saves results to cfg.eval_dir/epoch_{epoch}.json.
    """
    all_metrics: dict[str, dict] = {}

    for evaluator_cls in _EVALUATORS:
        try:
            evaluator = evaluator_cls(cfg)
            metrics = evaluator.evaluate_checkpoint(checkpoint_path, epoch)
            all_metrics[evaluator.name] = metrics
            print(f"[eval] {evaluator.name}: {metrics}")
        except Exception as e:
            print(f"[eval] {evaluator.name} failed: {e}")
            all_metrics[evaluator.name] = {"error": str(e)}

    # Save per-epoch results
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.eval_dir / f"epoch_{epoch:04d}.json"
    with open(out_path, "w") as f:
        json.dump({"epoch": epoch, "checkpoint": str(checkpoint_path), "metrics": all_metrics}, f, indent=2)

    # Append to summary
    summary_path = cfg.eval_dir / "summary.json"
    summary: list[dict] = []
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    summary.append({"epoch": epoch, "metrics": all_metrics})
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_metrics


# Import evaluators to trigger registration
from . import similarity  # noqa: E402, F401
from . import cross_video_retrieval  # noqa: E402, F401
from . import knn  # noqa: E402, F401

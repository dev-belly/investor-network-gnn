"""Common utilities: config loading, logging, paths, I/O."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# --- OpenMP / thread tuning ------------------------------------------------
# macOS + XGBoost (and some other libs) hit pthread_mutex_init errors when
# OpenMP tries to spawn many workers in restricted sandboxes.  Pinning to a
# single thread sidesteps the issue while remaining fast enough for the
# medium-sized problem we run here.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once."""
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Set seed for Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config. Defaults to configs/config.yaml under project root."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Paths:
    """Resolved output paths."""

    root: Path
    data_raw: Path
    data_processed: Path
    figures: Path
    reports: Path
    models: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> "Paths":
        if root is None:
            root = PROJECT_ROOT
        root = Path(root)
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            figures=root / "outputs" / "figures",
            reports=root / "outputs" / "reports",
            models=root / "outputs" / "models",
        )

    def ensure(self) -> "Paths":
        for p in (
            self.data_raw,
            self.data_processed,
            self.figures,
            self.reports,
            self.models,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

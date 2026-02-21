"""
experiment_builder.py

Generic experiment plan builder for time-dependent guidance sweeps.
Creates a Cartesian-product grid of experiment configs and writes it to CSV.

Designed to work with guidance_registry.SCHEDULER_PROPERTIES (shape taxonomy).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from guidance_registry import SCHEDULER_PROPERTIES


Direction = Literal["increasing", "decreasing"]

PromptSpec = Union[
    Mapping[str, str],                  # {"P1": "text", "P2": "text"}
    Sequence[Tuple[str, str]],          # [("P1","text"), ...]
    Sequence[str],                      # ["text", ...] -> auto ids
]


def _normalize_prompts(prompts: PromptSpec) -> Dict[str, str]:
    if isinstance(prompts, dict):
        return dict(prompts)
    # list/tuple of (id, text)
    if len(prompts) > 0 and isinstance(prompts[0], (tuple, list)) and len(prompts[0]) == 2:
        return {str(k): str(v) for k, v in prompts}  # type: ignore[arg-type]
    # list of text
    return {f"P{i:03d}": str(t) for i, t in enumerate(prompts, start=1)}  # type: ignore[arg-type]


def _stable_hash(row: Dict[str, Any], ignore: Iterable[str] = ("prompt_text", "experiment_group")) -> str:
    if row['kind'] == 'baseline':
        ignore += ('w_min', 'w_max')
    if row['kind'] in ['constant', 'average_constant', 'baseline']:
        ignore += ('direction',)
    clean = {k: row[k] for k in row.keys() if k not in set(ignore)}
    payload = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


@dataclass
class ExperimentSpec:
    """High-level specification for an experiment grid."""
    experiment_group: str
    kinds: Sequence[str]
    directions: Sequence[Direction]
    weight_ranges: Sequence[Tuple[float, float]]
    prompts: PromptSpec
    seeds: Sequence[int]
    num_steps: Sequence[int] = (25,)
    # Optional sweep over scheduler params (shape-specific kwargs)
    # e.g. [{"alpha": 2.0}, {"alpha": 4.0}] for exponential/logarithmic
    params_list: Sequence[Dict[str, Any]] = field(default_factory=lambda: ({},))
    # Arbitrary metadata columns to stamp on every row
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Worker assignment (useful for distributed runs)
    num_workers: Optional[int] = None

    # Redundancy pruning
    prune_redundant: bool = True


def build_experiment_grid(spec: ExperimentSpec) -> pd.DataFrame:
    """Return a DataFrame containing the Cartesian product of the spec."""
    prompts = _normalize_prompts(spec.prompts)

    rows: List[Dict[str, Any]] = []

    for kind, direction, w_range, (prompt_id, prompt_text), seed, steps, params in itertools.product(
        spec.kinds,
        spec.directions,
        spec.weight_ranges,
        prompts.items(),
        spec.seeds,
        spec.num_steps,
        spec.params_list if spec.params_list else ({},),
    ):
        props = SCHEDULER_PROPERTIES.get(kind, {"type": "monotonic"})
        category = props.get("type", "monotonic")

        if spec.prune_redundant:
            # Static schedules ignore direction; keep one canonical direction.
            if category == "static" and direction != "increasing":
                continue

        row: Dict[str, Any] = {
            "experiment_group": spec.experiment_group,
            "kind": kind,
            "direction": direction,
            "w_min": float(w_range[0]),
            "w_max": float(w_range[1]),
            "prompt_id": str(prompt_id),
            "prompt_text": str(prompt_text),
            "seed": int(seed),
            "num_steps": int(steps),
            "params": json.dumps(params or {}, sort_keys=True),
        }

        # Stamp additional metadata
        for k, v in (spec.metadata or {}).items():
            # avoid collisions with core columns
            if k in row:
                row[f"meta_{k}"] = v
            else:
                row[k] = v

        row["experiment_id"] = _stable_hash(row)
        rows.append(row)

    df = pd.DataFrame(rows)

    if spec.num_workers and spec.num_workers > 0:
        df["worker_id"] = df.index % int(spec.num_workers)

    # Nice-to-have: stable column order
    preferred = [
        "experiment_group", "experiment_id",
        "kind", "direction", "w_min", "w_max",
        "num_steps", "seed",
        "prompt_id", "prompt_text",
        "params",
    ]
    # keep any extra metadata columns at the end
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def write_experiment_csv(df: pd.DataFrame, path: str) -> str:
    """Write the plan to CSV and return the path."""
    df_deduplicated = df.drop_duplicates(subset=["experiment_id"], keep="first")
    df_deduplicated.to_csv(path, index=False)
    return path


def build_and_write(spec: ExperimentSpec, csv_path: str) -> pd.DataFrame:
    """Convenience helper: build grid and write to disk."""
    df = build_experiment_grid(spec)
    write_experiment_csv(df, csv_path)
    return df

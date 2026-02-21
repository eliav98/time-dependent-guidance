# viz_grids.py
# Modular visualization for diffusion experiment outputs:
# - consumes results.jsonl + images only
# - optional: reconcile coverage vs plan CSV
#
# Dependencies: pandas, matplotlib, pillow (PIL)
# Optional: ipywidgets (for explorer)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import math

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# Loading
# -----------------------------

from pathlib import Path
import pandas as pd
import json

def build_results_df_from_plan_and_images(
    plan_csv: Path,
    artifacts_dir: Path,
    jsonl_path: Path | None = None,
) -> pd.DataFrame:
    """
    Create a complete results-like dataframe (144 rows) using:
      - plan CSV for parameters
      - images on disk for existence/img_path
      - optional JSONL to enrich fields (runtime_s, notes, etc) where present
    """
    plan = pd.read_csv(plan_csv)

    # add img_path based on experiment_id (filename convention)
    plan["img_path"] = plan["experiment_id"].astype(str).apply(
        lambda eid: str((artifacts_dir / f"{eid}.png").resolve())
    )
    plan["img_exists"] = plan["img_path"].apply(lambda p: Path(p).exists())

    # derived columns
    if "kind" in plan.columns and "direction" in plan.columns:
        plan["schedule"] = plan["kind"].astype(str) + ":" + plan["direction"].astype(str)
    if "w_min" in plan.columns and "w_max" in plan.columns:
        plan["w_min"] = pd.to_numeric(plan["w_min"], errors="coerce")
        plan["w_max"] = pd.to_numeric(plan["w_max"], errors="coerce")
        plan["w_range"] = plan.apply(lambda r: f"({r['w_min']},{r['w_max']})", axis=1)

    # optional: enrich from jsonl by experiment_id
    if jsonl_path is not None and Path(jsonl_path).exists():
        recs = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        jdf = pd.DataFrame(recs)
        if "experiment_id" in jdf.columns:
            plan = plan.merge(jdf, on="experiment_id", how="left", suffixes=("", "_json"))

            # if JSON has img_path, prefer it when present
            if "img_path_json" in plan.columns:
                plan["img_path"] = plan["img_path_json"].fillna(plan["img_path"])

    return plan

def load_results_jsonl(
    jsonl_path: Union[str, Path],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    resolve_img_paths: bool = True,
) -> pd.DataFrame:
    """
    Load runner outputs from results.jsonl into a DataFrame.

    - Adds derived columns:
        schedule = kind + ":" + direction
        w_range  = "(w_min,w_max)" string (stable, hashable, easy for widgets)
        w_min, w_max coerced to float when possible

    - If resolve_img_paths=True:
        - If img_path is relative, it is resolved relative to base_dir
        - base_dir defaults to the jsonl file's parent directory
    """
    jsonl_path = Path(jsonl_path)
    if base_dir is None:
        base_dir = jsonl_path.parent
    base_dir = Path(base_dir)

    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {jsonl_path}: {e}") from e
            records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Coerce numeric types if present
    for col in ("w_min", "w_max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived: schedule
    if "kind" in df.columns and "direction" in df.columns:
        df["schedule"] = df["kind"].astype(str) + ":" + df["direction"].astype(str)
    else:
        df["schedule"] = None

    # Derived: w_range as stable string (widgets-friendly & grouping-friendly)
    if "w_min" in df.columns and "w_max" in df.columns:
        df["w_range"] = df.apply(lambda r: f"({r['w_min']},{r['w_max']})", axis=1)
    else:
        df["w_range"] = None

    # Normalize img_path
    if "img_path" in df.columns and resolve_img_paths:
        def _resolve(p: Any) -> str:
            if p is None or (isinstance(p, float) and math.isnan(p)):  # NaN
                return ""
            p = str(p)
            if not p:
                return ""
            pp = Path(p)
            if pp.is_absolute():
                return str(pp)
            # resolve relative to base_dir
            return str((base_dir / pp).resolve())

        df["img_path"] = df["img_path"].apply(_resolve)

    return df


def load_plan_csv(plan_csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the experiment plan CSV (cartesian grid).
    This is optional and only used for coverage/missing analysis.
    """
    plan_csv_path = Path(plan_csv_path)
    df = pd.read_csv(plan_csv_path)

    # Make sure common keys exist and match types where possible
    for col in ("w_min", "w_max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "kind" in df.columns and "direction" in df.columns and "schedule" not in df.columns:
        df["schedule"] = df["kind"].astype(str) + ":" + df["direction"].astype(str)

    if "w_min" in df.columns and "w_max" in df.columns and "w_range" not in df.columns:
        df["w_range"] = df.apply(lambda r: f"({r['w_min']},{r['w_max']})", axis=1)

    return df


def merge_plan_with_results(
    plan_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    keys: Sequence[str] = ("experiment_group", "prompt_id", "seed", "kind", "direction", "w_min", "w_max", "num_steps"),
) -> pd.DataFrame:
    """
    Left-join plan onto results to identify missing runs.

    Output columns include:
      - has_result (bool)
      - img_path/result fields where available
    """
    # Ensure keys exist in both (best-effort)
    keys = [k for k in keys if k in plan_df.columns and k in results_df.columns]
    merged = plan_df.merge(
        results_df,
        on=keys,
        how="left",
        suffixes=("_plan", ""),
        indicator=True,
    )
    merged["has_result"] = merged["_merge"].eq("both")
    return merged.drop(columns=["_merge"])


def summarize_coverage(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize planned vs completed counts by useful dimensions.
    Requires output from merge_plan_with_results (has_result column).
    """
    if "has_result" not in merged_df.columns:
        raise ValueError("summarize_coverage expects a merged df with a 'has_result' column")

    dims = [c for c in ["experiment_group", "prompt_id", "schedule", "w_range", "seed"] if c in merged_df.columns]
    if not dims:
        dims = []

    out = (
        merged_df
        .groupby(dims, dropna=False)["has_result"]
        .agg(planned="count", completed="sum")
        .reset_index()
    )
    out["missing"] = out["planned"] - out["completed"]
    return out


# -----------------------------
# Grid rendering
# -----------------------------

@dataclass
class GridStyle:
    figsize_per_cell: Tuple[float, float] = (3.0, 3.0)   # inches per cell
    dpi: int = 150
    pad_inches: float = 0.05
    title_size: int = 14
    label_size: int = 10
    caption_size: int = 9
    missing_text: str = "MISSING"
    missing_alpha: float = 0.5


def _stable_unique(values: Iterable[Any]) -> List[Any]:
    """Stable unique preserving first-seen order."""
    seen = set()
    out = []
    for v in values:
        key = v
        try:
            h = hash(key)
        except Exception:
            key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _apply_order(
    values: List[Any],
    order: Optional[Sequence[Any]] = None,
    sort: bool = False,
) -> List[Any]:
    if order is not None:
        # Keep only those present, in requested order, then append any extras (stable)
        present = set(values)
        ordered = [v for v in order if v in present]
        extras = [v for v in values if v not in set(ordered)]
        return ordered + extras
    if sort:
        try:
            return sorted(values)
        except Exception:
            return sorted(values, key=lambda x: str(x))
    return values


def _load_image_safe(path: str) -> Optional[Image.Image]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def render_grid(
    df: pd.DataFrame,
    *,
    rows: str,
    cols: str,
    where: Optional[Dict[str, Any]] = None,
    pick: Optional[str] = None,
    caption: Optional[Union[str, Sequence[str]]] = None,
    row_order: Optional[Sequence[Any]] = None,
    col_order: Optional[Sequence[Any]] = None,
    row_sort: bool = True,
    col_sort: bool = True,
    title: Optional[str] = None,
    style: GridStyle = GridStyle(),
    show_row_labels: bool = True,
    show_col_labels: bool = True,
    show_captions: bool = True,
) -> plt.Figure:
    """
    Generic grid renderer.

    - rows, cols: column names to form grid axes
    - where: filters, e.g. {"prompt_id": "...", "w_range": "(1.0,12.0)"}
    - pick: optional column to disambiguate if multiple rows land in same cell.
            If provided, we pick the max of this field; else first row.
    - caption:
        - None: no captions
        - str: name of column to show under each cell
        - list/tuple of str: columns to combine (joined by " | ")
    """
    if where:
        mask = pd.Series(True, index=df.index)
        for k, v in where.items():
            if k not in df.columns:
                # If user filters on derived columns not present, just filter nothing.
                mask &= False
                continue
            mask &= (df[k] == v)
        df = df[mask].copy()

    # Identify axis values
    if rows not in df.columns or cols not in df.columns:
        raise ValueError(f"rows='{rows}' or cols='{cols}' not present in df columns")

    row_vals = _stable_unique(df[rows].tolist())
    col_vals = _stable_unique(df[cols].tolist())

    row_vals = _apply_order(row_vals, order=row_order, sort=row_sort)
    col_vals = _apply_order(col_vals, order=col_order, sort=col_sort)

    nrows = len(row_vals)
    ncols = len(col_vals)

    if nrows == 0 or ncols == 0:
        raise ValueError("No data after filtering; grid would be empty.")

    figsize = (style.figsize_per_cell[0] * ncols, style.figsize_per_cell[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=style.dpi)

    # Make axes always 2D
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    # Index df for faster lookup
    # Group by (rows, cols)
    grouped = df.groupby([rows, cols], dropna=False)

    def cell_record(rval: Any, cval: Any) -> Optional[pd.Series]:
        if (rval, cval) not in grouped.groups:
            return None
        sub = grouped.get_group((rval, cval))
        if len(sub) == 1:
            return sub.iloc[0]
        if pick and pick in sub.columns:
            try:
                return sub.sort_values(pick, ascending=False).iloc[0]
            except Exception:
                return sub.iloc[0]
        return sub.iloc[0]

    for i, rval in enumerate(row_vals):
        for j, cval in enumerate(col_vals):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            rec = cell_record(rval, cval)
            if rec is None:
                ax.text(
                    0.5, 0.5, style.missing_text,
                    ha="center", va="center",
                    fontsize=style.label_size,
                    alpha=style.missing_alpha,
                    transform=ax.transAxes,
                )
                continue

            img = _load_image_safe(str(rec.get("img_path", "")))
            if img is None:
                ax.text(
                    0.5, 0.5, style.missing_text,
                    ha="center", va="center",
                    fontsize=style.label_size,
                    alpha=style.missing_alpha,
                    transform=ax.transAxes,
                )
            else:
                ax.imshow(img)

            # Captions
            if show_captions and caption is not None:
                if isinstance(caption, str):
                    cap = rec.get(caption, "")
                else:
                    parts = [str(rec.get(c, "")) for c in caption]
                    cap = " | ".join(parts)
                ax.set_title(str(cap), fontsize=style.caption_size, pad=2)

            # Row/col labels on edges
            if show_row_labels and j == 0:
                ax.text(
                    -0.02, 0.5, str(rval),
                    ha="right", va="center",
                    fontsize=style.label_size,
                    rotation=90,
                    transform=ax.transAxes,
                )
            if show_col_labels and i == 0:
                ax.text(
                    0.5, 1.02, str(cval),
                    ha="center", va="bottom",
                    fontsize=style.label_size,
                    transform=ax.transAxes,
                )

    if title:
        fig.suptitle(title, fontsize=style.title_size)

    fig.tight_layout(pad=style.pad_inches)
    return fig


# -----------------------------
# Preset grids (your 3 required comparisons)
# -----------------------------

def grid_shapes_same_prompt_same_range(
    df: pd.DataFrame,
    *,
    prompt_id: str,
    w_range: str,
    cols: str = "seed",
    title: Optional[str] = None,
    caption: Optional[Union[str, Sequence[str]]] = ("kind", "direction", "w_range", "seed"),
    style: GridStyle = GridStyle(),
) -> plt.Figure:
    """
    (1) Different schedule shapes: same prompt, same weight range.
    Rows: schedule
    Cols: seed (default)
    """
    where = {"prompt_id": prompt_id, "w_range": w_range}
    if title is None:
        title = f"Shapes | prompt={prompt_id} | w_range={w_range}"
    return render_grid(
        df,
        rows="schedule",
        cols=cols,
        where=where,
        caption=caption,
        title=title,
        style=style,
    )


def grid_weights_same_prompt_same_schedule(
    df: pd.DataFrame,
    *,
    prompt_id: str,
    schedule: str,
    cols: str = "seed",
    title: Optional[str] = None,
    caption: Optional[Union[str, Sequence[str]]] = ("schedule", "w_range", "seed"),
    style: GridStyle = GridStyle(),
) -> plt.Figure:
    """
    (2) Same schedule shape: same prompt, different weight ranges.
    Rows: w_range
    Cols: seed (default)
    """
    where = {"prompt_id": prompt_id, "schedule": schedule}
    if title is None:
        title = f"Weights | prompt={prompt_id} | schedule={schedule}"
    return render_grid(
        df,
        rows="w_range",
        cols=cols,
        where=where,
        caption=caption,
        title=title,
        style=style,
    )


def grid_prompts_same_schedule_same_weights(
    df: pd.DataFrame,
    *,
    schedule: str,
    w_range: str,
    cols: str = "seed",
    title: Optional[str] = None,
    caption: Optional[Union[str, Sequence[str]]] = ("prompt_id", "seed"),
    style: GridStyle = GridStyle(),
) -> plt.Figure:
    """
    (3) Same schedule shape: different prompts, same weight range.
    Rows: prompt_id
    Cols: seed (default)
    """
    where = {"schedule": schedule, "w_range": w_range}
    if title is None:
        title = f"Prompts | schedule={schedule} | w_range={w_range}"
    return render_grid(
        df,
        rows="prompt_id",
        cols=cols,
        where=where,
        caption=caption,
        title=title,
        style=style,
    )


# -----------------------------
# Interactive explorer (optional)
# -----------------------------

def interactive_explorer(df: pd.DataFrame):
    """
    Notebook-only (ipywidgets) explorer.
    Returns a widget box that renders selected grids.

    Modes:
      - shapes: different schedules, same prompt & w_range
      - weights: different w_range, same prompt & schedule
      - prompts: different prompts, same schedule & w_range
      - custom: generic render_grid with chosen rows/cols and filters
    """
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
    except Exception as e:
        raise ImportError("ipywidgets + IPython are required for interactive_explorer()") from e

    # Helper: safe uniques
    def uniq(col: str) -> List[Any]:
        if col not in df.columns:
            return []
        vals = [v for v in df[col].dropna().tolist()]
        return _apply_order(_stable_unique(vals), sort=True)

    modes = ["shapes", "weights", "prompts", "custom"]

    mode_dd = W.Dropdown(options=modes, value="shapes", description="mode:")

    prompt_dd = W.Dropdown(options=uniq("prompt_id"), description="prompt:")
    schedule_dd = W.Dropdown(options=uniq("schedule"), description="schedule:")
    wrange_dd = W.Dropdown(options=uniq("w_range"), description="w_range:")

    rows_dd = W.Dropdown(options=sorted(df.columns), value="schedule", description="rows:")
    cols_dd = W.Dropdown(options=sorted(df.columns), value="seed", description="cols:")

    # Common filters for custom
    filter_key_dd = W.Dropdown(options=["(none)"] + sorted(df.columns), value="(none)", description="filter key:")
    filter_val_dd = W.Dropdown(options=[""], value="", description="filter val:")

    out = W.Output()

    def update_filter_vals(*_):
        key = filter_key_dd.value
        if key == "(none)" or key not in df.columns:
            filter_val_dd.options = [""]
            filter_val_dd.value = ""
            return
        vals = df[key].dropna().astype(str).unique().tolist()
        vals = sorted(vals)
        filter_val_dd.options = [""] + vals
        filter_val_dd.value = ""

    filter_key_dd.observe(update_filter_vals, names="value")
    update_filter_vals()

    def render(*_):
        with out:
            clear_output(wait=True)
            mode = mode_dd.value

            style = GridStyle()

            if mode == "shapes":
                if not prompt_dd.value or not wrange_dd.value:
                    print("Select prompt and w_range")
                    return
                fig = grid_shapes_same_prompt_same_range(
                    df,
                    prompt_id=str(prompt_dd.value),
                    w_range=str(wrange_dd.value),
                    style=style,
                )
                plt.show()

            elif mode == "weights":
                if not prompt_dd.value or not schedule_dd.value:
                    print("Select prompt and schedule")
                    return
                fig = grid_weights_same_prompt_same_schedule(
                    df,
                    prompt_id=str(prompt_dd.value),
                    schedule=str(schedule_dd.value),
                    style=style,
                )
                plt.show()

            elif mode == "prompts":
                if not schedule_dd.value or not wrange_dd.value:
                    print("Select schedule and w_range")
                    return
                fig = grid_prompts_same_schedule_same_weights(
                    df,
                    schedule=str(schedule_dd.value),
                    w_range=str(wrange_dd.value),
                    style=style,
                )
                plt.show()

            elif mode == "custom":
                where = {}
                k = filter_key_dd.value
                v = filter_val_dd.value
                if k != "(none)" and v != "":
                    # match stringified value; robust for widgets
                    where[k] = _coerce_value_to_dtype(df, k, v)

                fig = render_grid(
                    df,
                    rows=rows_dd.value,
                    cols=cols_dd.value,
                    where=where or None,
                    caption=("experiment_id",),
                    title="Custom grid",
                    style=style,
                )
                plt.show()

    def _coerce_value_to_dtype(df_: pd.DataFrame, col: str, v: str):
        # try to coerce to numeric if the column looks numeric
        if pd.api.types.is_numeric_dtype(df_[col].dtype):
            try:
                return float(v)
            except Exception:
                return v
        return v

    # observe
    for w in [mode_dd, prompt_dd, schedule_dd, wrange_dd, rows_dd, cols_dd, filter_key_dd, filter_val_dd]:
        w.observe(render, names="value")

    controls_common = W.VBox([mode_dd])
    controls_shapes = W.VBox([prompt_dd, wrange_dd])
    controls_weights = W.VBox([prompt_dd, schedule_dd])
    controls_prompts = W.VBox([schedule_dd, wrange_dd])
    controls_custom = W.VBox([rows_dd, cols_dd, filter_key_dd, filter_val_dd])

    def controls_for_mode(m: str):
        if m == "shapes":
            return controls_shapes
        if m == "weights":
            return controls_weights
        if m == "prompts":
            return controls_prompts
        return controls_custom

    controls_box = W.VBox([controls_common, controls_for_mode(mode_dd.value)])

    def mode_controls_update(*_):
        controls_box.children = [controls_common, controls_for_mode(mode_dd.value)]
        render()

    mode_dd.observe(mode_controls_update, names="value")
    mode_controls_update()

    ui = W.HBox([controls_box, out])
    return ui
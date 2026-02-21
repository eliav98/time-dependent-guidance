# viz_grids.py
# Modular visualization for diffusion experiment outputs:
# - consumes results.jsonl + images only
# - optional: reconcile coverage vs plan CSV
#
# Dependencies: pandas, matplotlib, pillow (PIL)
# Optional: ipywidgets (for explorer)

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------
# Robust pathlib paths (relative to notebook file)
# -------------------------------------------------
from pathlib import Path

# Current working directory (where notebook runs)
HERE = Path.cwd()

# Parent directory
EXPERIMENTS_DIR = HERE.parent

# Artifacts directory
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts" / "EXP_B_EXTREME_TRADEOFF"

RESULTS_JSONL = ARTIFACTS_DIR / "results.jsonl"
PLAN_CSV = ARTIFACTS_DIR / "EXP_B_EXTREME_TRADEOFF_plan.csv"
# -----------------------------
# Loading
# -----------------------------

def build_results_df_from_plan_and_images(
        plan_csv: Path,
        artifacts_dir: Path,
        jsonl_path: Path | None = None,
) -> pd.DataFrame:
    """
    Create a complete results-like dataframe using plan CSV and images on disk.
    """
    plan = pd.read_csv(plan_csv)

    plan["img_path"] = plan["experiment_id"].astype(str).apply(
        lambda eid: str((artifacts_dir / f"{eid}.png").resolve())
    )
    plan["img_exists"] = plan["img_path"].apply(lambda p: Path(p).exists())

    if "kind" in plan.columns and "direction" in plan.columns:
        plan["schedule"] = plan["kind"].astype(str) + ":" + plan["direction"].astype(str)
    if "w_min" in plan.columns and "w_max" in plan.columns:
        plan["w_min"] = pd.to_numeric(plan["w_min"], errors="coerce")
        plan["w_max"] = pd.to_numeric(plan["w_max"], errors="coerce")
        plan["w_range"] = plan.apply(lambda r: f"({r['w_min']},{r['w_max']})", axis=1)

    if jsonl_path is not None and Path(jsonl_path).exists():
        recs = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        jdf = pd.DataFrame(recs)
        if "experiment_id" in jdf.columns:
            plan = plan.merge(jdf, on="experiment_id", how="left", suffixes=("", "_json"))
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
    Load runner outputs from results.jsonl into a DataFrame with derived columns.
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

    for col in ("w_min", "w_max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "kind" in df.columns and "direction" in df.columns:
        df["schedule"] = df["kind"].astype(str) + ":" + df["direction"].astype(str)
    else:
        df["schedule"] = None

    if "w_min" in df.columns and "w_max" in df.columns:
        df["w_range"] = df.apply(lambda r: f"({r['w_min']},{r['w_max']})", axis=1)
    else:
        df["w_range"] = None

    if "img_path" in df.columns and resolve_img_paths:
        def _resolve(p: Any) -> str:
            if p is None or (isinstance(p, float) and math.isnan(p)):
                return ""
            p = str(p)
            if not p:
                return ""
            pp = Path(p)
            if pp.is_absolute():
                return str(pp)
            return str((base_dir / pp).resolve())

        df["img_path"] = df["img_path"].apply(_resolve)

    return df


def load_plan_csv(plan_csv_path: Union[str, Path]) -> pd.DataFrame:
    plan_csv_path = Path(plan_csv_path)
    df = pd.read_csv(plan_csv_path)

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
        keys: Sequence[str] = ("experiment_group", "prompt_id", "seed", "kind", "direction", "w_min", "w_max",
                               "num_steps"),
) -> pd.DataFrame:
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
    figsize_per_cell: Tuple[float, float] = (3.0, 3.0)
    dpi: int = 150
    pad_inches: float = 0.05
    title_size: int = 14
    label_size: int = 10
    caption_size: int = 9
    missing_text: str = "MISSING"
    missing_alpha: float = 0.5


def _stable_unique(values: Iterable[Any]) -> List[Any]:
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
        pick: Optional[Union[str, Tuple[str, Any]]] = "first",
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
    Generic grid renderer with dynamic axes and robust multi-match resolution.

    Args:
        rows, cols: Columns to form grid axes.
        where: Filters to apply.
        pick: Strategy for resolving multiple rows mapped to a single cell.
            - "first": Picks the first encountered row (warns if duplicates exist).
            - "last": Picks the last encountered row.
            - "<col_name>": Picks the row with the maximum value in <col_name> (e.g., "runtime_s").
            - ("<col_name>", <value>): Picks the exact row where col matches value (e.g., ("seed", 42)).
    """
    if where:
        mask = pd.Series(True, index=df.index)
        for k, v in where.items():
            if k not in df.columns:
                mask &= False
                continue
            mask &= (df[k] == v)
        df = df[mask].copy()

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

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    grouped = df.groupby([rows, cols], dropna=False)

    def cell_record(rval: Any, cval: Any) -> Optional[pd.Series]:
        if (rval, cval) not in grouped.groups:
            return None
        sub = grouped.get_group((rval, cval))
        if len(sub) == 1:
            return sub.iloc[0]

        # Multi-match handling
        warn_prefix = f"Multi-match at ({rows}={rval}, {cols}={cval})."

        if pick == "first" or pick is None:
            warnings.warn(f"{warn_prefix} Picking 'first'. Specify a 'pick' strategy to suppress.")
            return sub.iloc[0]
        elif pick == "last":
            return sub.iloc[-1]
        elif isinstance(pick, tuple) and len(pick) == 2:
            k, v = pick
            if k in sub.columns:
                matched = sub[sub[k] == v]
                if not matched.empty:
                    return matched.iloc[0]
            warnings.warn(f"{warn_prefix} Exact match {k}={v} not found. Falling back to 'first'.")
            return sub.iloc[0]
        elif isinstance(pick, str):
            if pick in sub.columns:
                try:
                    return sub.sort_values(pick, ascending=False).iloc[0]
                except Exception:
                    pass
            warnings.warn(f"{warn_prefix} Maximize col '{pick}' failed/missing. Falling back to 'first'.")
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
                    0.5, 0.5, style.missing_text, ha="center", va="center",
                    fontsize=style.label_size, alpha=style.missing_alpha, transform=ax.transAxes
                )
                continue

            img = _load_image_safe(str(rec.get("img_path", "")))
            if img is None:
                ax.text(
                    0.5, 0.5, style.missing_text, ha="center", va="center",
                    fontsize=style.label_size, alpha=style.missing_alpha, transform=ax.transAxes
                )
            else:
                ax.imshow(img)

            if show_captions and caption is not None:
                if isinstance(caption, str):
                    cap = rec.get(caption, "")
                else:
                    parts = [str(rec.get(c, "")) for c in caption]
                    cap = " | ".join(parts)
                ax.set_title(str(cap), fontsize=style.caption_size, pad=2)

            if show_row_labels and j == 0:
                ax.text(
                    -0.02, 0.5, str(rval), ha="right", va="center",
                    fontsize=style.label_size, rotation=90, transform=ax.transAxes
                )
            if show_col_labels and i == 0:
                ax.text(
                    0.5, 1.02, str(cval), ha="center", va="bottom",
                    fontsize=style.label_size, transform=ax.transAxes
                )

    if title:
        fig.suptitle(title, fontsize=style.title_size)

    fig.tight_layout(pad=style.pad_inches)
    return fig


# -----------------------------
# Preset grids
# -----------------------------

def grid_shapes_same_prompt_same_range(
        df: pd.DataFrame,
        *,
        prompt_id: str,
        w_range: str,
        rows: str = "schedule",
        cols: str = "seed",
        pick: Optional[Union[str, Tuple[str, Any]]] = "first",
        title: Optional[str] = None,
        caption: Optional[Union[str, Sequence[str]]] = ("kind", "direction", "w_range", "seed"),
        style: GridStyle = GridStyle(),
) -> plt.Figure:
    where = {"prompt_id": prompt_id, "w_range": w_range}
    if title is None:
        title = f"Shapes | prompt={prompt_id} | w_range={w_range}"
    return render_grid(
        df, rows=rows, cols=cols, where=where, pick=pick, caption=caption, title=title, style=style
    )


def grid_weights_same_prompt_same_schedule(
        df: pd.DataFrame,
        *,
        prompt_id: str,
        schedule: str,
        rows: str = "w_range",
        cols: str = "seed",
        pick: Optional[Union[str, Tuple[str, Any]]] = "first",
        title: Optional[str] = None,
        caption: Optional[Union[str, Sequence[str]]] = ("schedule", "w_range", "seed"),
        style: GridStyle = GridStyle(),
) -> plt.Figure:
    where = {"prompt_id": prompt_id, "schedule": schedule}
    if title is None:
        title = f"Weights | prompt={prompt_id} | schedule={schedule}"
    return render_grid(
        df, rows=rows, cols=cols, where=where, pick=pick, caption=caption, title=title, style=style
    )


def grid_prompts_same_schedule_same_weights(
        df: pd.DataFrame,
        *,
        schedule: str,
        w_range: str,
        rows: str = "prompt_id",
        cols: str = "seed",
        pick: Optional[Union[str, Tuple[str, Any]]] = "first",
        title: Optional[str] = None,
        caption: Optional[Union[str, Sequence[str]]] = ("prompt_id", "seed"),
        style: GridStyle = GridStyle(),
) -> plt.Figure:
    where = {"schedule": schedule, "w_range": w_range}
    if title is None:
        title = f"Prompts | schedule={schedule} | w_range={w_range}"
    return render_grid(
        df, rows=rows, cols=cols, where=where, pick=pick, caption=caption, title=title, style=style
    )


# -----------------------------
# Interactive explorer
# -----------------------------

def interactive_explorer(df: pd.DataFrame):
    """
    Notebook-only (ipywidgets) explorer supporting dynamic axes and pick strategies.
    """
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
    except Exception as e:
        raise ImportError("ipywidgets + IPython are required for interactive_explorer()") from e

    def uniq(col: str) -> List[Any]:
        if col not in df.columns:
            return []
        vals = [v for v in df[col].dropna().tolist()]
        return _apply_order(_stable_unique(vals), sort=True)

    all_cols = sorted(df.columns)

    # UI Elements
    mode_dd = W.Dropdown(options=["shapes", "weights", "prompts", "custom"], value="shapes", description="preset:")
    rows_dd = W.Dropdown(options=all_cols, value="schedule", description="rows axis:")
    cols_dd = W.Dropdown(options=all_cols, value="seed", description="cols axis:")

    prompt_dd = W.Dropdown(options=uniq("prompt_id"), description="prompt:")
    schedule_dd = W.Dropdown(options=uniq("schedule"), description="schedule:")
    wrange_dd = W.Dropdown(options=uniq("w_range"), description="w_range:")

    filter_key_dd = W.Dropdown(options=["(none)"] + all_cols, value="(none)", description="filter key:")
    filter_val_dd = W.Dropdown(options=[""], value="", description="filter val:")

    # Pick Strategy UI
    pick_rule_dd = W.Dropdown(
        options=["first", "last", "max_col", "exact_match"],
        value="first",
        description="multi-pick:"
    )
    pick_col_dd = W.Dropdown(options=all_cols, value="seed", description="pick col:")
    pick_val_txt = W.Text(value="42", description="pick val:")

    out = W.Output()

    def update_filter_vals(*_):
        key = filter_key_dd.value
        if key == "(none)" or key not in df.columns:
            filter_val_dd.options = [""]
            filter_val_dd.value = ""
            return
        vals = sorted(df[key].dropna().astype(str).unique().tolist())
        filter_val_dd.options = [""] + vals
        filter_val_dd.value = ""

    filter_key_dd.observe(update_filter_vals, names="value")
    update_filter_vals()

    def get_pick_arg():
        rule = pick_rule_dd.value
        if rule in ("first", "last"):
            return rule
        elif rule == "max_col":
            return pick_col_dd.value
        elif rule == "exact_match":
            v = pick_val_txt.value
            col = pick_col_dd.value
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    v = float(v) if '.' in v else int(v)
                except Exception:
                    pass
            return (col, v)
        return "first"

    def render(*_):
        with out:
            clear_output(wait=True)
            mode = mode_dd.value
            style = GridStyle()
            pick_arg = get_pick_arg()

            try:
                if mode == "shapes":
                    if not prompt_dd.value or not wrange_dd.value:
                        print("Select prompt and w_range")
                        return
                    fig = grid_shapes_same_prompt_same_range(
                        df, prompt_id=str(prompt_dd.value), w_range=str(wrange_dd.value),
                        rows=rows_dd.value, cols=cols_dd.value, pick=pick_arg, style=style,
                    )
                elif mode == "weights":
                    if not prompt_dd.value or not schedule_dd.value:
                        print("Select prompt and schedule")
                        return
                    fig = grid_weights_same_prompt_same_schedule(
                        df, prompt_id=str(prompt_dd.value), schedule=str(schedule_dd.value),
                        rows=rows_dd.value, cols=cols_dd.value, pick=pick_arg, style=style,
                    )
                elif mode == "prompts":
                    if not schedule_dd.value or not wrange_dd.value:
                        print("Select schedule and w_range")
                        return
                    fig = grid_prompts_same_schedule_same_weights(
                        df, schedule=str(schedule_dd.value), w_range=str(wrange_dd.value),
                        rows=rows_dd.value, cols=cols_dd.value, pick=pick_arg, style=style,
                    )
                elif mode == "custom":
                    where = {}
                    k = filter_key_dd.value
                    v = filter_val_dd.value
                    if k != "(none)" and v != "":
                        # basic type coercion
                        if pd.api.types.is_numeric_dtype(df[k]):
                            try:
                                v = float(v)
                            except:
                                pass
                        where[k] = v

                    fig = render_grid(
                        df, rows=rows_dd.value, cols=cols_dd.value, where=where or None,
                        pick=pick_arg, caption=("experiment_id",), title="Custom grid", style=style,
                    )
                plt.show()
            except Exception as e:
                print(f"Error rendering grid: {e}")

    # dynamic UI updates
    def update_ui_layout(*_):
        mode = mode_dd.value

        # update default axes if mode changed and it's a "standard" preset change
        # (Optional: might overwrite user preference, but helpful for quick toggling)
        if mode == "shapes":
            rows_dd.value = "schedule"
        elif mode == "weights":
            rows_dd.value = "w_range"
        elif mode == "prompts":
            rows_dd.value = "prompt_id"

        preset_box = W.VBox()
        if mode == "shapes":
            preset_box.children = [prompt_dd, wrange_dd]
        elif mode == "weights":
            preset_box.children = [prompt_dd, schedule_dd]
        elif mode == "prompts":
            preset_box.children = [schedule_dd, wrange_dd]
        elif mode == "custom":
            preset_box.children = [filter_key_dd, filter_val_dd]

        pick_box = W.VBox([pick_rule_dd])
        if pick_rule_dd.value == "max_col":
            pick_box.children = [pick_rule_dd, pick_col_dd]
        elif pick_rule_dd.value == "exact_match":
            pick_box.children = [pick_rule_dd, pick_col_dd, pick_val_txt]

        controls_box.children = [
            W.HTML("<b>Presets & Context</b>"), mode_dd, preset_box,
            W.HTML("<hr><b>Axes Configuration</b>"), rows_dd, cols_dd,
            W.HTML("<hr><b>Multi-Match Resolution</b>"), pick_box
        ]
        render()

    mode_dd.observe(update_ui_layout, names="value")
    pick_rule_dd.observe(update_ui_layout, names="value")

    # Observe all render-triggering widgets
    for w in [prompt_dd, schedule_dd, wrange_dd, rows_dd, cols_dd, filter_key_dd, filter_val_dd, pick_col_dd,
              pick_val_txt]:
        w.observe(render, names="value")

    controls_box = W.VBox(layout=W.Layout(min_width="300px", padding="10px", border="1px solid #ddd"))
    update_ui_layout()

    return W.HBox([controls_box, out])

if __name__ == "__main__":
    import viz_grids as vg

    # 1. Load the data
    df = build_results_df_from_plan_and_images(
        plan_csv=ARTIFACTS_DIR / "EXP_B_EXTREME_TRADEOFF_plan.csv",
        artifacts_dir=ARTIFACTS_DIR,
        jsonl_path=ARTIFACTS_DIR / "results.jsonl",
    )

    # 2. Render a generic grid with dynamic axes: weights vs schedule
    # We have multiple seeds, so let's explicitly pick seed=42
    fig = vg.render_grid(
        df,
        rows="w_range",
        cols="schedule",
        where={"prompt_id": "P_APPLE_COUNT_DETAILED"},
        pick=("seed", 42),  # Resolves duplicates by picking seed 42
        caption=("seed", "experiment_id"),
        title="Weights vs Schedule (Seed 42)"
    )

    # 3. Use presets with overridden columns
    # Compare schedules vs. runtimes dynamically
    fig2 = vg.grid_weights_same_prompt_same_schedule(
        df,
        prompt_id="P_APPLE_COUNT_DETAILED",
        schedule="constant:increasing",
        cols="schedule",  # Override standard cols="seed"
        pick="runtime_s"  # Automatically grabs the max runtime row for the cell
    )

    # 4. Launch the Interactive Explorer
    # All these axes and pick rules are completely adjustable in the UI
    app = vg.interactive_explorer(df)
    app
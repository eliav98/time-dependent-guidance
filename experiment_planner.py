# experiment_planner.py
import pandas as pd
import hashlib
import json
import itertools
from guidance_registry import SCHEDULER_PROPERTIES

# --- EXPERIMENT DEFINITION ---
EXPERIMENT_NAME = "EXP_01_CORE_SWEEP"

SCHEDULERS = list(SCHEDULER_PROPERTIES.keys())
MODES = ["increasing", "decreasing"]
PROMPTS = {
    "P_DETAIL": "A close-up portrait of a cyberpunk robot with intricate clockwork gears and neon eyes, 8k resolution, macro photography",
    "P_COMP": "A symmetrical wide shot of a lone tree in a snowy field, Wes Anderson style, pastel colors",
    "P_TEXT": "A wooden sign hanging on a door that says 'CLOSED' clearly carved into the wood"
}
SEEDS = [42]

# --- THE FIX ---
# We use (1.0, 7.5) so that:
# 1. w_min = 1.0 (No guidance / Floor)
# 2. w_max = 7.5 (Standard SD 1.5 Default / Ceiling)
#
# Our 'constant' scheduler returns w_max, so it will now run at exactly 7.5.
WEIGHT_RANGES = [(1.0, 7.5)]

NUM_STEPS = [25]

EXPERIMENT1 = [EXPERIMENT_NAME, SCHEDULERS, MODES, PROMPTS, SEEDS, WEIGHT_RANGES, NUM_STEPS]


def generate_experiment_1(num_workers=3):
    rows = []

    for kind, mode, w_range, p_id, seed, num_steps in itertools.product(
            SCHEDULERS, MODES, WEIGHT_RANGES, PROMPTS.keys(), SEEDS, NUM_STEPS
    ):
        # Optimization: Constant is static, so 'decreasing' is redundant.
        if kind == "constant" and mode == "decreasing":
            continue

        row = {
            "experiment_group": EXPERIMENT_NAME,
            "kind": kind,
            "direction": mode,
            "w_min": w_range[0],
            "w_max": w_range[1],
            "prompt_id": p_id,
            "prompt_text": PROMPTS[p_id],
            "seed": seed,
            "params": "{}",
            "num_steps": num_steps
        }

        # Unique ID Hashing
        clean = {k: v for k, v in row.items() if k not in ["experiment_group", "prompt_text"]}
        row["experiment_id"] = hashlib.md5(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:12]

        rows.append(row)

    df = pd.DataFrame(rows)
    df["worker_id"] = df.index % num_workers
    return df


if __name__ == "__main__":
    df = generate_experiment_1()
    df.to_csv(f"{EXPERIMENT_NAME}_plan.csv", index=False)
    print(f"Generated {len(df)} experiments with Baseline Range: {WEIGHT_RANGES[0]}")
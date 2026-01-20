import pandas as pd
import hashlib
import json
import itertools

# --- CONFIGURATION SPACE ---
PROMPTS = {
    "P1_REF": "A photorealistic portrait of an elderly man, cinematic lighting",
    "P2_TEXT": "A neon sign that says 'Bouldering' on a dark brick wall"
}

SEARCH_SPACE = {
    "phases": {
        1: {"w_ranges": [(1.0, 7.5), (0.0, 10.0), (1.0, 15.0)], "prompts": ["P1_REF"]},
        2: {"w_ranges": [(1.0, 10.0)], "prompts": ["P1_REF", "P2_TEXT"]}
    },
    "schedulers": [
        {"name": "linear", "cat": "Increasing"},
        {"name": "linear", "cat": "Decreasing"},
        {"name": "cosine", "cat": "Monotonic"},
        {"name": "exponential", "cat": "Monotonic", "params": {"alpha": 4.0}},
        {"name": "v_shape", "cat": "Non-Monotonic"}
    ],
    "seeds": [42]
}


def get_id(cfg):
    """Deterministically hash the experiment config."""
    # Exclude non-defining keys like 'worker_id'
    clean_cfg = {k: v for k, v in cfg.items() if k not in ['worker_id', 'prompt_text']}
    s = json.dumps(clean_cfg, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def generate_plan(num_workers=3):
    rows = []
    for phase_id, phase_cfg in SEARCH_SPACE["phases"].items():
        # Cartesian product of schedulers, ranges, prompts, and seeds
        for s, r, p_id, seed in itertools.product(
                SEARCH_SPACE["schedulers"], phase_cfg["w_ranges"], phase_cfg["prompts"], SEARCH_SPACE["seeds"]
        ):
            row = {
                "phase": phase_id,
                "category": s["cat"],
                "schedule": s["name"],
                "w_min": r[0],
                "w_max": r[1],
                "prompt_id": p_id,
                "prompt_text": PROMPTS[p_id],
                "seed": seed,
                "params": s.get("params", {})
            }
            row["experiment_id"] = get_id(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["worker_id"] = df.index % num_workers
    return df


if __name__ == "__main__":
    df = generate_plan()
    df.to_csv("master_plan.csv", index=False)
    print(f"Generated {len(df)} unique experiments.")
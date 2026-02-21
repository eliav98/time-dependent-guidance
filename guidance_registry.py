# guidance_registry.py
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Literal


# --- 1. RAW SHAPES (Normalized 0 -> 1) ---
class Shapes:
    # --- Monotonic (0 to 1) ---
    @staticmethod
    def linear(p, **kw):
        return p

    @staticmethod
    def cosine(p, **kw):
        return 0.5 - 0.5 * math.cos(math.pi * p)

    @staticmethod
    def exponential(p, alpha=3.0, **kw):
        return (math.exp(alpha * p) - 1) / (math.exp(alpha) - 1)

    @staticmethod
    def logarithmic(p, alpha=3.0, **kw):
        return math.log(1 + (math.exp(alpha) - 1) * p) / alpha

    @staticmethod
    def sigmoid(p, k=12, p0=0.5, **kw):
        return 1 / (1 + math.exp(-k * (p - p0)))

    # --- Non-Monotonic (0 -> 1 -> 0) ---
    @staticmethod
    def triangular(p, peak_pos=0.5, **kw):
        # Mountain shape
        if p < peak_pos:
            return p / peak_pos
        else:
            return (1 - p) / (1 - peak_pos)

    @staticmethod
    def constant(p, **kw):
        return 1.0

    @staticmethod
    def parabolic(p, **kw):
        # Normalized Parabola: 4 * p * (1 - p)
        # This creates a symmetric arch 0 -> 1 -> 0
        return 4 * p * (1 - p)


# --- 2. CONFIGURATION ---
# Define how 'direction' should be interpreted for each shape
SCHEDULER_PROPERTIES = {
    "linear":      {"type": "monotonic"},
    "cosine":      {"type": "monotonic"},
    "exponential": {"type": "monotonic"},
    "logarithmic": {"type": "monotonic"},
    "sigmoid":     {"type": "monotonic"},
    "triangular":  {"type": "non_monotonic"},
    "parabolic":   {"type": "non_monotonic"}, # New!
    "constant":    {"type": "static"}
}


# --- 3. SCHEDULER OBJECT ---
@dataclass
class GuidanceScheduler:
    kind: str
    w_min: float
    w_max: float
    direction: Literal["increasing", "decreasing"] = "increasing"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._func = getattr(Shapes, self.kind, Shapes.linear)
        self._props = SCHEDULER_PROPERTIES.get(self.kind, {"type": "monotonic"})

    def __call__(self, step: int, total_steps: int) -> float:
        # Normalize Progress t (0.0 to 1.0)
        t = step / max(1, total_steps - 1)

        # LOGIC BRANCHING
        category = self._props["type"]
        y = 0.0

        if category == "static":
            y = 1.0
        elif category == "monotonic":
            # Direction affects TIME (Left-to-Right vs Right-to-Left)
            eff_t = t if self.direction == "increasing" else (1.0 - t)
            y = self._func(eff_t, **self.params)
        elif category == "non_monotonic":
            # Direction affects AMPLITUDE (Mountain vs Valley)
            raw_y = self._func(t, **self.params)
            # If 'increasing' -> Standard shape (Mountain)
            # If 'decreasing' -> Inverted shape (Valley / V-Shape)
            y = raw_y if self.direction == "increasing" else (1.0 - raw_y)

        # Scale to Range [w_min, w_max]
        return self.w_min + (self.w_max - self.w_min) * y

    def __repr__(self):
        return f"<Scheduler: {self.kind} | Mode: {self.direction} | Range: {self.w_min}-{self.w_max}>"


def create_scheduler(kind, w_min, w_max, direction="increasing", **kwargs):
    return GuidanceScheduler(kind, w_min, w_max, direction, kwargs)
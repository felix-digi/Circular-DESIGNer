"""Validate positional goal weights before calculating or editing R strategies.

Copyright (C) 2026 Digiser, Riech & Nebel
SPDX-License-Identifier: GPL-3.0-only
"""

from collections.abc import Mapping
import math
from typing import Any


def validate_goal_weights(weights: Any) -> None:
    """Require ten finite numeric weights from -4 to 4 for every named goal.

    Entries correspond to R0 through R9 in order. Decimal commas cannot be
    repaired reliably after JSON parsing, so malformed rows are rejected with
    an actionable error. Validation does not modify the supplied data.
    """
    if not isinstance(weights, Mapping):
        raise ValueError(
            "Goal weights must be a JSON object mapping goal names to lists "
            "of exactly 10 weights for R0 through R9."
        )

    errors = []
    for goal, row in weights.items():
        label = f"Goal {goal!r}"
        if not isinstance(goal, str) or not goal.strip():
            errors.append(f"{label}: the goal name must be a non-empty string.")
        if not isinstance(row, list):
            errors.append(f"{label}: expected a list of exactly 10 weights for R0 through R9.")
            continue
        if len(row) != 10:
            errors.append(
                f"{label}: expected exactly 10 weights for R0 through R9, found {len(row)}. "
                "Use a decimal point (for example, 1.172317705); "
                "a comma creates an extra JSON value."
            )
            continue

        for index, value in enumerate(row):
            prefix = f"{label}, R{index}"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{prefix}: the weight must be a number, not {value!r}.")
            elif isinstance(value, float) and not math.isfinite(value):
                errors.append(f"{prefix}: the weight must be finite.")
            elif not -4.0 <= value <= 4.0:
                errors.append(f"{prefix}: the weight must be between -4.0 and 4.0, found {value!r}.")

    if errors:
        raise ValueError("Invalid goal-to-R-strategy weights:\n" + "\n".join(errors))

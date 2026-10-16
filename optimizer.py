"""
optimizer.py
------------
Sprint optimizer: grid search over planned_story_points range
to find the value that maximizes success probability.
"""

import pandas as pd
import numpy as np
from simulation import simulate_story_points


def optimize_story_points(model, input_data: pd.DataFrame) -> dict:
    """
    Try a range of planned_story_points values and find the optimal one.

    Args:
        model: Trained RandomForestClassifier
        input_data: Single-row DataFrame with all 16 features

    Returns:
        dict with optimal_story_points, optimal_probability, improvement_percent, message, curve
    """
    current_sp = int(input_data.iloc[0]["planned_story_points"])
    prev_velocity = int(input_data.iloc[0]["previous_sprint_velocity"])

    # Search range: 60% to 130% of current, bounded by realistic limits
    low = max(10, int(current_sp * 0.5))
    high = min(80, int(current_sp * 1.3))
    step = max(1, (high - low) // 20)

    candidates = list(range(low, high + 1, step))
    if current_sp not in candidates:
        candidates.append(current_sp)
    candidates = sorted(set(candidates))

    best_sp = current_sp
    best_proba = 0.0
    curve = []

    for sp in candidates:
        result = simulate_story_points(model, input_data, sp)
        proba = result["new_probability"]
        curve.append({"story_points": sp, "success_probability": proba})
        if proba > best_proba:
            best_proba = proba
            best_sp = sp

    original_proba = simulate_story_points(model, input_data, current_sp)["old_probability"]
    improvement = round(best_proba - original_proba, 2)

    if best_sp < current_sp:
        action = "Reducing"
        direction = "down"
    elif best_sp > current_sp:
        action = "Increasing"
        direction = "up"
    else:
        action = "Keeping"
        direction = "as-is"

    message = (
        f"{action} planned story points from {current_sp} → {best_sp} "
        f"improves success probability by {improvement:.1f}%."
        if direction != "as-is"
        else f"Current story points ({current_sp}) are already near-optimal for this sprint."
    )

    return {
        "current_story_points": current_sp,
        "optimal_story_points": best_sp,
        "current_probability": round(original_proba, 2),
        "optimal_probability": round(best_proba, 2),
        "improvement_percent": improvement,
        "message": message,
        "curve": curve,  # list of {story_points, success_probability} for plotting
    }

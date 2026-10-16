"""
simulation.py
-------------
What-if simulation: modify planned_story_points and observe
how sprint success probability changes using the same trained model.
"""

import pandas as pd
import numpy as np


def simulate_story_points(model, input_data: pd.DataFrame, new_story_points: int) -> dict:
    """
    Simulate a change in planned_story_points and return delta in success probability.

    Args:
        model: Trained RandomForestClassifier
        input_data: Original single-row DataFrame (16 features)
        new_story_points: New value for planned_story_points to test

    Returns:
        dict with old_probability, new_probability, improvement_percent, delta_label
    """
    # Original probability (class 1 = success)
    old_proba = model.predict_proba(input_data)[0][1]

    # Build modified input — recalculate derived features
    modified = input_data.copy()
    row = modified.iloc[0]

    new_sp = max(1, int(new_story_points))
    prev_velocity = int(row["previous_sprint_velocity"])
    total_tasks = int(row["total_tasks"])
    team_size = int(row["team_size"])
    carry_over = int(row["carry_over_story_points"])
    prev_planned = int(row["previous_planned_story_points"])
    prev_completed = int(row["previous_completed_story_points"])

    # Recalculate derived features that depend on planned_story_points
    new_velocity_ratio = new_sp / (prev_velocity + 1)
    new_carry_over_ratio = carry_over / (new_sp + 1)
    new_workload_per_developer = total_tasks / team_size  # unchanged
    estimation_accuracy = prev_completed / (prev_planned + 1)
    buffer_ratio_previous = max((prev_planned - prev_completed) / (prev_planned + 1), 0)

    modified.at[0, "planned_story_points"] = new_sp
    modified.at[0, "velocity_ratio"] = new_velocity_ratio
    modified.at[0, "carry_over_ratio"] = new_carry_over_ratio
    modified.at[0, "workload_per_developer"] = new_workload_per_developer
    modified.at[0, "estimation_accuracy"] = estimation_accuracy
    modified.at[0, "buffer_ratio_previous"] = buffer_ratio_previous

    new_proba = model.predict_proba(modified)[0][1]

    improvement = new_proba - old_proba
    improvement_pct = round(improvement * 100, 2)

    if improvement > 0.02:
        delta_label = f"↑ Success probability improves by {improvement_pct:.1f}%"
    elif improvement < -0.02:
        delta_label = f"↓ Success probability drops by {abs(improvement_pct):.1f}%"
    else:
        delta_label = "→ Negligible change in success probability"

    return {
        "old_story_points": int(input_data.iloc[0]["planned_story_points"]),
        "new_story_points": new_sp,
        "old_probability": round(float(old_proba) * 100, 2),
        "new_probability": round(float(new_proba) * 100, 2),
        "improvement_percent": improvement_pct,
        "delta_label": delta_label,
        "modified_input": modified,
    }

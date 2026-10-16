"""
recommendation.py
-----------------
Rule-based recommendation engine for sprint planning.
Uses feature thresholds derived from dataset statistics.
NO randomness, NO LLM calls — pure deterministic logic.
"""

import pandas as pd


def get_recommendations(input_data: pd.DataFrame) -> list:
    """
    Generate 2–4 actionable recommendations based on feature values.

    Args:
        input_data: Single-row DataFrame with all 16 features

    Returns:
        List of dicts with keys: priority, action, rationale
    """
    row = input_data.iloc[0]
    recommendations = []

    # --- RULE 1: Overcommitment (velocity ratio) ---
    if row["velocity_ratio"] > 1.2:
        excess_pct = round((row["velocity_ratio"] - 1.0) * 100, 1)
        recommendations.append({
            "priority": "HIGH",
            "action": f"Reduce planned story points by ~{min(excess_pct, 30):.0f}%",
            "rationale": (
                f"Your velocity ratio is {row['velocity_ratio']:.2f} — you are planning "
                f"{excess_pct}% more than your team historically delivers. "
                "Trim scope to match proven capacity."
            ),
        })

    # --- RULE 2: Workload per developer ---
    if row["workload_per_developer"] > 4.5:
        recommendations.append({
            "priority": "HIGH",
            "action": "Redistribute or defer tasks across team members",
            "rationale": (
                f"Each developer is assigned ~{row['workload_per_developer']:.1f} tasks on average. "
                "Above 4.5 tasks/developer is a strong predictor of sprint failure. "
                "Consider deferring low-priority tasks to the backlog."
            ),
        })
    elif row["workload_per_developer"] > 3.5:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Monitor task allocation — workload is approaching risky levels",
            "rationale": (
                f"Workload per developer is {row['workload_per_developer']:.1f} tasks. "
                "While not critical, aim to keep this below 4 for healthy sprint execution."
            ),
        })

    # --- RULE 3: Carry-over debt ---
    if row["carry_over_ratio"] > 0.2:
        pct = round(row["carry_over_ratio"] * 100, 1)
        recommendations.append({
            "priority": "HIGH",
            "action": "Clear carry-over backlog before accepting new work",
            "rationale": (
                f"{pct}% of this sprint's capacity is already allocated to carry-over work. "
                "Prioritize completing unfinished work before committing to new features. "
                "Compounding debt degrades team morale and velocity."
            ),
        })
    elif row["carry_over_ratio"] > 0.1:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Review and resolve carry-over items early in the sprint",
            "rationale": (
                f"Carry-over ratio is {row['carry_over_ratio']:.2f} — moderate backlog debt detected. "
                "Assign owners to carry-over items on Day 1 to avoid re-deferral."
            ),
        })

    # --- RULE 4: Poor estimation accuracy ---
    if row["estimation_accuracy"] < 0.80:
        recommendations.append({
            "priority": "HIGH",
            "action": "Conduct an estimation calibration workshop this sprint",
            "rationale": (
                f"Historical estimation accuracy is {row['estimation_accuracy']:.0%} — "
                "team is consistently underestimating effort. "
                "Use Planning Poker with reference stories to re-calibrate point values."
            ),
        })
    elif row["estimation_accuracy"] < 0.90:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Add a 10–15% buffer to story point estimates",
            "rationale": (
                f"Estimation accuracy is {row['estimation_accuracy']:.0%}. "
                "A modest planning buffer will absorb estimation gaps without overloading the sprint."
            ),
        })

    # --- RULE 5: High bug debt from previous sprint ---
    if row["bugs_reported_previous_sprint"] >= 10:
        recommendations.append({
            "priority": "HIGH",
            "action": "Allocate dedicated bug-fix capacity (20–30% of sprint)",
            "rationale": (
                f"{int(row['bugs_reported_previous_sprint'])} bugs were reported last sprint. "
                "Unaddressed defects compound and consume unplanned bandwidth mid-sprint. "
                "Reserve explicit capacity before committing to new features."
            ),
        })
    elif row["bugs_reported_previous_sprint"] >= 6:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Schedule bug triage in Sprint Planning",
            "rationale": (
                f"{int(row['bugs_reported_previous_sprint'])} bugs carried from last sprint. "
                "Triage and prioritize fixes before finalizing the sprint backlog."
            ),
        })

    # --- RULE 6: High priority task concentration ---
    if row["high_priority_tasks"] / max(row["total_tasks"], 1) > 0.4:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Reassess task priority distribution — too many P1 items",
            "rationale": (
                f"{int(row['high_priority_tasks'])} of {int(row['total_tasks'])} tasks are high priority "
                f"({row['high_priority_tasks']/row['total_tasks']:.0%}). "
                "When everything is urgent, nothing is. Re-prioritize with stakeholders."
            ),
        })

    # Cap at 4 recommendations — return highest priority first
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 2))

    return recommendations[:4]

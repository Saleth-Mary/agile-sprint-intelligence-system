"""
retrospective.py
----------------
Auto-generates a structured sprint retrospective based on:
- Feature values (pre-sprint data)
- Model prediction & probability
- SHAP explanations
NO generic text. Every sentence is grounded in actual data.
"""

import pandas as pd


def _format_feature(name: str) -> str:
    return name.replace("_", " ").title()


def generate_retrospective(
    input_data: pd.DataFrame,
    prediction: int,
    probability: float,
    explanations: list,
) -> dict:
    """
    Generate a 3-section retrospective based on model outputs.

    Args:
        input_data: Single-row DataFrame with all 16 features
        prediction: 1 = success, 0 = failure
        probability: Success probability (0.0–1.0)
        explanations: List of dicts from explainability.explain_prediction()

    Returns:
        dict with sections: went_wrong, went_well, to_improve
    """
    row = input_data.iloc[0]
    pct = round(probability * 100, 1)

    went_wrong = []
    went_well = []
    to_improve = []

    # ---- WHAT WENT WRONG (risk factors) ----
    risk_factors = [e for e in explanations if e["impact"] == "increases risk"]
    protective = [e for e in explanations if e["impact"] == "reduces risk"]

    if row["velocity_ratio"] > 1.15:
        went_wrong.append(
            f"**Overcommitment:** The team planned {int(row['planned_story_points'])} story points against "
            f"a historical velocity of {int(row['previous_sprint_velocity'])}. "
            f"This {round((row['velocity_ratio'] - 1) * 100, 1)}% overshoot is a proven failure driver."
        )

    if row["carry_over_ratio"] > 0.15:
        went_wrong.append(
            f"**Carry-over debt:** {int(row['carry_over_story_points'])} story points ({round(row['carry_over_ratio']*100,1)}% of capacity) "
            "were inherited from the previous sprint, compressing available bandwidth before Day 1."
        )

    if row["workload_per_developer"] > 4.0:
        went_wrong.append(
            f"**Developer overload:** Each developer was assigned ~{row['workload_per_developer']:.1f} tasks. "
            "Sustained above 4 tasks/person leads to context-switching, incomplete reviews, and burnout."
        )

    if row["estimation_accuracy"] < 0.88:
        went_wrong.append(
            f"**Estimation gap:** The team historically completes only "
            f"{round(row['estimation_accuracy']*100, 1)}% of what they plan. "
            "Story point inflation is inflating the backlog without reflecting true capacity."
        )

    if row["bugs_reported_previous_sprint"] >= 8:
        went_wrong.append(
            f"**Quality debt:** {int(row['bugs_reported_previous_sprint'])} bugs from the previous sprint "
            "consumed unplanned sprint capacity, derailing feature delivery timelines."
        )

    if row["high_priority_tasks"] / max(row["total_tasks"], 1) > 0.4:
        went_wrong.append(
            f"**Priority fragmentation:** {int(row['high_priority_tasks'])} of {int(row['total_tasks'])} tasks "
            "were marked high priority. When everything is urgent, teams lose focus on delivery."
        )

    # Add top SHAP risk factors if nothing triggered above
    if not went_wrong and risk_factors:
        for e in risk_factors[:2]:
            went_wrong.append(
                f"**{_format_feature(e['feature'])}** (value: {e['value']}) — {e['reason']}"
            )

    if not went_wrong:
        went_wrong.append(
            f"No major failure patterns detected. Predicted success probability is {pct}%. "
            "Minor refinements could still improve execution confidence."
        )

    # ---- WHAT WENT WELL (green signals) ----
    if row["estimation_accuracy"] >= 0.92:
        went_well.append(
            f"**Estimation maturity:** The team completed {round(row['estimation_accuracy']*100,1)}% of planned work last sprint — "
            "strong historical accuracy builds planning confidence."
        )

    if row["bugs_reported_previous_sprint"] < 5:
        went_well.append(
            f"**Code quality:** Only {int(row['bugs_reported_previous_sprint'])} bugs reported last sprint. "
            "Clean code output reduces unplanned rework and preserves delivery bandwidth."
        )

    if row["carry_over_ratio"] <= 0.1:
        went_well.append(
            f"**Clean handoff:** Only {int(row['carry_over_story_points'])} story points carried over "
            "from the previous sprint. The team entered this sprint with fresh capacity."
        )

    if row["velocity_ratio"] <= 1.05:
        went_well.append(
            f"**Realistic scoping:** Planned story points ({int(row['planned_story_points'])}) align closely "
            f"with historical velocity ({int(row['previous_sprint_velocity'])}). "
            "No significant overcommitment detected."
        )

    if row["workload_per_developer"] <= 3.5:
        went_well.append(
            f"**Healthy workload:** {row['workload_per_developer']:.1f} tasks per developer "
            "is within the optimal range for focused, high-quality delivery."
        )

    # Add protective SHAP factors if nothing triggered
    if not went_well and protective:
        for e in protective[:2]:
            went_well.append(
                f"**{_format_feature(e['feature'])}** (value: {e['value']}) — {e['reason']}"
            )

    if not went_well:
        went_well.append(
            "Sprint parameters are within baseline range. "
            "No standout positive signals detected — focus on incremental improvements."
        )

    # ---- WHAT TO IMPROVE ----
    if row["velocity_ratio"] > 1.1:
        to_improve.append(
            f"**Scope calibration:** Reduce planned story points to ≤ {int(row['previous_sprint_velocity'])} "
            "(matching historical velocity) to eliminate the overcommitment pattern."
        )

    if row["estimation_accuracy"] < 0.88:
        to_improve.append(
            "**Estimation process:** Run a 2-sprint estimation recalibration using reference stories. "
            "Introduce velocity-based ceilings in Sprint Planning to prevent inflation."
        )

    if row["carry_over_ratio"] > 0.15:
        to_improve.append(
            "**Definition of Done:** Strengthen DoD enforcement. "
            "Stories should not be accepted into the sprint if they cannot realistically complete within the sprint window."
        )

    if row["bugs_reported_previous_sprint"] >= 8:
        to_improve.append(
            "**Quality gates:** Introduce a mandatory code review SLA and automated test coverage targets "
            "to reduce defect escape rate into production."
        )

    if row["workload_per_developer"] > 4.0:
        to_improve.append(
            f"**Task distribution:** Cap tasks per developer at 4. "
            "Use a capacity planning buffer (80% of available hours) to absorb unexpected work."
        )

    if not to_improve:
        to_improve.append(
            f"Sprint execution risk is low ({pct}% success probability). "
            "Maintain current planning discipline and focus on continuous delivery cadence."
        )

    return {
        "went_wrong": went_wrong,
        "went_well": went_well,
        "to_improve": to_improve,
        "sprint_status": "PREDICTED SUCCESS" if prediction == 1 else "PREDICTED FAILURE",
        "probability": pct,
    }

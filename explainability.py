"""
explainability.py
-----------------
Improved SHAP-based feature explanation for sprint risk prediction.
Now includes:
- dynamic reasoning
- impact strength (strong / moderate / minor)
- value-based contextual explanations
- non-repetitive output
"""

import shap
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def get_dynamic_reason(feature, value, shap_value):
    """Generate realistic explanation based on value + context"""

    if feature == "workload_per_developer":
        if value > 5:
            return "Developers are heavily overloaded, increasing burnout and delivery delays."
        elif value > 3.5:
            return "Workload is moderately high, which may impact focus and delivery speed."
        else:
            return "Workload is well balanced across developers."

    elif feature == "velocity_ratio":
        if value > 1.3:
            return "Sprint scope significantly exceeds team capacity, indicating overcommitment."
        elif value > 1.1:
            return "Planned work is slightly above capacity, introducing moderate risk."
        else:
            return "Planned scope aligns well with team capacity."

    elif feature == "carry_over_ratio":
        if value > 0.25:
            return "Large portion of unfinished work is carried over, reducing available capacity."
        elif value > 0.1:
            return "Some carry-over work exists, which may slightly impact progress."
        else:
            return "Minimal carry-over allows a fresh start for the sprint."

    elif feature == "bugs_reported_previous_sprint":
        if value >= 10:
            return "High number of bugs indicates quality issues affecting upcoming sprint."
        elif value >= 5:
            return "Moderate bug count may consume some development effort."
        else:
            return "Low bug count supports smoother sprint execution."

    elif feature == "estimation_accuracy":
        if value < 0.8:
            return "Team tends to underestimate effort, leading to incomplete work."
        elif value > 1.05:
            return "Team often over-delivers, indicating strong execution capability."
        else:
            return "Estimation accuracy is reasonably balanced."

    elif feature == "high_priority_tasks":
        if value > 8:
            return "Too many high-priority tasks create context switching and pressure."
        elif value > 4:
            return "Moderate number of high-priority tasks may affect focus."
        else:
            return "Priority distribution is manageable."

    elif feature == "total_tasks":
        if value > 30:
            return "High task count increases coordination complexity."
        elif value > 20:
            return "Moderate number of tasks may slightly increase workload."
        else:
            return "Task count is manageable."

    elif feature == "team_size":
        if value > 8:
            return "Larger teams may introduce communication overhead."
        elif value < 4:
            return "Smaller team may face capacity limitations."
        else:
            return "Team size is balanced."

    elif feature == "planned_story_points":
        if value > 70:
            return "Large sprint scope increases delivery complexity."
        elif value > 50:
            return "Moderately large sprint scope may introduce risk."
        else:
            return "Sprint scope is reasonable."
    elif feature == "carry_over_ratio":
        if value > 0.25:
            return "Significant unfinished work from previous sprint may reduce current capacity."
        elif value > 0.1:
            return "Some carry-over work may slightly affect sprint progress."
        else:
            return "Minimal carry-over work allows better focus on new tasks."

    else:
        return f"Current value is {round(value,2)}"


def explain_prediction(model, input_data: pd.DataFrame, top_n: int = 3) -> list:
    """
    Compute SHAP values and return realistic explanations
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # use failure class
    sv_failure = np.array(shap_values)[0, :, 1]

    features = list(input_data.columns)
    feature_vals = input_data.iloc[0].to_dict()

    ranked = sorted(
        zip(features, sv_failure),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    results = []

    for feat, sv in ranked:
        val = feature_vals[feat]

        # impact strength
        impact_strength = abs(sv)
        if impact_strength > 0.15:
            level = "strong"
        elif impact_strength > 0.07:
            level = "moderate"
        else:
            level = "minor"

        # direction
        if sv > 0:
            impact_label = f"{level} risk increase"
        else:
            impact_label = f"{level} risk reduction"

        reason = get_dynamic_reason(feat, val, sv)

        results.append({
            "feature": feat,
            "shap_value": round(float(sv), 4),
            "impact": impact_label,
            "reason": reason,
            "value": round(val, 3),
        })

    return results
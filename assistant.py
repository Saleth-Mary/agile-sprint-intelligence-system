"""
assistant.py
------------
Rule-based natural language assistant for sprint analysis.
Uses model output + SHAP explanations to answer user questions.
NO hallucination — all answers grounded in actual feature values.
"""

import re
import pandas as pd


# Keyword → answer builder mapping
QUESTION_HANDLERS = {
    "fail": "failure_reason",
    "why": "failure_reason",
    "risk": "risk_breakdown",
    "what went wrong": "failure_reason",
    "success": "success_chance",
    "probability": "success_chance",
    "chance": "success_chance",
    "improve": "improvement_advice",
    "fix": "improvement_advice",
    "recommend": "improvement_advice",
    "workload": "workload_analysis",
    "team": "team_analysis",
    "velocity": "velocity_analysis",
    "carry": "carryover_analysis",
    "bug": "bug_analysis",
    "estimate": "estimation_analysis",
    "plan": "planning_summary",
    "summary": "planning_summary",
    "overview": "planning_summary",
}


def _format_feature(name: str) -> str:
    return name.replace("_", " ").title()


def answer_question(
    question: str,
    input_data: pd.DataFrame,
    prediction: int,
    probability: float,
    explanations: list,
    recommendations: list,
) -> str:
    """
    Generate a grounded natural language answer to the user's sprint question.
    """

    # Normalize question
    q = question.lower().strip()
    row = input_data.iloc[0]
    status = "likely to SUCCEED" if prediction == 1 else "likely to FAIL"
    pct = round(probability * 100, 1)

    # ===============================
    # SPECIAL FIX (FAIL QUESTION)
    # ===============================
    if "fail" in q:
        if prediction == 1:
            return (
                f"Sprint Status: {status} ({pct}% success probability)\n\n"
                "There are no major failure risks identified. However, potential risks could arise if:\n"
                "- Workload increases unexpectedly\n"
                "- New bugs or unplanned tasks appear\n"
                "- Team capacity reduces (leave, blockers)\n\n"
                "Currently, the sprint setup looks stable."
            )
        else:
            reasons = []
            for e in explanations:
                if "increase" in e["impact"]:
                    reasons.append(f"- {e['reason']}")

            return (
                f"Sprint Status: {status} ({pct}% success probability)\n\n"
                "Key reasons for potential failure:\n"
                + "\n".join(reasons[:3])
            )

    # ===============================
    # NORMAL HANDLER FLOW (FIXED)
    # ===============================

    handler = None
    for keyword, h in QUESTION_HANDLERS.items():
        if keyword in q:
            handler = h
            break

    # ✅ Handle unknown questions
    if handler is None:
        return (
            "🤖 I couldn’t fully understand your question.\n\n"
            "You can try asking things like:\n"
            "- Why will this sprint fail?\n"
            "- How do I improve success?\n"
            "- Analyze workload\n"
            "- What is my velocity risk?"
         )

    # --- HANDLERS ---

    if handler == "failure_reason":
        top = explanations[:3]
        lines = [
            f"**Sprint Status:** {status} ({pct}% success probability)\n",
            "**Root Causes (SHAP Analysis):**",
        ]
        for i, e in enumerate(top, 1):
            sign = "⬆ Risk factor" if e["impact"] == "increases risk" else "⬇ Protective factor"
            lines.append(
                f"{i}. **{_format_feature(e['feature'])}** = {e['value']} — {sign}\n"
                f"   {e['reason']}"
            )
        return "\n".join(lines)

    elif handler == "success_chance":
        return (
            f"**Success Probability: {pct}%** — This sprint is {status}.\n"
        )

    elif handler == "improvement_advice":
        lines = [f"**How to improve success ({pct}% → higher):**\n"]
        for i, r in enumerate(recommendations[:3], 1):
            lines.append(f"{i}. {r['action']}\n   {r['rationale']}")
        return "\n".join(lines)

    elif handler == "workload_analysis":
        wpd = row["workload_per_developer"]
        return (
            f"Workload per developer: {wpd:.1f}\n"
            + ("⚠️ High workload" if wpd > 4.5 else "✅ Balanced workload")
        )

    elif handler == "velocity_analysis":
        vr = row["velocity_ratio"]
        return (
            f"Velocity ratio: {vr:.2f}\n"
            + ("⚠️ Overcommitment" if vr > 1.1 else "✅ Within capacity")
        )

    elif handler == "risk_breakdown":
        lines = [f"Risk Overview ({pct}% success):"]
        for e in explanations:
            direction = "↑" if "increase" in e["impact"] else "↓"
            lines.append(f"{direction} {_format_feature(e['feature'])}")
        return "\n".join(lines)

    # DEFAULT
    return (
        f"Sprint Status: {status} ({pct}% success probability)\n\n"
        "Ask:\n"
        "- Why will this sprint fail?\n"
        "- How do I improve success?\n"
        "- Analyze workload\n"
        "- What is my velocity risk?"
    )
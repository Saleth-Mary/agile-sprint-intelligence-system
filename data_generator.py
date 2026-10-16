import pandas as pd
import numpy as np

def generate_sprint_data(n=3000, seed=42):
    np.random.seed(seed)
    R = n

    # ── TEAM IDENTITY ────────────────────────────────────────────────────
    team_ids = [f"TEAM-{str(i).zfill(3)}" for i in np.random.randint(1, 31, R)]
    sprint_ids = [f"SPR-{str(i).zfill(4)}" for i in range(1, R + 1)]

    team_exp = np.random.choice(["Junior", "Mixed", "Senior"], R, p=[0.25, 0.50, 0.25])
    team_size = np.where(team_exp == "Junior",
                    np.random.randint(4, 7, R),
                    np.where(team_exp == "Senior",
                        np.random.randint(5, 10, R),
                        np.random.randint(4, 9, R)))

    dev_exp_avg = np.where(team_exp == "Junior",
                    np.round(np.random.uniform(1, 3, R), 1),
                    np.where(team_exp == "Senior",
                        np.round(np.random.uniform(6, 12, R), 1),
                        np.round(np.random.uniform(3, 7, R), 1)))

    sprint_dur = np.random.choice([10, 14], R, p=[0.25, 0.75])

    # ── A. SPRINT PLANNING ────────────────────────────────────────────────
    planned_pts = np.random.randint(25, 75, R)
    prev_velocity = (planned_pts * np.random.uniform(0.65, 1.0, R)).round().astype(int)
    velocity_trend = np.random.choice(["Increasing", "Stable", "Decreasing"], R, p=[0.35, 0.40, 0.25])

    vel_last3 = (prev_velocity * np.random.uniform(0.85, 1.1, R)).round().astype(int)
    vel_variance = np.abs(np.random.normal(0, 5, R)).round(2)

    carry_over_tasks = np.random.randint(0, 8, R)
    carry_over_pts   = (carry_over_tasks * np.random.uniform(2, 5, R)).round().astype(int)

    # ── B. EXECUTION (SCOPE CREEP & FLOW) ────────────────────────────────
    total_tasks        = np.random.randint(12, 32, R)
    scope_added_tasks  = np.random.randint(0, 7, R)
    scope_removed_tasks= np.random.randint(0, 4, R)
    reopened_tasks     = np.random.randint(0, 5, R)
    avg_blocked_time   = np.round(np.random.uniform(0, 48, R), 1)   # hours
    blocked_tasks      = np.random.randint(0, 6, R)
    wip_tasks          = np.random.randint(1, 10, R)
    priority_changes   = np.random.randint(0, 8, R)
    large_story_ratio  = np.round(np.random.uniform(0, 0.5, R), 2)  # fraction of stories > 8 pts

    avg_task_complexity = np.round(np.random.uniform(1.5, 5.0, R), 2)
    high_priority_tasks = np.random.randint(2, 10, R)

    # ── C. BUG & QUALITY ─────────────────────────────────────────────────
    bugs_prev_sprint      = np.random.randint(0, 14, R)
    bugs_created_sprint   = np.random.randint(0, 10, R)
    bugs_resolved         = np.random.randint(0, 12, R)
    critical_bugs         = np.random.randint(0, 4, R)

    # ── D. TIME & FLOW METRICS ────────────────────────────────────────────
    avg_cycle_time  = np.round(np.random.uniform(0.5, 5.0, R), 2)   # days
    avg_lead_time   = np.round(avg_cycle_time + np.random.uniform(0.5, 3.0, R), 2)
    time_in_review  = np.round(np.random.uniform(0.5, 3.0, R), 2)   # days

    # ── E. TEAM BEHAVIOR ─────────────────────────────────────────────────
    meetings_hours        = np.random.randint(4, 22, R)
    workload_variance     = np.round(np.random.uniform(0.0, 3.0, R), 2)
    avg_comments_per_task = np.round(np.random.uniform(1, 8, R), 2)

    # ── DERIVED FEATURES ─────────────────────────────────────────────────
    workload_per_dev = np.round(total_tasks / team_size, 2)
    velocity_ratio   = np.round(planned_pts / (prev_velocity + 1), 3)
    bug_pressure     = np.round(bugs_prev_sprint / (team_size + 1), 3)
    meeting_load     = np.round(meetings_hours / (team_size + 1), 3)
    scope_creep_rate = np.round(scope_added_tasks / (total_tasks + 1), 3)
    bug_resolution_rate = np.round(bugs_resolved / (bugs_created_sprint + bugs_prev_sprint + 1), 3)

    # ── TARGET: COMPLETED STORY POINTS (regression) ──────────────────────
    exp_bonus = np.where(team_exp == "Senior", 5,
                np.where(team_exp == "Junior", -5, 0))

    noise = np.random.normal(0, 4, R)

    completed_pts = (
        prev_velocity
        + exp_bonus
        - blocked_tasks * 2.5
        - avg_blocked_time * 0.05
        - bugs_prev_sprint * 0.4
        - critical_bugs * 1.5
        - scope_added_tasks * 1.2
        - carry_over_tasks * 0.8
        - priority_changes * 0.5
        - wip_tasks * 0.3
        + bugs_resolved * 0.3
        + noise
    ).round().clip(5).astype(int)

    # ── TARGET: SPRINT SUCCESS (classification) ───────────────────────────
    sprint_success = np.where(
        completed_pts >= 0.75 * planned_pts, 1, 0
    )

    df = pd.DataFrame({
        "sprint_id":                sprint_ids,
        "team_id":                  team_ids,

        # Team
        "team_size":                team_size,
        "team_experience_level":    team_exp,
        "developer_experience_avg": dev_exp_avg,
        "sprint_duration_days":     sprint_dur,

        # Planning
        "planned_story_points":     planned_pts,
        "previous_sprint_velocity": prev_velocity,
        "velocity_last_3_avg":      vel_last3,
        "velocity_variance":        vel_variance,
        "team_velocity_trend":      velocity_trend,
        "carry_over_tasks":         carry_over_tasks,
        "carry_over_story_points":  carry_over_pts,

        # Execution
        "total_tasks":              total_tasks,
        "scope_added_tasks":        scope_added_tasks,
        "scope_removed_tasks":      scope_removed_tasks,
        "reopened_tasks":           reopened_tasks,
        "blocked_tasks":            blocked_tasks,
        "avg_blocked_time":         avg_blocked_time,
        "wip_tasks":                wip_tasks,
        "priority_changes":         priority_changes,
        "large_story_ratio":        large_story_ratio,
        "avg_task_complexity":      avg_task_complexity,
        "high_priority_tasks":      high_priority_tasks,

        # Bug & Quality
        "bugs_reported_previous_sprint": bugs_prev_sprint,
        "bugs_created_current_sprint":   bugs_created_sprint,
        "bugs_resolved":                 bugs_resolved,
        "critical_bugs":                 critical_bugs,

        # Time & Flow
        "avg_cycle_time":           avg_cycle_time,
        "avg_lead_time":            avg_lead_time,
        "time_in_review":           time_in_review,

        # Team Behavior
        "meetings_hours":           meetings_hours,
        "workload_variance":        workload_variance,
        "avg_comments_per_task":    avg_comments_per_task,

        # Derived
        "workload_per_developer":   workload_per_dev,
        "velocity_ratio":           velocity_ratio,
        "bug_pressure":             bug_pressure,
        "meeting_load":             meeting_load,
        "scope_creep_rate":         scope_creep_rate,
        "bug_resolution_rate":      bug_resolution_rate,

        # Targets
        "completed_story_points":   completed_pts,
        "sprint_success":           sprint_success,
    })

    return df


def get_feature_cols():
    return [
        "team_size", "developer_experience_avg", "sprint_duration_days",
        "planned_story_points", "previous_sprint_velocity", "velocity_last_3_avg",
        "velocity_variance", "carry_over_tasks", "carry_over_story_points",
        "total_tasks", "scope_added_tasks", "scope_removed_tasks",
        "reopened_tasks", "blocked_tasks", "avg_blocked_time",
        "wip_tasks", "priority_changes", "large_story_ratio",
        "avg_task_complexity", "high_priority_tasks",
        "bugs_reported_previous_sprint", "bugs_created_current_sprint",
        "bugs_resolved", "critical_bugs",
        "avg_cycle_time", "avg_lead_time", "time_in_review",
        "meetings_hours", "workload_variance", "avg_comments_per_task",
        "workload_per_developer", "velocity_ratio", "bug_pressure",
        "meeting_load", "scope_creep_rate", "bug_resolution_rate",
        # encoded categoricals added at runtime
        "team_experience_enc", "velocity_trend_enc",
    ]


if __name__ == "__main__":
    df = generate_sprint_data()
    df.to_csv("sprint_data.csv", index=False)
    print(f"Generated {len(df)} sprints with {len(df.columns)} columns")
    print(df["sprint_success"].value_counts())

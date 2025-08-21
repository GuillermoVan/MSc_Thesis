# -*- coding: utf-8 -*-
"""
Creates 3 figures (one per parameter) with 5 subplots each (3 on top, 2 centered below).
Plots Deterministic (solid) vs Scenario-based (dashed) with ±1 std shaded bands.
Legends are drawn inside each subplot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ---- INPUT CSV ----
csv_path = r"C:\Users\guillermolambertus.v\Thesis_project_code\Results_benchmark\Sensitivity analysis\Sensitivity_results.csv"

# ---- Load with European number parsing ----
df = pd.read_csv(csv_path, sep=';', decimal=',', thousands='.', engine='python')

# Clean up headers & text
df.columns = df.columns.str.strip()
df["Scheduler_type"] = df["Scheduler_type"].astype(str).str.strip()
df["Parameter"]      = df["Parameter"].astype(str).str.strip()
df["Scheduler_type"] = df["Scheduler_type"].replace({"Stochastic": "Scenario-based"})

# Ensure Value is numeric
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Coerce KPI columns to numeric
for col in [
    "Objective_mean", "Objective_STD",
    "Makespan_mean", "Makespan_STD",
    "Total_delay_mean", "Total_delay_STD",
    "Delayed_count_mean", "Delayed_count_STD",
    "Max_delay_mean", "Max_delay_STD",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

valid_sched = ["Deterministic", "Scenario-based"]
df = df[df["Scheduler_type"].isin(valid_sched)]

# Metrics to plot: (mean_col, std_col, title, y_label)
METRICS = [
    ("Objective_mean",      "Objective_STD",      "Objective",          "Objective"),
    ("Makespan_mean",       "Makespan_STD",       "Makespan",           "Makespan"),
    ("Total_delay_mean",    "Total_delay_STD",    "Total delay",        "Total delay (minutes)"),
    ("Delayed_count_mean",  "Delayed_count_STD",  "Delayed count",      "Delayed count (#)"),
    ("Max_delay_mean",      "Max_delay_STD",      "Maximum delay",      "Maximum delay (minutes)"),
]

def _plot_metric_on_ax(ax, dparam, schedulers_to_plot, mean_col, std_col, title, ylab, xlab):
    styles = {"Deterministic": "-", "Scenario-based": "--"}
    line_handles = []

    for sched in schedulers_to_plot:
        d = (dparam[dparam["Scheduler_type"] == sched]
             .dropna(subset=["Value", mean_col])
             .sort_values("Value"))
        if d.empty:
            continue

        line, = ax.plot(
            d["Value"], d[mean_col],
            linestyle=styles.get(sched, "-"),
            marker="o",
            label=f"{sched} mean"
        )
        c = line.get_color()
        line_handles.append(line)

        s = d[std_col].to_numpy() if std_col in d else None
        if s is not None:
            m = d[mean_col].to_numpy()
            x = d["Value"].to_numpy()
            mask = pd.notna(s)
            if mask.any():
                ax.fill_between(x[mask], m[mask]-s[mask], m[mask]+s[mask],
                                color=c, alpha=0.25)

    ax.set_title(title, pad=6)
    ax.set_xlabel(xlab, labelpad=8)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.06)

    if line_handles:
        std_patch = Patch(color="gray", alpha=0.25, label="±1 Std Dev")
        # Legend INSIDE the plot
        ax.legend(handles=line_handles + [std_patch],
                  fontsize=9, frameon=True, framealpha=0.8,
                  loc="upper left")

def plot_parameter_figure(param_name: str, subdf: pd.DataFrame):
    schedulers_to_plot = ["Scenario-based"] if param_name.strip().lower() == "scenario count" else valid_sched
    dparam = subdf[subdf["Scheduler_type"].isin(schedulers_to_plot)].copy()

    # Wider and less tall figure
    fig = plt.figure(figsize=(20, 7))
    gs  = gridspec.GridSpec(2, 12, figure=fig)
    gs.update(left=0.04, right=0.98, top=0.90, bottom=0.10, wspace=1.0, hspace=0.35)

    # Top row: three plots spread wide
    ax1 = fig.add_subplot(gs[0, 0:4])
    ax2 = fig.add_subplot(gs[0, 4:8])
    ax3 = fig.add_subplot(gs[0, 8:12])
    # Bottom row: two centered
    ax4 = fig.add_subplot(gs[1, 2:6])
    ax5 = fig.add_subplot(gs[1, 6:10])
    axes = [ax1, ax2, ax3, ax4, ax5]

    for ax, (mean_col, std_col, title, ylab) in zip(axes, METRICS):
        _plot_metric_on_ax(ax, dparam, schedulers_to_plot, mean_col, std_col, title, ylab, param_name)

    fig.suptitle(f"Sensitivity — {param_name}", fontsize=16, y=0.95)
    plt.show()

# ---- Generate the 3 figures ----
for param_name in ["Trigger period", "EDD-FS priority weight", "Scenario count"]:
    sub = df[df["Parameter"].str.lower() == param_name.lower()]
    if sub.empty:
        print(f"[skip] No rows found for parameter '{param_name}'.")
        continue
    plot_parameter_figure(param_name, sub)

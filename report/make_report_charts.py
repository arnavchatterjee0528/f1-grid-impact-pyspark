"""Compact, print-sized versions of the four charts for the 4-page report (reads ../output/*.csv and metrics.json)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "output")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(DST, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5})


def style(ax, title):
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=6)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.5, length=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(DST, name), dpi=300, facecolor="white")
    plt.close(fig)


corr = pd.read_csv(os.path.join(OUT, "corr_by_decade.csv"))
fig, ax = plt.subplots(figsize=(3.3, 2.0))
x, y = corr["decade"], corr["grid_finish_corr"]
ax.plot(x, y, color=BLUE, lw=1.6, marker="o", ms=4, markeredgecolor="white", markeredgewidth=1)
ax.annotate(f"{y.iloc[0]:.2f}", (x.iloc[0], y.iloc[0]), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=6.5, color=INK)
ax.annotate(f"{y.iloc[-1]:.2f}", (x.iloc[-1], y.iloc[-1]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{str(d)[2:]}s" for d in x])
ax.set_ylim(0.4, 0.7); ax.yaxis.grid(True, color=GRID, lw=0.6)
style(ax, "(a) Grid–finish correlation (r) by decade")
save(fig, "fig_a_corr.png")

ph = pd.read_csv(os.path.join(OUT, "phase_gains.csv"))
ph["label"] = ["Lap 1", "Pit-stop laps", "Middle stint", "Final third"]
v = ph["avg_net_gain_per_driver_race"]
fig, ax = plt.subplots(figsize=(3.3, 2.0))
ax.bar(ph["label"], v, color=[BLUE if a >= 0 else ORANGE for a in v], width=0.55)
ax.axhline(0, color=MUTED, lw=0.8)
for i, a in enumerate(v):
    ax.text(i, a + (0.12 if a >= 0 else -0.12), f"{a:+.1f}", ha="center", va="bottom" if a >= 0 else "top", fontsize=6.5, color=INK)
ax.set_ylim(min(v) - 0.9, max(v) + 0.9); ax.yaxis.grid(True, color=GRID, lw=0.6)
style(ax, "(b) Net places gained per driver, by race phase")
save(fig, "fig_b_phase.png")

oi = pd.read_csv(os.path.join(OUT, "overtaking_index.csv")).sort_values("overtaking_index")
fig, ax = plt.subplots(figsize=(3.3, 3.0))
ax.barh(oi["country"], oi["overtaking_index"], color=[ORANGE if i < 3 else BLUE for i in range(len(oi))], height=0.68)
ax.invert_yaxis(); ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_xlabel("Avg places changed, grid to finish (2010–2024)", fontsize=6.5, color=MUTED)
style(ax, "(c) Circuit overtaking index (by host country)")
save(fig, "fig_c_overtaking.png")

m = json.load(open(os.path.join(OUT, "metrics.json")))
pred = pd.read_csv(os.path.join(OUT, "predictions_2023.csv"))
fig, ax = plt.subplots(figsize=(3.3, 3.0))
ax.plot([1, 22], [1, 22], ls="--", color=MUTED, lw=0.8)
ax.scatter(pred["label"], pred["prediction"], s=8, color=BLUE, alpha=0.45, edgecolors="white", linewidths=0.3)
ax.set_xlim(0, 23); ax.set_ylim(0, 23); ax.grid(True, color=GRID, lw=0.6)
ax.set_xlabel("Actual finishing position", fontsize=6.5, color=MUTED); ax.set_ylabel("Predicted", fontsize=6.5, color=MUTED)
ax.spines["left"].set_visible(True); ax.spines["left"].set_color(MUTED)
style(ax, f"(d) Random Forest on 2023: RMSE {m['random_forest']['rmse']:.2f}")
save(fig, "fig_d_pred.png")
print("figures written to", DST)

"""Train the departure-delay model and check it against simple baselines.

We train one gradient-boosted model per quantile, so for each person and
hangout we get a range (p10 ... p90) instead of a single guess. The alert uses
p80: "leave early enough that you're on time about 80% of the time."

Run:  python train.py
Out:  models/lateness.joblib, results/metrics.json, results/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, to_model_frame

ROOT = Path(__file__).parent
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9]
ALERT_Q = 0.8
ON_TIME_GRACE = 5  # minutes late that still counts as "on time"

# Chart colors (light theme)
BLUE, GRAY, INK, INK2, GRID, SURFACE = "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def train_models(X, y):
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
            max_leaf_nodes=15, min_samples_leaf=20, categorical_features="from_dtype",
            random_state=0,
        )
        models[q] = m.fit(X, y)
    return models


def predict_quantiles(models, X):
    preds = np.column_stack([models[q].predict(X) for q in QUANTILES])
    return np.sort(preds, axis=1)  # keep p10 <= p50 <= p90 (quantile models can cross)


def simulate_hangouts(test, buffer_col):
    """Everyone gets an alert at start - maps_travel - buffer.
    Lateness = departure delay - buffer + (actual travel - maps estimate)."""
    late = test["departure_delay"] - test[buffer_col] + (test["actual_travel_min"] - test["travel_min"])
    per_event = pd.DataFrame({"event_id": test["event_id"], "late": late})
    everyone_on_time = per_event.groupby("event_id")["late"].max() <= ON_TIME_GRACE
    return {
        "everyone_on_time_pct": round(100 * everyone_on_time.mean(), 1),
        "person_on_time_pct": round(100 * (late <= ON_TIME_GRACE).mean(), 1),
        "avg_min_late_when_late": round(late[late > ON_TIME_GRACE].mean(), 1),
        "avg_min_waiting_early": round((-late.clip(upper=0)).mean(), 1),
    }


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(ROOT / "data" / "hangouts.csv", parse_dates=["start"])

    # Split by time, not randomly: train on the past, test on the future,
    # which is how the model will be used for real.
    cutoff = df["start"].quantile(0.75)
    train, test = df[df.start <= cutoff].copy(), df[df.start > cutoff].copy()
    print(f"Train: {len(train)} rows up to {cutoff.date()} | Test: {len(test)} rows after")

    models = train_models(to_model_frame(train), train["departure_delay"])
    preds = predict_quantiles(models, to_model_frame(test))
    for i, q in enumerate(QUANTILES):
        test[f"p{int(q * 100)}"] = preds[:, i]
    y = test["departure_delay"]

    # 1) Accuracy of the typical (median) prediction vs. baselines
    mae = {
        "Maps only (assume on time)": np.abs(y - 0).mean(),
        "Personal average": np.abs(y - test["hist_mean"]).mean(),
        "Our model (p50)": np.abs(y - test["p50"]).mean(),
    }
    # The floor: even a model that knew every habit exactly can't predict the
    # random part of each day. Only measurable on synthetic data.
    if "no_noise_delay" in test:
        mae["Best possible (knows every habit)"] = np.abs(y - test["no_noise_delay"]).mean()
    # 2) Calibration: does p90 actually cover ~90% of real outcomes?
    calibration = {f"p{int(q * 100)}": round(100 * (y <= test[f"p{int(q * 100)}"]).mean(), 1) for q in QUANTILES}
    interval_80 = round(100 * ((y >= test.p10) & (y <= test.p90)).mean(), 1)

    # 3) Product metric: simulate alerts for every test hangout
    test["buf_maps"] = 0.0
    test["buf_personal"] = test["hist_mean"]
    test["buf_model"] = test["p80"]
    test["buf_flat"] = test["buf_model"].mean()  # same total extra time as the model, spread evenly
    sims = {
        "Maps only": simulate_hangouts(test, "buf_maps"),
        "Personal average": simulate_hangouts(test, "buf_personal"),
        f"Flat {test['buf_flat'].iloc[0]:.0f}-min buffer": simulate_hangouts(test, "buf_flat"),
        "Our model": simulate_hangouts(test, "buf_model"),
    }

    metrics = {
        "mae_minutes": {k: round(v, 2) for k, v in mae.items()},
        "calibration_pct_below": calibration,
        "p10_to_p90_coverage_pct": interval_80,
        "simulated_hangouts": sims,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"models": models, "quantiles": QUANTILES, "alert_q": ALERT_Q, "features": FEATURES,
                 "prior": float(df["departure_delay"].mean())}, ROOT / "models" / "lateness.joblib")

    make_charts(df, test, sims, calibration)
    print("Saved models/lateness.joblib and charts in results/")


def make_charts(df, test, sims, calibration):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})

    # Chart 1: the headline number, two panels on separate axes
    names = list(sims)
    colors = [BLUE if n == "Our model" else GRAY for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, key, title, fmt in [
        (axes[0], "everyone_on_time_pct", "Hangouts where everyone arrives on time (%)", "{:.0f}%"),
        (axes[1], "avg_min_waiting_early", "Avg minutes each person waits around early", "{:.1f}"),
    ]:
        style(ax)
        vals = [sims[n][key] for n in names]
        bars = ax.bar(names, vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle(f"Simulated on test hangouts (on time = within {ON_TIME_GRACE} min)", fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "baseline_comparison.png", dpi=160)
    plt.close(fig)

    # Chart 2: calibration
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    style(ax)
    nominal = [q * 100 for q in QUANTILES]
    observed = list(calibration.values())
    ax.plot([0, 100], [0, 100], color=GRAY, linewidth=1.5, linestyle="--", label="Perfect calibration")
    ax.plot(nominal, observed, color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="Our model")
    ax.set_xlabel("Predicted percentile")
    ax.set_ylabel("% of real delays at or below the prediction")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Are the ranges trustworthy?", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "calibration.png", dpi=160)
    plt.close(fig)

    # Chart 3: each friend's lateness profile (median + p10-p90 of real delays)
    stats = df.groupby(["user", "persona"]).departure_delay.quantile([0.1, 0.5, 0.9]).unstack().reset_index()
    stats = stats.sort_values(0.5)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
    ypos = np.arange(len(stats))
    ax.hlines(ypos, stats[0.1], stats[0.9], color=BLUE, alpha=0.35, linewidth=6)
    ax.scatter(stats[0.5], ypos, color=BLUE, s=60, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(ypos, [f"{u.title()}  ({p})" for u, p in zip(stats.user, stats.persona)], fontsize=9)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Minutes after the alert they actually leave")
    ax.set_title("Every friend runs late differently\n(dot = typical day, bar = 10th to 90th percentile)",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "lateness_profiles.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

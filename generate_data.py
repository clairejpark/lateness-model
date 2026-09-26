"""Generate a synthetic dataset of hangouts and how late each friend left.

Each friend is a "persona" with built-in habits. The model never sees the
persona labels; the point is to check that it can discover these habits
from behavior alone.

Target column: departure_delay = minutes after their "leave now" alert that a
person actually started moving (negative = left early).

Run:  python generate_data.py
Out:  data/hangouts.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import CONTEXT_FLAGS, history_features

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"

N_EVENTS = 700
START = pd.Timestamp("2025-09-01")
DAYS = 365

# base: usual delay (min). noise: how unpredictable they are.
# The other keys add minutes in specific situations.
PERSONAS = {
    "maya":   dict(kind="always early",     base=-3, noise=1.5),
    "jake":   dict(kind="chronically late", base=10, noise=6.0),
    "sofia":  dict(kind="morning struggler", base=1, noise=3.0, morning=15),
    "leo":    dict(kind="rain hater",       base=2, noise=3.0, rain=8),
    "priya":  dict(kind="overbooked",       base=2, noise=3.0, b2b=12, b2b_rate=0.55),
    "ethan":  dict(kind="improving",        base=14, noise=4.0, improve_to=2),
    "hana":   dict(kind="steady",           base=3, noise=2.0),
    "marcus": dict(kind="party procrastinator", base=3, noise=3.5, party=10),
    "zoe":    dict(kind="forgets far-off plans", base=2, noise=3.0, far_ahead=9),
    "omar":   dict(kind="average",          base=4, noise=4.0),
}
USERS = list(PERSONAS)

# Hangouts from 8am to 10pm, more common in the evening
HOURS = np.arange(8, 23)
HOUR_WEIGHTS = np.r_[[1] * 4, [2] * 5, [3.5] * 6]
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


def simulate_delay(p, ev, progress):
    """Minutes after the alert this person actually leaves."""
    base = p["base"]
    if "improve_to" in p:  # gets better over the year
        base = p["base"] + (p["improve_to"] - p["base"]) * progress
    d = base
    d += p.get("morning", 0) * ev["is_morning"]
    d += p.get("rain", 0) * ev["is_raining"]
    d += p.get("b2b", 0) * ev["back_to_back"]
    d += p.get("party", 0) * ev["is_party"]
    d += p.get("far_ahead", 0) * (ev["lead_time_hours"] > 72)
    # Effects shared by everyone
    d += 2.0 * ev["is_party"] + 0.5 * max(ev["group_size"] - 3, 0) + 1.5 * ev["is_raining"]
    no_noise = d  # what a model with perfect knowledge of every habit would predict
    # Lopsided noise: usually small, occasionally very late
    d += p["noise"] * (np.exp(rng.normal(0, 0.6)) - 1)
    return max(d, -10.0), no_noise


def simulate_travel(mode, planned):
    """Actual travel minutes vs. the Maps estimate."""
    sigma = {"walk": 0.05, "drive": 0.15, "transit": 0.25}[mode]
    return planned * np.exp(rng.normal(0, sigma))


def main():
    rows = []
    for event_id in range(N_EVENTS):
        day = START + pd.Timedelta(days=int(rng.integers(0, DAYS)))
        hour = int(rng.choice(HOURS, p=HOUR_WEIGHTS))
        start = day + pd.Timedelta(hours=hour)
        ev_type = rng.choice(["food", "study", "party", "hangout"], p=[0.35, 0.2, 0.15, 0.3])
        event = dict(
            event_id=event_id,
            start=start,
            hour=hour,
            day_of_week=start.dayofweek,
            is_weekend=int(start.dayofweek >= 5),
            is_morning=int(hour < 10),
            lead_time_hours=float(rng.choice([2, 6, 24, 48, 96, 168], p=[0.15, 0.2, 0.25, 0.2, 0.1, 0.1])),
            event_type=ev_type,
            is_party=int(ev_type == "party"),
            is_raining=int(rng.random() < 0.2),
        )
        group = rng.choice(USERS, size=int(rng.integers(2, 7)), replace=False)
        event["group_size"] = len(group)
        for user in group:
            p = PERSONAS[user]
            travel = float(rng.uniform(5, 40))
            mode = "walk" if travel < 15 else rng.choice(["drive", "transit"])
            row = dict(event, user=user, persona=p["kind"], travel_min=round(travel, 1), travel_mode=mode,
                       back_to_back=int(rng.random() < p.get("b2b_rate", 0.25)))
            progress = (start - START).days / DAYS
            row["far_ahead"] = int(row["lead_time_hours"] > 72)
            delay, no_noise = simulate_delay(p, row, progress)
            row["departure_delay"] = round(delay, 1)
            row["no_noise_delay"] = round(no_noise, 1)  # for measuring the best possible score, never a feature
            row["actual_travel_min"] = round(simulate_travel(mode, travel), 1)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["start", "event_id", "user"]).reset_index(drop=True)

    # History features, using ONLY hangouts that happened before each row
    # (no peeking at the future, same as the real app).
    past_by_user = {u: [] for u in USERS}
    all_past = []
    hist_rows = []
    for i, r in df.iterrows():
        prior = float(np.mean(all_past)) if all_past else 3.0
        current = {f: int(r[f]) for f in CONTEXT_FLAGS}
        hist_rows.append(history_features(past_by_user[r.user], current, prior))
        past_by_user[r.user].append({"delay": r.departure_delay, **current})
        # Update the group average only once the whole hangout is done, so
        # friends at the same hangout don't leak into each other's features.
        if i + 1 == len(df) or df.at[i + 1, "event_id"] != r.event_id:
            all_past.extend(df.loc[df.event_id == r.event_id, "departure_delay"].tolist())

    df = pd.concat([df, pd.DataFrame(hist_rows)], axis=1)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "hangouts.csv", index=False)
    print(f"Wrote {len(df)} rows ({df.event_id.nunique()} hangouts, {len(USERS)} friends) to data/hangouts.csv")
    print(df.groupby("persona").departure_delay.describe()[["mean", "50%", "max"]].round(1))


if __name__ == "__main__":
    main()

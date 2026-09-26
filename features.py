"""Feature engineering shared by data generation, training, and prediction.

The model never sees a user ID. Instead it sees a summary of that person's
past behavior ("history features"). That way it works for brand-new users
too: with no history, the features fall back to the group average.
"""
import numpy as np

# Situations where people's lateness often changes. For each one, we measure
# how this person usually behaves in the same situation.
# Rule of thumb: the model can only learn a habit if some feature can reveal it.
# (far_ahead = planned more than 3 days before. Adding it let the model finally
# notice friends who forget about plans made long in advance.)
CONTEXT_FLAGS = ["is_raining", "back_to_back", "is_morning", "is_party", "far_ahead"]

# How many hangouts of evidence before we trust someone's personal average
# over the group average. estimate = (n * their_avg + K * group_avg) / (n + K)
K = 5
DEFAULT_STD = 5.0  # minutes; used until someone has a few hangouts

EVENT_TYPES = ["food", "study", "party", "hangout"]
TRAVEL_MODES = ["walk", "drive", "transit"]

CONTEXT_FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_morning", "lead_time_hours",
    "group_size", "event_type", "travel_min", "travel_mode",
    "is_raining", "back_to_back", "is_party",
]
HISTORY_FEATURES = (
    ["hist_n", "hist_mean", "hist_std", "hist_recent"]
    + [f"hist_ctx_{f}" for f in CONTEXT_FLAGS]
)
FEATURES = CONTEXT_FEATURES + HISTORY_FEATURES
CATEGORICAL = ["event_type", "travel_mode"]


def shrink(total, n, prior, k=K):
    """Blend a personal average with a prior, trusting it more as n grows."""
    return (total + k * prior) / (n + k)


def history_features(past, current, prior):
    """Summarize one person's past hangouts.

    past:    list of dicts (oldest first), each with "delay" and the CONTEXT_FLAGS
    current: dict with the CONTEXT_FLAGS for the upcoming hangout
    prior:   group-wide average departure delay (minutes)
    """
    delays = np.array([p["delay"] for p in past], dtype=float)
    n = len(delays)
    mean = shrink(delays.sum(), n, prior)
    feats = {
        "hist_n": n,
        "hist_mean": mean,
        "hist_std": delays.std() if n >= 3 else DEFAULT_STD,
    }
    last5 = delays[-5:]
    feats["hist_recent"] = shrink(last5.sum(), len(last5), mean, k=2)

    # Personal effect of each situation: how much later (or earlier) than their
    # own average this person is when the situation matches the upcoming hangout.
    for flag in CONTEXT_FLAGS:
        same = np.array([p["delay"] for p in past if p[flag] == current[flag]])
        feats[f"hist_ctx_{flag}"] = shrink(same.sum(), len(same), mean) - mean
    return feats


def to_model_frame(df):
    """Select model columns and set categorical dtypes consistently."""
    import pandas as pd
    X = df[FEATURES].copy()
    X["event_type"] = pd.Categorical(X["event_type"], categories=EVENT_TYPES)
    X["travel_mode"] = pd.Categorical(X["travel_mode"], categories=TRAVEL_MODES)
    return X

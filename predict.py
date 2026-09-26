"""The function the backend calls to decide when to send someone's "leave now" alert.

    from predict import predict_departure
    result = predict_departure(past_hangouts, event)
    result["leave_at"]   -> when to send the alert
    result["message"]    -> notification text, with the reasons

Run `python predict.py` for a demo on the synthetic friends.
"""
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import CONTEXT_FLAGS, history_features, to_model_frame

ROOT = Path(__file__).parent
_bundle = None

# Situations we can explain. Each entry: how to "turn the situation off" to
# measure how many minutes it adds for this person.
REASONS = {
    "back_to_back": ("you're coming from another event", lambda e: {**e, "back_to_back": 0}),
    "is_raining":   ("it's raining", lambda e: {**e, "is_raining": 0}),
    "is_morning":   ("it's a morning hangout", lambda e: {**e, "start": e["start"].replace(hour=13)}),
    "is_party":     ("it's a party", lambda e: {**e, "event_type": "hangout"}),
    "far_ahead":    ("this was planned days ago", lambda e: {**e, "lead_time_hours": 24}),
}


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(ROOT / "models" / "lateness.joblib")
    return _bundle


def _feature_row(past_hangouts, event, prior):
    start = event["start"]
    row = {
        "hour": start.hour,
        "day_of_week": start.weekday(),
        "is_weekend": int(start.weekday() >= 5),
        "is_morning": int(start.hour < 10),
        "lead_time_hours": event["lead_time_hours"],
        "far_ahead": int(event["lead_time_hours"] > 72),
        "group_size": event["group_size"],
        "event_type": event["event_type"],
        "is_party": int(event["event_type"] == "party"),
        "travel_min": event["travel_min"],
        "travel_mode": event["travel_mode"],
        "is_raining": int(event["is_raining"]),
        "back_to_back": int(event["back_to_back"]),
    }
    row.update(history_features(past_hangouts, {f: row[f] for f in CONTEXT_FLAGS}, prior))
    return row


def _quantiles(past_hangouts, event):
    b = _load()
    X = to_model_frame(pd.DataFrame([_feature_row(past_hangouts, event, b["prior"])]))
    preds = np.sort([b["models"][q].predict(X)[0] for q in b["quantiles"]])
    return dict(zip(b["quantiles"], preds))


def predict_departure(past_hangouts, event):
    """
    past_hangouts: this person's previous hangouts, oldest first. Each is a dict:
        {"delay": minutes_after_alert_they_left, "is_raining": 0/1,
         "back_to_back": 0/1, "is_morning": 0/1, "is_party": 0/1, "far_ahead": 0/1}
        An empty list is fine (new user -> group average).
    event: {"start": datetime, "travel_min": float (from Maps), "travel_mode": "walk"|"drive"|"transit",
            "event_type": "food"|"study"|"party"|"hangout", "group_size": int,
            "lead_time_hours": float, "is_raining": 0/1, "back_to_back": 0/1}
    """
    alert_q = _load()["alert_q"]
    q = _quantiles(past_hangouts, event)
    buffer = max(q[alert_q], 0.0)  # never tell someone to leave *later* than Maps says
    maps_leave = event["start"] - timedelta(minutes=event["travel_min"])
    leave_at = maps_leave - timedelta(minutes=buffer)

    # Explain the alert: for each situation that applies, predict again as if it
    # didn't, and report the difference.
    active = {
        "back_to_back": event["back_to_back"], "is_raining": event["is_raining"],
        "is_morning": event["start"].hour < 10, "is_party": event["event_type"] == "party",
        "far_ahead": event["lead_time_hours"] > 72,
    }
    reasons = []
    for key, (label, turn_off) in REASONS.items():
        if active[key]:
            minutes = q[alert_q] - _quantiles(past_hangouts, turn_off(event))[alert_q]
            if minutes >= 1:
                reasons.append((label, round(minutes)))
    reasons.sort(key=lambda r: -r[1])

    earlier = round(buffer)
    msg = f"🚶 Leave by {leave_at:%-I:%M %p}"
    if earlier >= 1:
        msg += f", {earlier} min before Maps would tell you"
        if reasons:
            msg += ": " + ", ".join(f"{label} (+{m})" for label, m in reasons)
        elif len(past_hangouts) < 3:
            msg += " (you're new, so this uses the group's average)"
        else:
            msg += f", since you usually head out about {max(round(q[0.5]), 1)} min after your alert"
    msg += "."

    return {
        "departure_delay_p10": round(q[0.1], 1),
        "departure_delay_p50": round(q[0.5], 1),
        "departure_delay_p90": round(q[0.9], 1),
        "buffer_min": round(buffer, 1),
        "leave_at": leave_at,
        "reasons": reasons,
        "message": msg,
    }


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "hangouts.csv", parse_dates=["start"])

    def history(user):
        rows = df[df.user == user].sort_values("start")
        return [{"delay": r.departure_delay, **{f: int(r[f]) for f in CONTEXT_FLAGS}} for _, r in rows.iterrows()]

    tonight = datetime(2026, 9, 26, 19, 0)
    base_event = dict(start=tonight, travel_min=18, travel_mode="drive", event_type="food",
                      group_size=4, lead_time_hours=24, is_raining=0, back_to_back=0)
    demos = [
        ("Maya (always early), dinner at 7", "maya", base_event),
        ("Leo (rain hater), dinner at 7, raining", "leo", {**base_event, "is_raining": 1}),
        ("Priya (overbooked), coming straight from class", "priya", {**base_event, "back_to_back": 1}),
        ("Jake (chronically late), party at 7", "jake", {**base_event, "event_type": "party"}),
        ("Brand-new user, no history", None, base_event),
    ]
    for title, user, ev in demos:
        r = predict_departure(history(user) if user else [], ev)
        print(f"\n{title}")
        print(f"  predicted departure delay: typical {r['departure_delay_p50']} min, "
              f"range {r['departure_delay_p10']} to {r['departure_delay_p90']} min")
        print(f"  {r['message']}")

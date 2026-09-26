"""Web API around the lateness model. This is the file the backend person owns.

predict.py only does the math. It never calls the internet. This file:
  1. receives a request from the app,
  2. calls the outside APIs (Google Maps for travel time, Open-Meteo for rain),
  3. looks up the person's past hangouts,
  4. calls predict_departure() and sends the answer back.

Setup:
    pip install -r requirements.txt
    cp .env.example .env        # Windows: copy .env.example .env
    # open .env and paste your Google Maps key
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to try the endpoints in your browser.

No Maps key yet? It still runs: travel time falls back to a rough
straight-line estimate so the rest of the team isn't blocked.
"""
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features import CONTEXT_FLAGS
from predict import predict_departure

# ---------------------------------------------------------------------------
# 1) API KEYS: they live in the .env file next to this one, never in the code.
#    .env is listed in .gitignore so it never gets pushed to GitHub.
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
# Open-Meteo (weather) is free and needs no key.

app = FastAPI(title="Lateness model API")


# ---------------------------------------------------------------------------
# 2) OUTSIDE APIS
# ---------------------------------------------------------------------------
class LatLng(BaseModel):
    lat: float
    lng: float


def get_travel_minutes(origin: LatLng, dest: LatLng, mode: str, start: datetime) -> float:
    """Travel time from Google Maps (Routes API). Enable "Routes API" in Google Cloud."""
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == "paste-your-key-here":
        return _rough_travel_minutes(origin, dest, mode)
    body = {
        "origin": {"location": {"latLng": {"latitude": origin.lat, "longitude": origin.lng}}},
        "destination": {"location": {"latLng": {"latitude": dest.lat, "longitude": dest.lng}}},
        "travelMode": {"walk": "WALK", "drive": "DRIVE", "transit": "TRANSIT"}[mode],
    }
    if mode == "drive":
        body["routingPreference"] = "TRAFFIC_AWARE"
    if mode == "transit":
        body["arrivalTime"] = start.isoformat()
    r = httpx.post(
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        json=body,
        headers={"X-Goog-Api-Key": GOOGLE_MAPS_API_KEY, "X-Goog-FieldMask": "routes.duration"},
        timeout=10,
    )
    r.raise_for_status()
    routes = r.json().get("routes")
    if not routes:
        raise HTTPException(422, "Google Maps found no route between those points")
    return int(routes[0]["duration"].rstrip("s")) / 60  # "1234s" -> minutes


def _rough_travel_minutes(origin: LatLng, dest: LatLng, mode: str) -> float:
    """Straight-line fallback for development without a Maps key."""
    lat1, lat2 = math.radians(origin.lat), math.radians(dest.lat)
    dlat, dlng = lat2 - lat1, math.radians(dest.lng - origin.lng)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    km = 2 * 6371 * math.asin(math.sqrt(a)) * 1.3  # roads aren't straight
    kmh = {"walk": 5, "drive": 30, "transit": 18}[mode]
    return max(km / kmh * 60, 2)


def get_is_raining(place: LatLng, start: datetime) -> int:
    """Rain forecast for the hangout's hour from Open-Meteo (free, no key)."""
    try:
        r = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": place.lat, "longitude": place.lng, "hourly": "precipitation",
                    "timezone": "auto", "start_date": start.date().isoformat(),
                    "end_date": start.date().isoformat()},
            timeout=10,
        )
        r.raise_for_status()
        hourly = r.json()["hourly"]
        hour_key = start.strftime("%Y-%m-%dT%H:00")
        return int(hourly["precipitation"][hourly["time"].index(hour_key)] >= 0.2)  # mm in that hour
    except Exception:
        return 0  # if the weather call fails, don't block the alert


# ---------------------------------------------------------------------------
# 3) STORAGE: replace these two dicts with your real database
# ---------------------------------------------------------------------------
HISTORY: dict[str, list[dict]] = {}  # user_id -> past hangouts, oldest first
PENDING: dict[tuple[str, str], dict] = {}  # (hangout_id, user_id) -> situation flags, until they leave


def get_past_hangouts(user_id: str) -> list[dict]:
    return HISTORY.get(user_id, [])


def save_hangout(user_id: str, record: dict) -> None:
    HISTORY.setdefault(user_id, []).append(record)


@app.on_event("startup")
def load_demo_friends():
    """Load the synthetic friends (maya, jake, leo, ...) so the demo works right away."""
    csv = Path(__file__).parent / "data" / "hangouts.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv, parse_dates=["start"]).sort_values("start")
        for user, rows in df.groupby("user"):
            HISTORY[user] = [{"delay": r["departure_delay"], **{f: int(r[f]) for f in CONTEXT_FLAGS}}
                             for r in rows.to_dict("records")]


# ---------------------------------------------------------------------------
# 4) ENDPOINTS: what the app calls
# ---------------------------------------------------------------------------
class DepartureRequest(BaseModel):
    hangout_id: str
    user_id: str
    start: datetime  # hangout start, local time with offset, e.g. "2026-09-26T19:00:00-04:00"
    planned_at: datetime  # when the hangout was created
    origin: LatLng  # where this person is coming from
    destination: LatLng
    travel_mode: Literal["walk", "drive", "transit"]
    event_type: Literal["food", "study", "party", "hangout"]
    group_size: int
    back_to_back: bool = False  # true if they have a calendar event ending within ~30 min of leaving


@app.post("/predict-departure")
def predict(req: DepartureRequest):
    """Call this when tracking starts (about an hour before). Schedule the push
    notification for `leave_at` and show `message` in it."""
    travel_min = get_travel_minutes(req.origin, req.destination, req.travel_mode, req.start)
    event = {
        "start": req.start,
        "travel_min": travel_min,
        "travel_mode": req.travel_mode,
        "event_type": req.event_type,
        "group_size": req.group_size,
        "lead_time_hours": (req.start - req.planned_at).total_seconds() / 3600,
        "is_raining": get_is_raining(req.destination, req.start),
        "back_to_back": int(req.back_to_back),
    }
    result = predict_departure(get_past_hangouts(req.user_id), event)
    leave_at = result["leave_at"].replace(second=0, microsecond=0)  # round down to the minute

    # Remember the situation so we can learn from what actually happens
    PENDING[(req.hangout_id, req.user_id)] = {
        "is_raining": event["is_raining"], "back_to_back": event["back_to_back"],
        "is_morning": int(req.start.hour < 10), "is_party": int(req.event_type == "party"),
        "far_ahead": int(event["lead_time_hours"] > 72),
        "alert_at": leave_at,
    }
    return {
        "leave_at": leave_at.isoformat(),
        "message": result["message"],
        "travel_min": round(travel_min, 1),
        "is_raining": bool(event["is_raining"]),
        "departure_delay_range_min": [float(result["departure_delay_p10"]), float(result["departure_delay_p90"])],
        "reasons": [{"reason": r, "minutes": m} for r, m in result["reasons"]],
    }


class DepartedRequest(BaseModel):
    hangout_id: str
    user_id: str
    departed_at: datetime  # when location tracking saw them start moving


@app.post("/record-departure")
def record_departure(req: DepartedRequest):
    """Call this when tracking sees someone start moving. This is how the model
    learns each person's habits over time."""
    pending = PENDING.pop((req.hangout_id, req.user_id), None)
    if pending is None:
        raise HTTPException(404, "No prediction found for this hangout and user")
    alert_at = pending.pop("alert_at")
    delay = (req.departed_at - alert_at).total_seconds() / 60
    save_hangout(req.user_id, {"delay": round(delay, 1), **pending})
    return {"departure_delay_min": round(delay, 1), "hangouts_on_record": len(get_past_hangouts(req.user_id))}

@app.get("/history/{user_id}")
def history(user_id: str):
    past = get_past_hangouts(user_id)
    return {"user_id": user_id, "hangouts": len(past), "last_5": past[-5:]}
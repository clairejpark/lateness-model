# Personal lateness model

Predicts how many minutes after their "leave now" alert each friend actually
leaves, so each person gets their own alert time. Maps knows the traffic; this
knows your friends.

## Run it (about 15 seconds total)

```bash
pip install -r requirements.txt
python generate_data.py   # makes data/hangouts.csv (synthetic friends and hangouts)
python train.py           # trains, compares to baselines, saves models/ and results/
python predict.py         # demo notifications for a few friends

# Backend: run the web API
cp .env.example .env      # Windows: copy .env.example .env
                          # then open .env and paste your Google Maps key
uvicorn api:app --reload  # then open http://127.0.0.1:8000/docs
```

## Files

| File | What it does |
|---|---|
| `features.py` | Turns a person's past hangouts into model inputs. Shared by everything else. |
| `generate_data.py` | 10 synthetic friends with built-in habits (rain hater, morning struggler, ...) and 700 hangouts over a year. |
| `train.py` | Trains one model per percentile (p10 to p90), tests on the most recent 25% of hangouts, makes the charts. |
| `predict.py` | `predict_departure(past_hangouts, event)`: the model math. Returns the alert time and the notification text. Makes no internet calls. |
| `api.py` | The web API the app calls. Gets travel time from Google Maps and rain from Open-Meteo, stores each person's history, and calls `predict_departure`. |

## How it works

- **Target:** departure delay = minutes between the alert and when someone starts moving. Travel time comes from Maps; we don't model it.
- **No user IDs in the model.** It sees a summary of each person's history instead (their average, spread, recent trend, and how they act in the rain, when coming from another event, in the morning, at parties). That's why it works for new users: with no history, those features fall back to the group average.
- **Ranges, not one number.** One model per percentile. The alert uses p80, meaning "leave early enough that you're on time about 80% of the time."
- **Explanations.** For each situation that applies, `predict.py` predicts again as if it didn't and reports the difference, e.g. "it's raining (+10)".

## Current results (synthetic test set, see `results/metrics.json`)

| Approach | Everyone on time (within 5 min) | Avg min waiting early |
|---|---|---|
| Maps only | 12% | 0.5 |
| Personal average | 44% | 2.6 |
| Flat buffer (same total extra time as model) | 49% | 3.5 |
| **Our model** | **61%** | **2.8** |

Median prediction error: 2.1 min (vs 3.6 for personal average, 6.4 for Maps only).

**Best possible score: 1.8 min.** The fake data includes random day-to-day noise that no model can predict. A model that knew every friend's habits exactly would still be off by 1.8 min on average, so the model has closed about 93% of the gap between Maps only and perfect.

The flat-buffer row is the fairest comparison: it gives everyone the same
total extra time as the model, spread evenly. The model beats it because it
gives the extra time to the people who need it.

**Known weak spot:** the low end of the range is off (24% of real delays fall
below p10 instead of 10%), so the p10 to p90 range covers 66% instead of 80%.
The upper end, which the alert uses, is well calibrated (p80 covers 79%, p90 covers 89%).

## Ideas for next steps

1. **Live update during the trip.** Add "minutes since alert without moving" as a feature and retrain, so the arrival estimate updates when someone hasn't left yet.
2. **Tune the alert percentile** (p75 vs p80 vs p85) and show the trade-off between on-time rate and waiting time.
3. **Replace the in-memory storage in `api.py`** (`HISTORY`, `PENDING`) with a real database so history survives restarts.
4. Optional: swap in LightGBM (`pip install lightgbm`, `LGBMRegressor(objective="quantile", alpha=q)`) if you want. scikit-learn's version works the same way and needs no extra install.

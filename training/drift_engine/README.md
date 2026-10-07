# GarbhaAI drift engine (Phase 8, runbook Step 16)

Two scripts:

- **`compute_baseline.py`** -- run once per model promotion (after
  `training/promote_engine/promote.py` moves the `production` alias).
  Walks the full Day3+Day4 training dataset, extracts 512-dim embeddings
  via the feature-extraction service, and logs a training-set baseline
  (the raw embeddings, plus per-dimension decile bucket edges/proportions)
  as an MLflow artifact on the current production run.
- **`detect_drift.py`** -- run on a schedule (see below). Pulls a sample
  of recently-graded images' embeddings out of the Postgres `image_features` table (the feature store),
  downloads the baseline the last `compute_baseline.py` run attached to
  the current production model, and compares them with two statistics:
  PSI (per-dimension population shift) and MMD (a kernel two-sample test
  for a joint-distribution shift, with a permutation-test p-value rather
  than a hand-picked threshold). Logs every run to a `drift_results`
  Postgres table. On `DRIFT_SUSTAINED_WINDOW` (default 3) consecutive
  flagged runs, publishes one alert to the `garbhaai:drift_alerts` Redis
  pub/sub channel -- the actual retrain decision stays a manual step a
  human acts on, per the runbook's own caution about not wiring the
  feedback loop closed until the rest of this pipeline has proven stable.

## Setup

```
infra/local && ./bootstrap.sh         # mlflow, postgres, rustfs, redis
cd ../../services && docker compose up -d   # feature-extraction, etc.
cd ../training/drift_engine
pip install -r requirements.txt

python compute_baseline.py    # once, after the first promote.py run
python detect_drift.py        # once, by hand, to confirm it runs clean
```

`detect_drift.py` needs real live traffic in the `image_features` table to do anything useful
-- send a handful of real `/grade` requests through `services/inference`
first (each one stores its embedding via `_push_features()`), or it will
print `SKIPPED: only N live embeddings available` and log a skipped row
rather than failing.

## Scheduling: cron, not GitHub Actions

The runbook names a scheduled GitHub Actions workflow as the other
zero-cost option. It doesn't work here: a GitHub-hosted runner is a
machine on GitHub's infrastructure, not this Mac, so it has no way to
reach `localhost:5500` (MLflow), `localhost:5432` (Postgres), or
`localhost:6379` (Redis) -- every one of which is published only to this
machine's loopback interface by `infra/local/docker-compose.yml`. A
self-hosted Actions runner or a tunnel would fix that, but at that point
it's extra infrastructure doing exactly what cron already does for free
on the machine that can already reach everything. Hence `run_drift_check.sh`
+ cron.

Add it to your crontab (`crontab -e`), e.g. every 4 hours:

```
0 */4 * * * /Users/hrishikeshwadki/Claude/Projects/FinalEndProjectISBAMPBA/garbhaai-platform/training/drift_engine/run_drift_check.sh >> /Users/hrishikeshwadki/Claude/Projects/FinalEndProjectISBAMPBA/garbhaai-platform/training/drift_engine/logs/cron.log 2>&1
```

Logs also land per-run under `training/drift_engine/logs/drift_check_<timestamp>.log`
(the script keeps the most recent 200 and prunes older ones itself); the
crontab redirect above additionally captures anything cron itself can't
hand off cleanly (e.g. the script failing before it opens its own log
file).

If `detect_drift.py`'s dependencies live in a venv rather than the system
Python, point the wrapper at it once:

```
PYTHON_BIN="$HOME/.venvs/<your-venv-name>/bin/python3" training/drift_engine/run_drift_check.sh   # test by hand first
```

then bake that same `PYTHON_BIN=...` prefix into the crontab line.

## Checking it's working

```
psql -h localhost -U garbhaai -d garbhaai -c \
  "SELECT checked_at, n_live_samples, mean_psi, mmd_p_value, drift_flagged, sustained_alert_fired FROM drift_results ORDER BY checked_at DESC LIMIT 10;"
```

(password `garbhaai_local_dev`, matching `infra/local/docker-compose.yml`.)

To watch for an alert being published without waiting for real sustained
drift, subscribe to the channel in one terminal (`redis-cli subscribe
garbhaai:drift_alerts`) while a run that happens to flag three times in a
row (or a deliberately corrupted/offset `DRIFT_PSI_THRESHOLD` for testing)
fires in another.

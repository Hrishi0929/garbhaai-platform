#!/usr/bin/env bash
# Phase 8, Task #28: the cron entry point for detect_drift.py.
#
# Why cron and not a scheduled GitHub Actions workflow (the runbook's other
# free option): a GitHub-hosted runner is a machine somewhere else on the
# internet -- it cannot reach http://localhost:5500 (MLflow),
# localhost:5432 (Postgres), or localhost:6379 (Redis) on this Mac, which
# is where this whole platform runs (infra/local/docker-compose.yml, all
# host-port-published). Every other option the runbook allows (a
# self-hosted runner, a tunnel) adds real infrastructure for no benefit
# over cron, which is already sitting on the machine that can actually
# reach these services. See training/drift_engine/README.md.
#
# This script, not a bare `python detect_drift.py` crontab line, exists
# because cron runs with almost none of an interactive shell's
# environment: no PATH beyond a minimal default, no activated venv, and a
# different cwd. Setting both explicitly here is what makes the cron job
# behave the same as running this by hand from a terminal.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Point this at your venv's python if detect_drift.py's dependencies
# (requirements.txt) aren't on the system python's path. Override by
# exporting PYTHON_BIN before calling this script, or editing the default
# below once -- e.g. PYTHON_BIN="$HOME/.venvs/garbhaai/bin/python3".
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/drift_check_$(date -u +%Y%m%dT%H%M%SZ).log"

# infra/local's docker-compose services publish to localhost on this Mac
# (not host.docker.internal -- that's only needed from inside the kind
# cluster, see infra/k8s/inference/rollout.yaml's comment). This script
# runs directly on the host, same as every other training/ script, so the
# localhost defaults baked into detect_drift.py's env.get() calls are
# correct as-is and nothing needs setting here unless your setup differs.

# Plain `date -u +%Y-%m-%dT%H:%M:%SZ` rather than GNU date's
# --iso-8601=seconds -- this runs on macOS (BSD date), which doesn't have
# that flag.
echo "=== drift check started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG_FILE"
"$PYTHON_BIN" detect_drift.py 2>&1 | tee -a "$LOG_FILE"
exit_code="${PIPESTATUS[0]}"
echo "=== drift check finished (exit $exit_code) ===" | tee -a "$LOG_FILE"

# Keep the last 200 log files so this doesn't grow unbounded on a Mac
# nobody's logging into daily -- a drift check running every few hours
# would otherwise accumulate thousands of files over the life of this
# capstone project. xargs here (not GNU xargs's -r/--no-run-if-empty,
# which macOS's BSD xargs doesn't have) guards the empty-input case
# explicitly instead.
OLD_LOGS="$(ls -1t "$LOG_DIR"/drift_check_*.log 2>/dev/null | tail -n +201)"
if [ -n "$OLD_LOGS" ]; then
  echo "$OLD_LOGS" | xargs rm --
fi

exit "$exit_code"

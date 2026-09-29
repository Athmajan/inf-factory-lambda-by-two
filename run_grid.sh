#!/usr/bin/env bash
# Launch N detached grid workers (they survive the terminal / Claude session ending).
# Usage: ./run_grid.sh [workers=4] [hours=24]
# Progress: ./status.sh        Stop: ./stop_grid.sh (finished tiles are kept; restart resumes)
set -euo pipefail
cd "$(dirname "$0")"
workers=${1:-4}
hours=${2:-24}
PY=${PY:-.venv/bin/python}
if pgrep -f 'scripts/run_cir_grid\.py' >/dev/null; then
  echo "grid workers are already running:" >&2
  pgrep -af 'scripts/run_cir_grid\.py' >&2
  exit 1
fi
mkdir -p logs results/cir_grid_lam2
# Build the manifest once before the workers start (fast, no GPU).
"$PY" scripts/run_cir_grid.py --dry_run >/dev/null
for ((k = 0; k < workers; k++)); do
  setsid nohup "$PY" scripts/run_cir_grid.py --hours "$hours" --shard "$k" --num_shards "$workers" \
    >> "logs/cir_grid_w${k}of${workers}.log" 2>&1 < /dev/null &
  echo "worker $k/$workers: PID $! -> logs/cir_grid_w${k}of${workers}.log"
done

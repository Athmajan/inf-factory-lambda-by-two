#!/usr/bin/env bash
# Progress of the lambda/2 grid: tiles/points done, worker processes, last log line per worker, GPU.
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
"$PY" - <<'EOF'
import json, os
m = json.load(open("results/cir_grid_lam2/manifest.json"))
c = m["points_per_tile"]
done = [k for k in c if c[k] and os.path.exists(f"results/cir_grid_lam2/tiles/tile_{k}.npz")]
pts = sum(c[k] for k in done)
print(f"tiles {len(done)}/{sum(1 for v in c.values() if v)}, points {pts:,}/{m['n_points_total']:,} "
      f"({100 * pts / m['n_points_total']:.2f}%)")
EOF
echo "workers running: $(pgrep -fc 'scripts/run_cir_grid\.py' || true)"
for f in logs/cir_grid_w*.log; do [ -f "$f" ] && echo "$f: $(tail -1 "$f")"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
du -sh results/cir_grid_lam2/tiles 2>/dev/null
avail_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${avail_gb} GB"
[ "${avail_gb:-0}" -lt 10 ] && echo "WARNING: less than 10 GB free disk - stop the workers (./stop_grid.sh) and flag the human"

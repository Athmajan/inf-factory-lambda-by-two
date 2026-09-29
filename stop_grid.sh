#!/usr/bin/env bash
# Stop all grid workers. The tile being solved is lost; every finished tile is kept and a restart resumes.
pids=$(pgrep -f 'scripts/run_cir_grid\.py' || true)
[ -z "$pids" ] && { echo "no grid workers running"; exit 0; }
echo "stopping: $pids"
kill $pids

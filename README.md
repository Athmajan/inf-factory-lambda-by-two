# inf-factory-lambda-by-two

λ/2 (3.89 cm) grid of Sionna RT channel impulse responses over the reachable floor of a 3GPP InF-SL factory hall
(3.85 GHz, one TX, isotropic antennas): 2,901,201 points, 200 strongest paths each, resumable 2 m × 2 m tiles.

Quick start (details and rules: **CLAUDE.md**):
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/check_env.py          # versions, CUDA variant, agreement with the original host, rate
.venv/bin/python scripts/run_cir_grid.py --dry_run   # must print 2,901,201 points in 1380 tiles
./run_grid.sh 4 24                             # 4 detached workers for 24 h
./status.sh                                    # progress
./stop_grid.sh                                 # stop (resumable)
```
Output: `results/cir_grid_lam2/{manifest.json,tiles/tile_<ix>_<iy>.npz}` with `pos, grid_idx, a, tau, phi,
theta, dropped` per point.

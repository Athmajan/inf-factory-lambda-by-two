# CLAUDE.md — λ/2 ray-traced CIR grid of the InF-SL factory hall (run on the GH200)

You are resuming a job that was prepared on another machine (x86_64 host, RTX PRO 6000) and moved here, a
Grace Hopper GH200 (ARM64 CPU + Hopper GPU), to run for a full day or more. Read this whole file before doing
anything. Report plainly, keep the user informed, and **ask before deviating from these steps** (the user's rule).

## What this job is
Sionna RT ray tracing of a 3GPP TR 38.901 InF-SL factory hall (120 × 60 × 10 m, 30% metal clutter, concrete
ceiling, one isotropic TX at (60, 30, 1.5) m, 3.85 GHz) on a fixed floor grid:
- spacing **λ/2 = 3.89 cm**, UE height 1.5 m, over the reachable free floor (≥ 0.5 m from walls/clutter, main
  connected component) → **2,901,201 points in 1,380 tiles** of 2 m × 2 m (≤ 2,704 points each);
- per point: the **K = 200 strongest valid paths** — complex gain `a`, delay `tau` (first arrival at 0),
  arrival angles `phi`, `theta` — and `dropped` (power fraction outside the 200);
- solver settings fixed and validated on the original machine: **one receiver per PathSolver call** (the only
  setting that converges; batching receivers truncates NLoS paths), 1e7 samples, `max_num_paths_per_src` 1e6,
  max depth 10, specular + diffraction, per-point seed `grid_ix * 100003 + grid_iy`. **Do not change these.**

Purpose: afterwards, UE trajectories of any length are synthesised from the grid (each path moved as a plane wave
from the nearest grid point; ≤ λ/4 away) and fed to a MAC-layer emulator (MAC-Gyver) for a world-model scheduler
study. The synthesis and its validation happen on the original machine, not here. Your job is only to fill
`results/cir_grid_lam2/tiles/` and hand it back.

## Files
- `scripts/run_cir_grid.py` — the job (tiled, resumable, shardable; `--dry_run` needs no GPU). Docstring = spec.
- `scripts/cir_common.py`, `scripts/agent_motion.py` — scene loading/solver and free-space map, copied unchanged
  from the original project (`~/inf-factory-scene` on the original host).
- `scenes/inf_sl_r30_s0_concrete_ceiling/` — the scene (hall.xml + meshes + clutter.json). Small; in git.
- `data/priority_ue_pos_seg1_120s.npy` — UE positions of the first 120 s of the original trajectory dataset;
  the 129 tiles they touch are solved first (they are the validation region).
- `data/reference_pathgain_seg1.csv` — 8 points with path gains from the original machine (environment check).
- `scripts/check_env.py`, `run_grid.sh`, `status.sh`, `stop_grid.sh`.

## Steps
0. **Resource check — before installing or running anything.** Show the user the output of:
   ```bash
   df -h . /tmp; free -g; nproc; uptime
   nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ps -eo pid,user,pcpu,pmem,etime,comm --sort=-pcpu | head -15; who
   ```
   **Stop and flag the human (do not continue) if any of these holds:**
   - free disk on the repo's filesystem < 30 GB (tiles grow to ~12 GB, the venv needs a few GB, plus margin),
     or `/tmp` < 5 GB;
   - free RAM < 16 GB;
   - the GPU is already busy: another process uses > 20% utilisation or > 10 GB GPU memory, or any compute
     process from another user (the machine may be shared — do not take a GPU someone else is using);
   - load average above half the core count, or another user's heavy job in the `ps` list;
   - `nvidia-smi` fails, shows no GPU, or reports errors/ECC problems.
   If everything is clear, say so with the numbers and continue. Re-check disk and GPU memory with
   `./status.sh` during the run; if free disk falls below 10 GB, stop the workers (`./stop_grid.sh`) and flag it.
1. **Environment** (Python 3.10–3.13; Mitsuba/Dr.Jit/Sionna publish aarch64 wheels for these pins):
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
   nvidia-smi   # driver must support OptiX (recent driver); note GPU model and memory
   ```
   If a wheel is missing for this Python version, report it and ask (do not build Mitsuba from source unasked).
2. **Check** — `.venv/bin/python scripts/check_env.py` (from the repo root). It prints versions, the Mitsuba
   variant (must be a `cuda_*` variant), path gains at 8 reference points vs the original machine
   (**PASS = every point within 1.5 dB**; ray sampling is stochastic, small differences are normal), and the
   single-process solve rate. If FAIL or not CUDA: stop and report.
3. **Dry run** — `.venv/bin/python scripts/run_cir_grid.py --dry_run` must print
   `2,901,201 points in 1380 tiles`. Any other count means the scene or free-space map differs: stop and report.
4. **Choose the worker count.** One process leaves the GPU partly idle (on the original RTX PRO 6000 one worker
   used ~45% of the GPU at ~14 solves/s). Start `./run_grid.sh 1 24`, let 2 tiles finish, note the pts/s in
   `logs/`; then `./stop_grid.sh` and try `./run_grid.sh 4 24`; compare the **sum** of pts/s over workers. Keep
   the count that gives the highest total without errors (watch GPU memory in `./status.sh`; each worker holds a
   few GB). Changing the worker count later is safe: finished tiles are skipped, shards are recomputed.
5. **Run** — `./run_grid.sh <workers> 24` (workers are detached with `setsid nohup`; they survive this session).
   The user wants it running all day.
6. **Monitor** — `./status.sh` (tiles/points done, workers alive, last log line each, GPU). The logs print a
   per-shard ETA. If a worker died, read its log, report, and restart with the same command (it resumes).

## Rules
- Never run two launches over each other (`run_grid.sh` refuses if workers exist; use `stop_grid.sh` first).
- Never delete or rewrite files in `results/cir_grid_lam2/tiles/` — each tile is written atomically and is final.
- Do not change solver settings, spacing, tile size, clearance, seeds or the scene: tiles from both machines must
  be interchangeable (the original host may run the same grid at night and skips tiles that exist).
- Do not commit `results/` or logs to git (ignored). This repository is temporary.

## Handing results back
The tiles (~12 GB when complete) go back to the original host into
`~/inf-factory-scene/results/cir_grid_lam2/tiles/` (same names, same format; `manifest.json` is identical because
the grid is deterministic). Partial handovers are fine at any time, e.g. from the original host:
```bash
rsync -av --ignore-existing <gh200>:<repo>/results/cir_grid_lam2/tiles/ ~/inf-factory-scene/results/cir_grid_lam2/tiles/
```
Tell the user the numbers from `./status.sh` (tiles/points done, pts/s, ETA) when you report.

## Context (for reference, not needed to run)
- Original project: `~/inf-factory-scene` (plan: `notes/2026-09-29_cir_grid_plan.md`); world-model study resume
  doc: `~/6grocket/T2.2_armB_status.md` (both on the original host).
- On the original host a one-tile spot check of this grid was queued to run after another ray-tracing job
  (segment 4 of the trajectory dataset) finishes; it may produce tile (22, 26) there too — harmless duplicate,
  useful as a cross-machine comparison.

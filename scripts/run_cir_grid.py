"""Ray-traced CIRs on a fixed lambda/2 floor grid of the InF-SL hall, resumable in tiles, shardable across workers.

Goal: a finite set of ray-traced points from which any number of UE trajectories can be synthesised later: near a
grid point x0 each path is moved as a plane wave, a_i(x) = a_i(x0) exp(-j 2 pi / lambda * r_hat_i . (x - x0)).
At lambda/2 spacing the extrapolation distance is at most lambda/4 (~2 cm at 3.85 GHz).

Grid: spacing lambda/2 over the reachable free floor with the SAME clearance as the trajectory generator
(agent_motion.FreeSpaceGrid, clear=0.5 m, main connected component). Solver settings as the validated multi-UE
trajectory runs (inf-factory-scene/scripts/run_cir_multiue.py): one receiver per PathSolver call (the only setting
that converges, see cir_common.solve_full), 1e7 samples, max_num_paths_per_src 1e6, K = 200 strongest valid paths
with arrival angles, UE height 1.5 m, concrete-ceiling scene, isotropic antennas, TX (60, 30, 1.5) m, 3.85 GHz.

Tiles of --tile_m x --tile_m metres are written atomically to <out>/tiles/tile_<ix>_<iy>.npz; a restart skips
finished tiles, so any number of runs (and machines) can fill the same folder. --hours stops cleanly before
starting a tile that would not finish in time. Tiles touched by the positions in --priority_file are solved first.
--shard k --num_shards n: this worker takes tiles with (ix * n_tiles_y + iy) % n == k.
--dry_run: build the grid/manifest and print the work split without loading Sionna (no GPU needed).
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from agent_motion import FreeSpaceGrid

FC = 3.85e9
LAM = 299792458.0 / FC

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/cir_grid_lam2")
ap.add_argument("--scene_dir", default="scenes/inf_sl_r30_s0_concrete_ceiling")
ap.add_argument("--spacing_m", type=float, default=LAM / 2)
ap.add_argument("--clear", type=float, default=0.5)
ap.add_argument("--tile_m", type=float, default=2.0)
ap.add_argument("--hours", type=float, default=24.0, help="wall-clock budget for this run")
ap.add_argument("--max_tiles", type=int, default=0, help="stop after this many tiles (0 = no limit)")
ap.add_argument("--priority_file", default="data/priority_ue_pos_seg1_120s.npy",
                help="positions [..., 2] whose tiles are solved first ('' = none)")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num_shards", type=int, default=1)
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()
assert 0 <= a.shard < a.num_shards

SAMPLES, CAP, K = 1e7, 1e6, 200
t_start = time.time()
tag = f"[w{a.shard}/{a.num_shards}]"
log = lambda *x: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}", *x, flush=True)
os.makedirs(f"{a.out}/tiles", exist_ok=True)

# ---- grid (deterministic: identical on every run and machine) ----
fs = FreeSpaceGrid(a.scene_dir, clear=a.clear)          # 0.1 m reachability map (main component)
s = a.spacing_m
xs = np.arange(s / 2, fs.L, s)
ys = np.arange(s / 2, fs.W, s)
n_tx, n_ty = int(np.ceil(fs.L / a.tile_m)), int(np.ceil(fs.W / a.tile_m))


def tile_points(ix, iy):
    """Grid points of tile (ix, iy) that are free (exact clearance) and in the reachable component."""
    gx = xs[(xs >= ix * a.tile_m) & (xs < (ix + 1) * a.tile_m)]
    gy = ys[(ys >= iy * a.tile_m) & (ys < (iy + 1) * a.tile_m)]
    if len(gx) == 0 or len(gy) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2), int)
    X, Y = np.meshgrid(gx, gy)
    X, Y = X.ravel(), Y.ravel()
    ok = fs.free(X, Y)
    cx = np.clip(np.round(X / fs.h).astype(int), 0, len(fs.gx) - 1)
    cy = np.clip(np.round(Y / fs.h).astype(int), 0, len(fs.gy) - 1)
    ok &= fs.M[cy, cx]
    gi = np.stack([np.round((X - s / 2) / s).astype(int), np.round((Y - s / 2) / s).astype(int)], 1)
    return np.stack([X, Y], 1)[ok], gi[ok]


manifest_path = f"{a.out}/manifest.json"
if not os.path.exists(manifest_path):
    counts = {f"{ix}_{iy}": int(len(tile_points(ix, iy)[0])) for ix in range(n_tx) for iy in range(n_ty)}
    man = {"built_by": "scripts/run_cir_grid.py", "scene_dir": a.scene_dir, "fc_hz": FC, "lambda_m": LAM,
           "spacing_m": s, "grid_origin_m": [s / 2, s / 2], "clear_m": a.clear, "ue_height_m": 1.5,
           "tile_m": a.tile_m, "n_tiles": [n_tx, n_ty], "K": K, "samples": SAMPLES, "max_num_paths_per_src": CAP,
           "points_per_tile": counts, "n_points_total": int(sum(counts.values())),
           "point_seed": "seed = grid_ix * 100003 + grid_iy"}
    tmp = f"{manifest_path}.{os.getpid()}.tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, manifest_path)          # atomic; parallel workers write identical content
man = json.load(open(manifest_path))
counts = man["points_per_tile"]
assert abs(man["spacing_m"] - s) < 1e-12 and man["tile_m"] == a.tile_m, "manifest differs from arguments"

# ---- tile order: priority region first, then row-major; this worker's shard only ----
order = [(ix, iy) for iy in range(n_ty) for ix in range(n_tx) if counts[f"{ix}_{iy}"] > 0]
prio = []
if a.priority_file and os.path.exists(a.priority_file):
    p = np.load(a.priority_file).reshape(-1, 2)
    for t in dict.fromkeys(zip((p[:, 0] // a.tile_m).astype(int).tolist(), (p[:, 1] // a.tile_m).astype(int).tolist())):
        if counts.get(f"{t[0]}_{t[1]}", 0) > 0:
            prio.append(t)
prio_set = set(prio)
order = prio + [t for t in order if t not in prio_set]
mine = [t for t in order if (t[0] * n_ty + t[1]) % a.num_shards == a.shard]
done = lambda t: os.path.exists(f"{a.out}/tiles/tile_{t[0]}_{t[1]}.npz")
todo = [t for t in mine if not done(t)]
pts_total = man["n_points_total"]
pts_mine = sum(counts[f"{ix}_{iy}"] for ix, iy in mine)
pts_done_all = sum(counts[f"{ix}_{iy}"] for ix, iy in order if done((ix, iy)))
pts_done_mine = sum(counts[f"{ix}_{iy}"] for ix, iy in mine if done((ix, iy)))
log(f"grid {s * 100:.2f} cm, {pts_total:,} points in {len(order)} tiles; all workers done {pts_done_all:,} "
    f"({100 * pts_done_all / pts_total:.1f}%); this shard {len(mine)} tiles / {pts_mine:,} points, "
    f"{len(todo)} tiles to go ({sum(t in prio_set for t in todo)} priority first); budget {a.hours:.1f} h")
if a.dry_run:
    log("dry run: first tiles to solve:", todo[:5])
    sys.exit(0)

from cir_common import make_scene, solve_full, LAM as LAM_RT, FC as FC_RT   # loads Sionna / Mitsuba
assert FC_RT == FC and abs(LAM_RT - LAM) < 1e-15
sc = make_scene(os.path.join(a.scene_dir, "hall.xml"))
rate = None          # points per second, measured
n_run = 0
for t in todo:
    if done(t):          # another run/machine may have finished it meanwhile
        continue
    pts, gi = tile_points(*t)
    elapsed = time.time() - t_start
    if rate and elapsed + len(pts) / rate > a.hours * 3600:
        log(f"stopping: next tile needs ~{len(pts) / rate / 60:.1f} min, budget left {(a.hours * 3600 - elapsed) / 60:.1f} min")
        break
    t0 = time.time()
    A = np.zeros((len(pts), K), np.complex64); TAU = np.zeros((len(pts), K), np.float32)
    PHI = np.zeros_like(TAU); TH = np.zeros_like(TAU); DR = np.zeros(len(pts), np.float32)
    for j, (pt, g) in enumerate(zip(pts, gi)):
        A[j], TAU[j], PHI[j], TH[j], DR[j] = [x[0] for x in solve_full(
            sc, pt[None, :], samples=SAMPLES, max_num_paths_per_src=CAP, seed=int(g[0] * 100003 + g[1]), K=K)]
    tmp = f"{a.out}/tiles/.tile_{t[0]}_{t[1]}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, pos=pts.astype(np.float64), grid_idx=gi.astype(np.int32), a=A, tau=TAU,
                        phi=PHI, theta=TH, dropped=DR)
    os.replace(tmp, f"{a.out}/tiles/tile_{t[0]}_{t[1]}.npz")
    dt_tile = time.time() - t0
    rate = len(pts) / dt_tile if rate is None else 0.8 * rate + 0.2 * len(pts) / dt_tile
    pts_done_mine += len(pts); n_run += 1
    log(f"tile {t} {len(pts)} pts in {dt_tile:.0f} s ({len(pts) / dt_tile:.2f} pts/s) | shard "
        f"{pts_done_mine:,}/{pts_mine:,} ({100 * pts_done_mine / pts_mine:.2f}%) | shard ETA "
        f"{(pts_mine - pts_done_mine) / rate / 3600:.1f} h")
    if a.max_tiles and n_run >= a.max_tiles:
        log("max_tiles reached")
        break
log(f"run finished: {n_run} tiles this run, shard {pts_done_mine:,}/{pts_mine:,} points done")

"""Environment check for the lambda/2 CIR grid: versions, GPU variant, and agreement with the original machine.

Solves the 8 reference points in data/reference_pathgain_seg1.csv (segment 1 of the multi-UE dataset, computed on
the original x86 + RTX PRO 6000 host with the same solver settings and seeds) and compares the wideband path gain
(sum |a|^2 over the K strongest paths). Ray sampling is stochastic, so small differences are expected across
GPUs/builds; PASS if every point is within 1.5 dB. Also measures the single-process solve rate.

Run from the repo root: python scripts/check_env.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib.metadata import version
import mitsuba as mi
import sionna.rt  # noqa: F401  (selects the Mitsuba variant)
print(f"python {sys.version.split()[0]}, drjit {version('drjit')}, mitsuba {version('mitsuba')}, "
      f"sionna-rt {version('sionna-rt')}, numpy {version('numpy')}")
from cir_common import make_scene, solve_full
print("mitsuba variant:", mi.variant())
if "cuda" not in (mi.variant() or ""):
    print("WARNING: not a CUDA variant - ray tracing would run on the CPU (very slow)")

sc = make_scene("scenes/inf_sl_r30_s0_concrete_ceiling/hall.xml")
ref = np.loadtxt("data/reference_pathgain_seg1.csv", delimiter=",", skiprows=1)
ok = True
for tick, ue, x, y, pg_ref in ref:
    a, tau, phi, th, dr = solve_full(sc, np.array([[x, y]]), samples=1e7, max_num_paths_per_src=1e6,
                                     seed=int(tick) * 1000 + int(ue), K=200)
    pg = 10 * np.log10((np.abs(a[0]) ** 2).sum())
    d = pg - pg_ref
    ok &= abs(d) <= 1.5
    print(f"tick {int(tick):6d} ue{int(ue)} ({x:6.2f},{y:6.2f}): path gain {pg:8.3f} dB, reference {pg_ref:8.3f} dB, "
          f"diff {d:+.3f} dB, valid paths {int((np.abs(a[0]) > 0).sum())}")
print("AGREEMENT:", "PASS" if ok else "FAIL (differences > 1.5 dB: stop and report before running the grid)")

rng = np.random.default_rng(0)
pts = np.column_stack([rng.uniform(20, 100, 20), rng.uniform(10, 50, 20)])
t0 = time.time()
for j, p in enumerate(pts):
    solve_full(sc, p[None, :], samples=1e7, max_num_paths_per_src=1e6, seed=j, K=200)
r = len(pts) / (time.time() - t0)
print(f"single-process rate: {r:.2f} solves/s -> 2.90 M points take {2.901201e6 / r / 3600:.1f} h with one worker")

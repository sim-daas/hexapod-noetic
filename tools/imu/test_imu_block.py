#!/usr/bin/env python3
"""Offline tests for scripts/imu_block.py and tools/imu/imu_check.py (no robot needed): python3 tools/imu/test_imu_block.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "scripts")]
from imu_block import G0, ImuBlock  # noqa: E402
import imu_check  # noqa: E402
import imu_synth  # noqa: E402

CALIB = os.path.join(HERE, "..", "..", "config", "imu_calib.json")


def run_block(raw, blk=None):
    blk = blk or ImuBlock(CALIB)
    out, ab = [], []
    for k in range(len(raw["t"])):
        out.append(blk.update(raw["a_raw"][k], raw["w_raw"][k], raw["t"][k])); ab.append(blk.ab.copy())
    return np.array(out), np.array(ab)


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)))


t, Rwb = imu_synth.trajectory(60.0, 50.0)
raw = imu_synth.raw_from_truth(CALIB, t, Rwb)
out, ab = run_block(raw)
settle = t > 3.0

# 1. projected gravity tracks truth through tilts and yaw spins, and is a unit vector
e = ang(out[:, 6:9], raw["g_true"])
print("gravity error: mean %.2f deg, p99 %.2f, max %.2f" % (e[settle].mean(), np.percentile(e[settle], 99), e[settle].max()))
assert np.allclose(np.linalg.norm(out[:, 6:9], axis=1), 1.0, atol=1e-9)
assert e[settle].mean() < 0.5 and e[settle].max() < 2.0

# 2. gyro in the body frame equals truth (bias removed, axes rotated)
assert np.abs(out[:, 3:6] - raw["w_b"]).max() < 1e-9

# 3. accel at the origin matches truth better with the lever-arm correction than without
rms = lambda x: float(np.sqrt(np.mean(x[settle] ** 2)))  # noqa: E731
with_corr = rms(out[:, 0:3] - raw["a_origin"])
blk0 = ImuBlock(CALIB); blk0.r = np.zeros(3)
no_corr = rms(run_block(raw, blk0)[0][:, 0:3] - raw["a_origin"])
print("accel@origin rms error: with lever-arm correction %.3f m/s^2, without %.3f" % (with_corr, no_corr))
assert with_corr < no_corr and with_corr < 0.25

# 4. static: level pose reads (0, 0, +9.81) and gravity (0, 0, -1); start() re-measures a wrong gyro bias
R = np.array(ImuBlock(CALIB).R)
a_level_raw = (R.T @ np.array([0, 0, G0])) / np.array(ImuBlock(CALIB).sc) + np.array(ImuBlock(CALIB).off)
true_bias = ImuBlock(CALIB).gyro_bias + np.array([0.03, -0.02, 0.01])  # temperature-drifted bias
blk = ImuBlock(CALIB)
blk.start([(a_level_raw, true_bias.copy()) for _ in range(100)])
x = blk.update(a_level_raw, true_bias, 0.0)
print("level after start(): accel %s gravity %s" % (np.round(x[:3], 3), np.round(x[6:], 3)))
assert np.allclose(x[:3], [0, 0, G0], atol=1e-6) and np.allclose(x[3:6], 0, atol=1e-9) and np.allclose(x[6:], [0, 0, -1])
try:
    ImuBlock(CALIB).start([(a_level_raw, true_bias + k * 0.1) for k in range(100)])
    raise AssertionError("moving start-up must be rejected")
except ValueError:
    pass

# 5. checker: passes on correct data, fails when a gyro axis is flipped or two axes are swapped
res = imu_check.analyse(t, ab, out[:, 3:6])
print(imu_check.report(res))
assert imu_check.passed(imu_check.verdict(res))
for name, bad in (("flipped gyro y", out[:, 3:6] * [1, -1, 1]), ("swapped gyro x/y", out[:, [4, 3, 5]])):
    r = imu_check.analyse(t, ab, bad)
    print("%s: median %.1f deg, p95 %.1f deg" % (name, r["err_med"], r["err_p95"]))
    assert not imu_check.passed(imu_check.verdict(r)) and r["err_p95"] > 10
# 6. the visualiser's attitude (what the mesh shows) equals the true rotation: front-up/left-up/CCW all match
import imu_viz  # noqa: E402  (matplotlib is imported lazily, so this is headless-safe)
att, worst = imu_viz.Attitude(), 0.0
for k in range(1, len(t)):
    att.step(out[k, 6:9], out[k, 3:6], t[k] - t[k - 1])
    if t[k] > 3.0:
        M = att.R.T @ Rwb[k]  # identity when the display matches the truth
        worst = max(worst, np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))
print("display attitude vs truth: worst rotation error %.2f deg over the whole session (tilt + yaw spins)" % worst)
assert worst < 3.0
print("ALL TESTS PASSED")

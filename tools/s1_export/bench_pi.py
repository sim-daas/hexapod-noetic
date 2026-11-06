"""Pi 4 latency test of the blind S1H4 control path: E_p TCN (W 50) + V heads + NumPy v_kin + actor (1,269-D).

Python 3.8 compatible. Usage: python3 bench_pi.py [n_iters]
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

D = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
K = np.load(os.path.join(D, "kin_consts.npz"))
HIP, HIP_YAW, FEMUR_OFF, TIBIA_OFF, TOE_OFF = (K[k].astype(np.float64) for k in
                                               ("HIP", "HIP_YAW", "FEMUR_OFF", "TIBIA_OFF", "TOE_OFF"))


def _ry(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack((c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]), axis=-1)


def _rz(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)


def toe_positions(q):
    qc, qf, qt = q[..., 0:6], q[..., 6:12], q[..., 12:18]
    shape = q.shape[:-1] + (6, 3)
    p = _ry(np.broadcast_to(TOE_OFF, shape), qt) + TIBIA_OFF
    p = _ry(p, qf) + FEMUR_OFF
    p = _rz(p, qc + HIP_YAW)
    return p + HIP


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def v_kin(q, qd, err, gyro, tau_e=0.005, s=0.003, delta=-0.005, eps=1e-3):
    """Single sample: q, qd, err (18,), gyro (3,) → v (3,), w (6,). Mirrors hexasim rl/s1/kinematics.py."""
    p = toe_positions(q)
    jqd = (toe_positions(q + eps * qd) - toe_positions(q - eps * qd)) / (2 * eps)
    v_foot = -(jqd + np.cross(np.broadcast_to(gyro, p.shape), p))
    e = np.abs(err[6:12]) + np.abs(err[12:18])
    A = np.concatenate((p[:, :2], np.ones((6, 1))), axis=-1)
    coef = np.linalg.lstsq(A, p[:, 2:3], rcond=None)[0]
    z_plane = (A @ coef)[:, 0]
    w = sig((e - tau_e) / s) * sig((z_plane - p[:, 2] + delta) / 0.005)
    v = (w[:, None] * v_foot).sum(0) / max(w.sum(), 1e-3)
    return v, w


def stats(ts):
    a = np.array(ts) * 1e3
    return "mean %.3f  p50 %.3f  p99 %.3f  max %.3f ms" % (a.mean(), np.percentile(a, 50), np.percentile(a, 99), a.max())


def sessions(threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = ["CPUExecutionProvider"]
    return (ort.InferenceSession(os.path.join(D, "aux.onnx"), so, providers=prov),
            ort.InferenceSession(os.path.join(D, "actor.onnx"), so, providers=prov))


# ---------------------------------------------------------------- parity
io = np.load(os.path.join(D, "sample_io.npz"))
aux, act = sessions(1)
z, dv, fh = aux.run(None, {"hist": io["hist"]})
a = act.run(None, {"obs": io["obs"]})[0]
print("parity  z %.2e  dv %.2e  fhat %.2e  action %.2e" % (
    abs(z - io["z"]).max(), abs(dv - io["dv"]).max(), abs(fh - io["fhat"]).max(), abs(a - io["action"]).max()))
ref = np.load(os.path.join(D, "vkin_ref.npz"))
vk_err = max(abs(v_kin(ref["q"][i], ref["qd"][i], ref["err"][i], ref["gyro"][i])[0] - ref["v"][i]).max()
             for i in range(len(ref["q"])))
print("parity  v_kin %.2e" % vk_err)

# ---------------------------------------------------------------- per-component latency
print("onnxruntime %s, %d iterations" % (ort.__version__, N))
rng = np.random.default_rng(0)
hist = io["hist"].astype(np.float32)
obs = np.zeros((1, 1269), np.float32)
for threads in (1, 2, 4):
    aux, act = sessions(threads)
    for _ in range(50):
        aux.run(None, {"hist": hist}); act.run(None, {"obs": obs})
    t_aux, t_vk, t_act, t_step = [], [], [], []
    for i in range(N):
        frame = (rng.standard_normal(111) * 0.1).astype(np.float32)
        t0 = time.perf_counter()
        hist = np.roll(hist, -1, axis=1); hist[0, -1] = frame                         # history update
        t1 = time.perf_counter()
        z, dv, fh = aux.run(None, {"hist": hist})
        t2 = time.perf_counter()
        vk, _ = v_kin(frame[0:18].astype(np.float64), frame[18:36], frame[36:54], frame[75:78])
        t3 = time.perf_counter()
        obs[0, 117:181] = z[0]; obs[0, 181:184] = vk + dv[0]; obs[0, 184:196] = fh[0]  # z_p, ê blocks
        act.run(None, {"obs": obs})
        t4 = time.perf_counter()
        t_aux.append(t2 - t1); t_vk.append(t3 - t2); t_act.append(t4 - t3); t_step.append(t4 - t0)
    print("threads=%d" % threads)
    print("  TCN+V   " + stats(t_aux))
    print("  v_kin   " + stats(t_vk))
    print("  actor   " + stats(t_act))
    print("  total   " + stats(t_step))

# ---------------------------------------------------------------- timed 100 Hz loop (compute only, no ROS/bus)
for threads in (1, 4):
    aux, act = sessions(threads)
    period, n = 0.01, 3000
    misses, worst, t_next = 0, 0.0, time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        hist = np.roll(hist, -1, axis=1)
        z, dv, fh = aux.run(None, {"hist": hist})
        vk, _ = v_kin(hist[0, -1, 0:18].astype(np.float64), hist[0, -1, 18:36], hist[0, -1, 36:54], hist[0, -1, 75:78])
        obs[0, 117:181] = z[0]; obs[0, 181:184] = vk + dv[0]; obs[0, 184:196] = fh[0]
        act.run(None, {"obs": obs})
        dt = time.perf_counter() - t0
        worst = max(worst, dt)
        t_next += period
        slack = t_next - time.perf_counter()
        if slack < 0:
            misses += 1
            t_next = time.perf_counter()
        else:
            time.sleep(slack)
    print("100 Hz loop, threads=%d: %d/%d deadline misses, worst compute %.2f ms" % (threads, misses, n, worst * 1e3))

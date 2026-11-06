"""Pi latency of the fused control graph (cone TCN + V + v_kin + actor). Python 3.8."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

D = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def session(threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(os.path.join(D, "control.onnx"), so, providers=["CPUExecutionProvider"])


io = np.load(os.path.join(D, "fused_io.npz"))
s = session(1)
print("parity action %.2e" % abs(s.run(None, {"hist": io["hist"], "obs_base": io["obs_base"]})[0] - io["action"]).max())
hist, obs = io["hist"].astype(np.float32), io["obs_base"].astype(np.float32)
rng = np.random.default_rng(0)
for threads in (1, 2):
    s = session(threads)
    for _ in range(100):
        s.run(None, {"hist": hist, "obs_base": obs})
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        hist[0, :-1] = hist[0, 1:]; hist[0, -1] = rng.standard_normal(111).astype(np.float32) * 0.1
        s.run(None, {"hist": hist, "obs_base": obs})
        ts.append(time.perf_counter() - t0)
    a = np.array(ts) * 1e3
    print("threads=%d  step mean %.3f  p50 %.3f  p99 %.3f  p99.9 %.3f  max %.3f ms" % (
        threads, a.mean(), np.percentile(a, 50), np.percentile(a, 99), np.percentile(a, 99.9), a.max()))
    for hz in (100, 50):
        period, misses, t_next, worst = 1.0 / hz, 0, time.perf_counter(), 0.0
        for _ in range(N):
            t0 = time.perf_counter()
            hist[0, :-1] = hist[0, 1:]
            s.run(None, {"hist": hist, "obs_base": obs})
            worst = max(worst, time.perf_counter() - t0)
            t_next += period
            slack = t_next - time.perf_counter()
            if slack < 0:
                misses += 1; t_next = time.perf_counter()
            else:
                time.sleep(slack)
        print("  %d Hz loop: %d/%d misses, worst compute %.2f ms" % (hz, misses, N, worst * 1e3))

#!/usr/bin/env python3
"""Robot-side gait clock and slew limit (scripts/s1h_policy.py) against a recording of the real hexasim env
(tools/sim/sim_clock_slew_record.py). NumPy only.

    HEXAPOD_POLICY=s1xls python3 tools/sim/check_clock_slew.py <recording.npz>

Per env (up to its first episode end): the robot Clock starts at the recorded phase of step 1 (the env's per-episode
jitter) and is fed the command column of each observation; its 12 clock values must equal the observation's. Step 0
is left out: the first observation of a freshly built env carries a zero command and the un-jittered phase. The sent
joint target must equal slew(clip(q_default + scale a), previous sent target)."""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from joint_map import Contract  # noqa: E402
from s1h_policy import Clock, slew  # noqa: E402

C = Contract()
r = dict(np.load(sys.argv[1]))  # all arrays in memory (an NpzFile decompresses on every access)
assert str(r["task"]) == C.raw["source"]["task_id"], (r["task"], C.raw["source"]["task_id"])
assert abs(float(r["rate_limit"]) - C.rate_limit) < 1e-9 and list(r["joint_names"]) == C.names
T, N = r["cmd"].shape[:2]
done = r["done"]
worst_clock = worst_slew = 0.0
n_steps = n_changes = n_stand = n_limited = 0
for e in range(N):
    end = int(np.argmax(done[:, e])) if done[:, e].any() else T  # obs up to index `end` belong to the first episode
    ck = Clock(C)
    ck.phase = r["phase"][1, e].astype(np.float64).copy()
    for t in range(1, end):
        mine = ck.frame(r["cmd"][t, e]) * float(C.raw["clock_on"])
        worst_clock = max(worst_clock, np.abs(mine - r["clock"][t, e]).max())
        n_stand += int(not r["cmd"][t, e].any())
        n_changes += int(t > 1 and np.abs(r["cmd"][t, e] - r["cmd"][t - 1, e]).max() > 1e-6)
        if t > 1:
            raw = np.clip(C.q_default + C.scale * r["a"][t, e], C.clip_lo, C.clip_hi)
            m = C.rate_limit * C.dt
            mine_q = slew(raw, r["q_sent"][t - 1, e], m) if m > 0 else raw
            worst_slew = max(worst_slew, np.abs(mine_q - r["q_sent"][t, e]).max())
            n_limited += int((np.abs(raw - r["q_sent"][t - 1, e]) > m).any()) if m > 0 else 0
        n_steps += 1
print("%s: %d envs, %d steps, %d command changes, %d standing steps, %d steps with the slew limit active"
      % (C.tag, N, n_steps, n_changes, n_stand, n_limited))
print("max |clock - env| %.2e   max |sent target - env| %.2e" % (worst_clock, worst_slew))
assert n_changes > 0 and n_stand > 0, "the recording has no stand / command change to test"
assert worst_clock < 1e-4 and worst_slew < 1e-5
print("OK")

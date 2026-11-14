"""Blind S1-family policy on the robot (S1 1,125-D, S1H4 1,269-D): o_p frame builder + gait clock + W-frame history +
fused ONNX graph. NumPy + onnxruntime, 3.8.

Mirrors hexasim HexapodS1Env.refresh_proprio for the actor input (mdp.layout):
  o_p (111) = [q - q_default, qd, q_cmd_prev - q, a_prev, imu acc (3), gyro (3), gravity (3), clock (12), mask (18)]
  obs       = [o_p | V_0 x keep (= 0) | twist_cmd | z_p, e_hat (computed inside the graph) | zeros]
The W-step history starts with its first frame repeated (the env's episode-start fill). The fused graph
(control.onnx in the contract's model directory, tools/s1_export/export_fused.py) takes hist (1, W, 111) + obs_base (1, obs_dim) and puts z_p
and e_hat into obs_base itself, so only o_p, V_0 and twist_cmd are filled here. q_cmd = clip(q_default + 0.5 a).
"""
import math
import os

import numpy as np


class ObsBuilder:
    def __init__(self, contract):
        self.c = contract
        self.clock_on = float(contract.raw["clock_on"])  # 0 for S1H4 (clock fully dropped in training)

    def frame(self, q, qd, mask, q_cmd_prev, a_prev, imu9, clock12=None):
        c = self.c
        clock = np.zeros(12) if clock12 is None else np.asarray(clock12) * self.clock_on
        return np.concatenate([q - c.q_default, qd, q_cmd_prev - q, a_prev, imu9, clock, mask])


class Clock:
    """Gait clock of hexasim HexapodS0bEnv, one robot. Per control step the phase of every leg advances by
    2 pi f dt, f = clamp(v_cmd_x / stride, f_min, f_max), and stands still under a zero command; the observation is
    (sin, cos) per leg, all zeros while standing. As in the env, the step that just ended decides both (its command
    is the one "in force"), so a command change reaches the clock one step after it reaches twist_cmd. The episode
    starts at the tripod offsets (the env adds a per-episode phase jitter; 0 here)."""

    def __init__(self, contract):
        k = contract.raw["clock"]
        self.offsets = np.array(k["offsets"], dtype=np.float64)
        self.stride, self.f_min, self.f_max = float(k["stride_m"]), float(k["f_min"]), float(k["f_max"])
        self.dt = contract.dt
        self.reset()

    def reset(self):
        self.phase = self.offsets.copy()
        self.cmd_last = None

    def frame(self, cmd):
        """cmd (3,): the command of this step's observation. -> clock (12,) = [sin0, cos0, sin1, cos1, ...]."""
        cmd = np.asarray(cmd, dtype=np.float64)
        ref = cmd if self.cmd_last is None else self.cmd_last
        active = bool(np.any(ref != 0.0))
        if self.cmd_last is not None and active:
            f = min(max(ref[0] / self.stride, self.f_min), self.f_max)
            self.phase = (self.phase + 2.0 * math.pi * f * self.dt) % (2.0 * math.pi)
        self.cmd_last = cmd.copy()
        return np.stack((np.sin(self.phase), np.cos(self.phase)), axis=-1).reshape(12) * float(active)


def slew(q_cmd, q_sent, max_step):
    """hexasim mdp/slew_action.py: the sent target moves at most max_step (rad) per joint per control step."""
    return q_sent + np.clip(q_cmd - q_sent, -max_step, max_step)


class S1HPolicy:
    def __init__(self, contract, onnx_path=None, threads=1):
        import onnxruntime as ort  # deferred: the bench/offline tools import this module's siblings without ORT

        self.c = contract
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = onnx_path or os.path.join(contract.model_dir, "control.onnx")
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        shapes = {i.name: i.shape for i in self.sess.get_inputs()}
        assert list(shapes["hist"])[1:] == [contract.window, contract.frame_dim], shapes
        assert list(shapes["obs_base"])[1] == contract.obs_dim, shapes
        self.hist = np.zeros((1, contract.window, contract.frame_dim), dtype=np.float32)
        self.obs = np.zeros((1, contract.obs_dim), dtype=np.float32)
        L = contract.layout
        self.s_op, self.s_v0, self.s_cmd = slice(0, contract.frame_dim), L["v0"], L["twist_cmd"]
        self.started = False

    def reset(self):
        self.started = False

    def act(self, o_p, cmd):
        """o_p (111,), cmd (3,) -> raw action a (18,) float64. First call after reset fills the history."""
        f = np.asarray(o_p, dtype=np.float32)
        if not self.started:
            self.hist[0, :] = f
            self.started = True
        else:
            self.hist[0, :-1] = self.hist[0, 1:]
            self.hist[0, -1] = f
        self.obs[0, self.s_op] = f
        self.obs[0, self.s_v0] = 0.0  # V_0 retired (v0_drop_p = 1)
        self.obs[0, self.s_cmd] = np.asarray(cmd, dtype=np.float32)
        a = self.sess.run(["action"], {"hist": self.hist, "obs_base": self.obs})[0]
        return a[0].astype(np.float64)

    def q_cmd(self, a):
        c = self.c
        return np.clip(c.q_default + c.scale * a, c.clip_lo, c.clip_hi)

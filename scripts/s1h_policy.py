"""Blind S1H4 policy on the robot: o_p frame builder + W-frame history + fused ONNX graph. NumPy + onnxruntime, 3.8.

Mirrors hexasim HexapodS1Env.refresh_proprio for the actor input (1,269-D layout, mdp.layout):
  o_p (111) = [q - q_default, qd, q_cmd_prev - q, a_prev, imu acc (3), gyro (3), gravity (3), clock (12), mask (18)]
  obs       = [o_p | V_0 x keep (= 0) | twist_cmd | z_p, e_hat (computed inside the graph) | zeros]
The W-step history starts with its first frame repeated (the env's episode-start fill). The fused graph
(models/control.onnx, tools/s1_export/export_fused.py) takes hist (1, W, 111) + obs_base (1, obs_dim) and puts z_p
and e_hat into obs_base itself, so only o_p, V_0 and twist_cmd are filled here. q_cmd = clip(q_default + 0.5 a).
"""
import os

import numpy as np

from joint_map import REPO


class ObsBuilder:
    def __init__(self, contract):
        self.c = contract
        self.clock_on = float(contract.raw["clock_on"])  # 0 for S1H4 (clock fully dropped in training)

    def frame(self, q, qd, mask, q_cmd_prev, a_prev, imu9, clock12=None):
        c = self.c
        clock = np.zeros(12) if clock12 is None else np.asarray(clock12) * self.clock_on
        return np.concatenate([q - c.q_default, qd, q_cmd_prev - q, a_prev, imu9, clock, mask])


class S1HPolicy:
    def __init__(self, contract, onnx_path=None, threads=1):
        import onnxruntime as ort  # deferred: the bench/offline tools import this module's siblings without ORT

        self.c = contract
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = onnx_path or os.path.join(REPO, "models", "control.onnx")
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

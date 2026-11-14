"""Sim joint order <-> servo IDs, by name, plus the contract constants. NumPy only, Python 3.8.

Sim order (hexasim): [coxa x 6, femur x 6, tibia x 6], legs RF RM RB LF LM LB. Everything in the runner is in sim
order; only the bus driver sees servo IDs. All values come from config/contract_<policy>.json
(tools/policy/export_contract.py), never typed by hand. The policy is picked by tag: HEXAPOD_POLICY in the
environment (default s1h4), or Contract(tag) / the runner's --policy.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
POLICY = os.environ.get("HEXAPOD_POLICY", "s1h4")
# Robot-specific SLEEP pose captured from the servos (tools/poses/capture_sleep.py); overrides the vendor
# sleep_positions when present.
SLEEP_FILE = os.path.join(REPO, "config", "sleep_pose.json")


def policy_files(tag):
    """(contract json, model directory) of an exported policy. s1h4, the first export, sits in models/ itself."""
    return (os.path.join(REPO, "config", "contract_%s.json" % tag),
            os.path.join(REPO, "models") if tag == "s1h4" else os.path.join(REPO, "models", tag))


class Contract:
    def __init__(self, tag=None, sleep_file=SLEEP_FILE):
        self.tag = tag or POLICY
        path, self.model_dir = policy_files(self.tag)
        c = json.load(open(path))
        self.raw = c
        self.names = list(c["joint_names"])
        assert len(self.names) == 18
        types = [n.rsplit("_", 1)[1] for n in self.names]
        assert types == ["coxa"] * 6 + ["femur"] * 6 + ["tibia"] * 6, types
        self.q_default = np.array(c["q_default"], dtype=np.float64)
        # Every command passes this box: the policy clip intersected with the URDF limits.
        self.lo = np.maximum(np.array(c["clip_lo"]), np.array(c["urdf_lo"]))
        self.hi = np.minimum(np.array(c["clip_hi"]), np.array(c["urdf_hi"]))
        self.clip_lo, self.clip_hi = np.array(c["clip_lo"]), np.array(c["clip_hi"])
        self.scale = float(c["action_scale"])
        self.dt = float(c["dt"])
        self.window = int(c["window"])
        self.obs_dim = int(c["obs_dim"])
        self.frame_dim = int(c["frame_dim"])
        self.layout = {k: slice(*v) for k, v in c["layout"].items()}
        self.tick = float(c["tick"])
        # Slew limit on the sent target that the policy trained with (rad/s; 0 = none): part of the action path.
        self.rate_limit = float(c.get("rate_limit", 0.0))
        self.ids = np.array([c["servo"]["ids"][n] for n in self.names], dtype=int)
        assert sorted(self.ids.tolist()) == list(range(1, 19)), self.ids
        self.sleep = np.array(c["sleep_positions"], dtype=np.float64)
        self.sleep_source = "vendor sleep_positions"
        if sleep_file and os.path.exists(sleep_file):
            sp = json.load(open(sleep_file))
            self.sleep = np.array([sp["q"][n] for n in self.names], dtype=np.float64)
            self.sleep_source = "%s (%s)" % (os.path.basename(sleep_file), sp.get("captured", "?"))
        self.command = c["command"]
        self.servo = c["servo"]
        self.kin = {k: np.array(v, dtype=np.float64) for k, v in c["kinematics"].items()}

    def clip(self, q):
        return np.clip(q, self.lo, self.hi)

    def short(self, i):
        """'RF coxa' style label for joint i (sim order)."""
        leg, t = self.names[i].rsplit("_", 1)
        a, b = leg.split("_")
        return "%s%s %s" % (a[0].upper(), b[0].upper(), t)

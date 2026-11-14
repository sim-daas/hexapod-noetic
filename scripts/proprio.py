"""NumPy 1:1 port of hexasim mdp/proprio.py: ProprioReader (single robot, no drop model). Python 3.8.

Per control step: reads quantised to the sim servo's tick, zero-order hold of q on a dropped read, two-sample
backward-difference velocity

    qd_t = (q_t - q_{t-2}) / (2 dt),  held at the last valid qd if either endpoint is a held value,

and an 18-wide validity mask (1 = fresh read). No filter. Parity with the torch original: tools/test_teleop_offline.py.
"""
import numpy as np


class ProprioReader:
    def __init__(self, num_joints, dt, tick):
        # tick: position resolution (rad) of the servo model the policy trained on (contract "tick"): the AX-12A's
        # 10-bit grid for the AX-trained checkpoints (the robot's 12-bit reads are re-quantised to it), the XL430's
        # own 2 pi / 4096 for the XL430-trained ones.
        self.dt, self.tick = dt, float(tick)
        self.q = np.zeros(num_joints)
        self.q_prev = np.zeros(num_joints)
        self.q_prev2 = np.zeros(num_joints)
        self.qd = np.zeros(num_joints)
        self.mask = np.ones(num_joints)
        self.mask_prev = np.ones(num_joints)
        self.mask_prev2 = np.ones(num_joints)

    def quantise(self, q):
        return np.round(np.asarray(q, dtype=np.float64) / self.tick) * self.tick

    def reset(self, q_true):
        q = self.quantise(q_true)
        self.q, self.q_prev, self.q_prev2 = q.copy(), q.copy(), q.copy()
        self.qd = np.zeros_like(q)
        self.mask, self.mask_prev, self.mask_prev2 = np.ones_like(q), np.ones_like(q), np.ones_like(q)

    def update(self, q_true, valid):
        """q_true: (n,) read angles (any value where invalid); valid: (n,) bool read success."""
        valid = np.asarray(valid, dtype=bool)
        self.q_prev2, self.q_prev = self.q_prev, self.q
        self.mask_prev2, self.mask_prev = self.mask_prev, self.mask
        self.q = np.where(valid, self.quantise(np.where(valid, q_true, 0.0)), self.q_prev)
        self.mask = valid.astype(np.float64)
        fresh = (self.mask * self.mask_prev2).astype(bool)
        self.qd = np.where(fresh, (self.q - self.q_prev2) / (2.0 * self.dt), self.qd)

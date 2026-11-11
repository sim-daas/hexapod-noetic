"""NumPy 1:1 port of hexasim mdp/proprio.py: ProprioReader (single robot, no drop model). Python 3.8.

Per control step: 10-bit reads, zero-order hold of q on a dropped read, two-sample backward-difference velocity

    qd_t = (q_t - q_{t-2}) / (2 dt),  held at the last valid qd if either endpoint is a held value,

and an 18-wide validity mask (1 = fresh read). No filter. Parity with the torch original: tools/test_teleop_offline.py.
"""
import math

import numpy as np

# The sim's servo tick (hexasim actuators.ax12a.TICK, = contract "tick"). The robot's XL430s read 12-bit positions;
# re-quantising them to this 10-bit grid keeps q, qd and the tracking error on the distribution the policy trained on.
TICK = math.radians(300.0) / 1023.0


def quantise(q):
    return np.round(np.asarray(q, dtype=np.float64) / TICK) * TICK


class ProprioReader:
    def __init__(self, num_joints, dt):
        self.dt = dt
        self.q = np.zeros(num_joints)
        self.q_prev = np.zeros(num_joints)
        self.q_prev2 = np.zeros(num_joints)
        self.qd = np.zeros(num_joints)
        self.mask = np.ones(num_joints)
        self.mask_prev = np.ones(num_joints)
        self.mask_prev2 = np.ones(num_joints)

    def reset(self, q_true):
        q = quantise(q_true)
        self.q, self.q_prev, self.q_prev2 = q.copy(), q.copy(), q.copy()
        self.qd = np.zeros_like(q)
        self.mask, self.mask_prev, self.mask_prev2 = np.ones_like(q), np.ones_like(q), np.ones_like(q)

    def update(self, q_true, valid):
        """q_true: (n,) read angles (any value where invalid); valid: (n,) bool read success."""
        valid = np.asarray(valid, dtype=bool)
        self.q_prev2, self.q_prev = self.q_prev, self.q
        self.mask_prev2, self.mask_prev = self.mask_prev, self.mask
        self.q = np.where(valid, quantise(np.where(valid, q_true, 0.0)), self.q_prev)
        self.mask = valid.astype(np.float64)
        fresh = (self.mask * self.mask_prev2).astype(bool)
        self.qd = np.where(fresh, (self.q - self.q_prev2) / (2.0 * self.dt), self.qd)

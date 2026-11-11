"""Leg FK / IK for pxmark4 from the contract's URDF constants. NumPy only, Python 3.8.

Chain per leg (= hexasim rl/s1/kinematics.py): base -> hip (x, y, yaw) -> Rz(q_coxa) -> +femur_off -> Ry(q_femur)
-> +tibia_off -> Ry(q_tibia) -> toe +toe_off. Leg plane: r = horizontal distance of the toe from the coxa axis,
z = height of the toe relative to the coxa plane (base_link z = 0). Joint arrays are in sim order (coxa x 6,
femur x 6, tibia x 6).
"""
import numpy as np


def _ry(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack((c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]), axis=-1)


def _rz(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)


class LegKin:
    def __init__(self, kin):
        """kin: Contract.kin (hip (6,3), hip_yaw (6,), femur_off, tibia_off, toe_off (3,))."""
        self.hip, self.hip_yaw = kin["hip"], kin["hip_yaw"]
        self.f_off, self.t_off, self.toe = kin["femur_off"], kin["tibia_off"], kin["toe_off"]
        assert abs(self.f_off[1]) < 1e-12 and abs(self.f_off[2]) < 1e-12
        self.r0 = float(self.f_off[0])  # femur axis distance from the coxa axis
        self.L1 = np.array([self.t_off[0], self.t_off[2]])  # femur link in the leg plane (x, z)
        self.L2 = np.array([self.toe[0], self.toe[2]])  # tibia link
        self.l1, self.l2 = np.linalg.norm(self.L1), np.linalg.norm(self.L2)
        self.a1, self.a2 = np.arctan2(self.L1[1], self.L1[0]), np.arctan2(self.L2[1], self.L2[0])
        self.reach = (abs(self.l1 - self.l2), self.l1 + self.l2)

    # ------------------------------------------------------------------ FK
    def toe_positions(self, q):
        """(..., 18) -> (..., 6, 3) toes in base_link (identical to hexasim kinematics.toe_positions)."""
        q = np.asarray(q, dtype=np.float64)
        qc, qf, qt = q[..., 0:6], q[..., 6:12], q[..., 12:18]
        shape = q.shape[:-1] + (6, 3)
        p = _ry(np.broadcast_to(self.toe, shape), qt) + self.t_off
        p = _ry(p, qf) + self.f_off
        p = _rz(p, qc + self.hip_yaw)
        return p + self.hip

    def plane_fk(self, qf, qt):
        """femur, tibia (rad, arrays OK) -> (r, z) of the toe in the leg plane."""
        p = _ry(np.broadcast_to(self.toe, np.shape(qt) + (3,)), qt) + self.t_off
        p = _ry(p, qf) + self.f_off
        return p[..., 0], p[..., 2]

    def knee(self, qf):
        """Tibia-joint (knee) position (r, z) in the leg plane."""
        p = _ry(np.broadcast_to(self.t_off, np.shape(qf) + (3,)), qf) + self.f_off
        return p[..., 0], p[..., 2]

    # ------------------------------------------------------------------ IK
    def plane_ik(self, r, z):
        """(r, z) -> (femur, tibia), knee-up branch (the branch of q_default and the sleep pose).

        Ry(a) acting on (x, z) is a plane rotation by -a: P = R(-qf) [L1 + R(-qt) L2], P = (r - r0, z).
        Raises ValueError if out of reach.
        """
        r, z = np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64)
        px, pz = r - self.r0, z
        d2 = px * px + pz * pz
        cb = (d2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        if np.any(np.abs(cb) > 1.0):
            raise ValueError("toe target out of reach: r=%s z=%s" % (r, z))
        beta = -np.arccos(cb)  # angle of L2 relative to L1 after the knee rotation (negative = knee up)
        chi = beta - self.a2 + self.a1  # = -q_tibia
        # direction of L1 + R(chi) L2 at zero femur angle
        vx = self.L1[0] + np.cos(chi) * self.L2[0] - np.sin(chi) * self.L2[1]
        vz = self.L1[1] + np.sin(chi) * self.L2[0] + np.cos(chi) * self.L2[1]
        psi = np.arctan2(pz, px) - np.arctan2(vz, vx)  # = -q_femur
        psi = (psi + np.pi) % (2 * np.pi) - np.pi
        return -psi, -chi

    def legs_ik(self, r, z, coxa=None):
        """Per-leg (6,) r and z (scalars broadcast) -> q (18,) in sim order."""
        r = np.broadcast_to(np.asarray(r, dtype=np.float64), (6,))
        z = np.broadcast_to(np.asarray(z, dtype=np.float64), (6,))
        qf, qt = self.plane_ik(r, z)
        qc = np.zeros(6) if coxa is None else np.broadcast_to(coxa, (6,))
        return np.concatenate([qc, qf, qt])

    def legs_plane(self, q):
        """q (18,) -> per-leg (r, z) arrays (6,), ignoring coxa."""
        return self.plane_fk(np.asarray(q)[6:12], np.asarray(q)[12:18])

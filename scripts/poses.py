"""Named poses and the stand-up / sit-down sequences for pxmark4. NumPy only, Python 3.8.

Geometry (leg plane, see leg_ik.py; feasible band from the clip limits, 3 deg margin):
  belly on the ground (toe z = -3.3 cm) needs r >= ~15.7 cm (femur clip -98 deg, tibia clip 125 deg);
  toe z = -17 cm allows r <= ~14.3 cm. So no single stance spans both: the body rises with the feet wide (r 16.5 cm)
  to an intermediate height, re-steps the feet in to the stand radius one leg at a time, then rises to STAND.
  One leg at a time (tripod order RF, LM, RB, LF, RM, LB): with a 3-leg tripod at r 16.5 cm each stance femur would
  carry ~1.1 N*m (~75 % of stall, above the 70 % torque limit used for scripted moves).

Poses (coxa 0 throughout):
  SLEEP       vendor sleep_positions (0, -1.57, +1.57): toe r 16.3 cm, z -3.0 cm, belly on the ground
  PLANT       toes r 16.5 cm, z -3.3 cm (level with the belly bottom)
  WIDE_STAND  toes r 16.5 cm, z -12.0 cm (belly 8.7 cm up)
  STAND       q_default exactly (toe r 6.7 cm, z -17.0 cm, belly 13.7 cm up): the policy's start pose

Every segment is cosine-eased and sampled per 20 ms tick; segment durations are stretched so no joint exceeds
RATE_MAX. The sit-down is the stand-up run backwards, segment by segment.
"""
import numpy as np

from leg_ik import LegKin

BELLY_Z = -0.033  # base_bottom joint in the URDF
R_WIDE = 0.165
Z_STEP = -0.12  # toe height (relative to the coxa plane) while re-stepping
LIFT = 0.02
RATE_MAX = 1.2  # rad/s, scripted moves (safety asserts 1.5)
STEP_ORDER = (0, 4, 2, 3, 1, 5)  # RF, LM, RB, LF, RM, LB (LEG_NAMES order RF RM RB LF LM LB)


def ease(s):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0.0, 1.0))


class Segment:
    def __init__(self, name, duration, f, stretch=True):
        """f(s) -> q (18,), s in [0, 1] (f does its own easing)."""
        self.name, self.f = name, f
        if stretch:
            s = np.linspace(0.0, 1.0, 401)
            qs = np.array([f(x) for x in s])
            peak = np.abs(np.diff(qs, axis=0)).max() / (s[1] - s[0])  # rad per unit s
            duration = max(duration, peak / RATE_MAX)
        self.duration = float(duration)

    def q(self, s):
        return self.f(s)

    def reversed(self):
        f = self.f
        return Segment(self.name + "~", self.duration, lambda s: f(1.0 - s), stretch=False)


def joint_ramp(name, q_a, q_b, duration):
    q_a, q_b = np.array(q_a, dtype=float), np.array(q_b, dtype=float)
    return Segment(name, duration, lambda s: q_a + (q_b - q_a) * ease(s))


class Poses:
    def __init__(self, contract):
        self.c = contract
        self.k = LegKin(contract.kin)
        r, z = self.k.legs_plane(contract.q_default)
        assert np.allclose(r, r[0]) and np.allclose(z, z[0])
        self.r_stand, self.z_stand = float(r[0]), float(z[0])
        self.SLEEP = contract.clip(contract.sleep)
        self.PLANT = self.k.legs_ik(R_WIDE, BELLY_Z)
        self.WIDE_STAND = self.k.legs_ik(R_WIDE, Z_STEP)
        self.STAND = contract.q_default.copy()
        self.named = {"sleep": self.SLEEP, "plant": self.PLANT, "wide_stand": self.WIDE_STAND, "stand": self.STAND}
        for n, q in self.named.items():
            assert np.all(q >= contract.lo - 1e-9) and np.all(q <= contract.hi + 1e-9), (n, q)
        self.core = self._standup_core()

    # ------------------------------------------------------------------ segments
    def _vertical(self, name, r, z_a, z_b, duration, end=None):
        k = self.k

        def f(s):
            if end is not None and s >= 1.0:
                return end.copy()
            return k.legs_ik(r, z_a + (z_b - z_a) * ease(s))

        return Segment(name, duration, f)

    def _step(self, leg, r_before, r_to, z, duration):
        """Leg ``leg`` swings from r_before[leg] to r_to (lift LIFT); the others hold r_before. Phases: lift
        [0, 0.25], move [0.1, 0.9], lower [0.75, 1]."""
        k, r_b = self.k, np.array(r_before, dtype=float)

        def f(s):
            r = r_b.copy()
            zz = np.full(6, z)
            r[leg] = r_b[leg] + (r_to - r_b[leg]) * ease((s - 0.1) / 0.8)
            zz[leg] = z + LIFT * (ease(s / 0.25) if s < 0.5 else ease((1.0 - s) / 0.25))
            return k.legs_ik(r, zz)

        return Segment("step_%s" % self.c.raw["leg_names"][leg], duration, f)

    def _standup_core(self):
        """SLEEP -> STAND (excluding the initial ramp from wherever the robot is to SLEEP)."""
        segs = [joint_ramp("plant", self.SLEEP, self.PLANT, 1.0),
                self._vertical("push_up", R_WIDE, BELLY_Z, Z_STEP, 2.5)]
        r = np.full(6, R_WIDE)
        for leg in STEP_ORDER:
            segs.append(self._step(leg, r, self.r_stand, Z_STEP, 0.8))
            r = r.copy()
            r[leg] = self.r_stand
        segs.append(self._vertical("rise", self.r_stand, Z_STEP, self.z_stand, 1.5, end=self.STAND))
        return segs

    # ------------------------------------------------------------------ plans
    def to_pose(self, name, q_from, q_to, min_duration=2.0, rate=0.6):
        """Joint-space ramp from an arbitrary (read) pose; duration from the largest joint distance."""
        d = float(np.abs(np.asarray(q_to) - np.asarray(q_from)).max())
        return joint_ramp(name, q_from, q_to, max(min_duration, d / rate * np.pi / 2))

    def standup(self, q_read, slow=False):
        return [self.to_pose("to_sleep", q_read, self.SLEEP, 4.0 if slow else 2.0, 0.3 if slow else 0.6)] + self.core

    def sitdown_from_stand(self):
        return [s.reversed() for s in reversed(self.core)]

    def sitdown_from_core(self, k, q_read):
        """Interrupted inside core segment k: ramp back to that segment's start, then run segments k-1 .. 0
        backwards (the exact reverse of the path taken so far)."""
        start = self.core[k].q(0.0)
        return [self.to_pose("back_%s" % self.core[k].name, q_read, start, 1.0)] + \
            [s.reversed() for s in reversed(self.core[:k])]


class Motion:
    """Plays a segment list at dt per tick. next() -> q (18,) or None when finished."""

    def __init__(self, segments, dt):
        self.segs, self.dt = list(segments), dt
        self.i, self.j = 0, 0  # segment, tick within it

    @property
    def segment(self):
        return self.segs[self.i] if self.i < len(self.segs) else None

    def next(self):
        """Segment of n = ceil(duration / dt) ticks yields s = 1/n .. 1 (s = 0 is the previous segment's end)."""
        while self.i < len(self.segs):
            seg = self.segs[self.i]
            n = max(1, int(np.ceil(seg.duration / self.dt - 1e-9)))
            if self.j < n:
                self.j += 1
                return seg.q(self.j / float(n))
            self.i += 1
            self.j = 0
        return None

    def done(self):
        return self.i >= len(self.segs)

    def ticks(self):
        """All q of the motion (for tests and renders)."""
        m = Motion(self.segs, self.dt)
        out = []
        while True:
            q = m.next()
            if q is None:
                return out
            out.append((m.i, q))

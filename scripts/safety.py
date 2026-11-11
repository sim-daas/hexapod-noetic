"""Runtime monitors for the teleop runner. NumPy only, Python 3.8.

Safety.step(...) is called once per control step and returns (level, reason) with level OK / WARN / FAULT, plus
KILL for a fall (tilt beyond KILL_TILT): torque off at once, sitting down would be pointless. Thresholds are module
constants so the offline tests and the log say exactly what tripped.
"""
from collections import deque

import numpy as np

OK, WARN, FAULT, KILL = 0, 1, 2, 3
LEVEL_NAME = {OK: "OK", WARN: "WARN", FAULT: "FAULT", KILL: "KILL"}

TILT_WARN, TILT_FAULT, KILL_TILT = 20.0, 30.0, 60.0  # deg, projected-gravity angle from level
MAX_CONSEC_DROPS = 5
DROP_RATE_MAX, DROP_WINDOW = 0.20, 50  # per servo over 1 s
IMU_STALE_S = 0.060
OVERRUN_S, OVERRUN_COUNT, OVERRUN_WINDOW = 0.025, 5, 50
POLICY_STEP_MAX = 0.5  # rad per step, gross-error clamp on the policy's q_cmd
SCRIPT_RATE_MAX = 1.5  # rad/s, scripted moves
TRACK_ERR, TRACK_HOLD_S = 0.6, 0.5
TEMP_WARN, TEMP_FAULT = 65.0, 70.0
V_WARN, V_FAULT = 10.8, 10.5
V_WINDOW = 18  # health reads (one servo per step): median over the last full round
BUS_DEAD_STEPS = 10


def tilt_deg(gravity):
    g = np.asarray(gravity, dtype=float)
    n = np.linalg.norm(g)
    if not np.isfinite(n) or n < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(-g[2] / n, -1.0, 1.0))))


class Safety:
    def __init__(self, n_joints, dt):
        self.n, self.dt = n_joints, dt
        self.consec = np.zeros(n_joints, dtype=int)
        self.drop_hist = deque(maxlen=DROP_WINDOW)
        self.overruns = deque(maxlen=OVERRUN_WINDOW)
        self.track_t = np.zeros(n_joints)
        self.volts = deque(maxlen=V_WINDOW)
        self.temps = {}
        self.all_dead = 0
        self.imu_t = None
        self.last = (OK, "")

    # ------------------------------------------------------------------ inputs
    def health(self, sid, volts, temp):
        self.volts.append(volts)
        self.temps[sid] = temp

    def imu_ok(self, t):
        self.imu_t = t

    @property
    def v_batt(self):
        return float(np.median(self.volts)) if self.volts else float("nan")

    @property
    def max_temp(self):
        return max(self.temps.values()) if self.temps else float("nan")

    def drop_rate(self):
        return np.mean(self.drop_hist, axis=0) if self.drop_hist else np.zeros(self.n)

    # ------------------------------------------------------------------ check
    def step(self, t, valid, q, q_cmd, gravity, period, torque_on, servo_alarm=None):
        """t: monotonic now; valid (n,) bool; q (n,) ZOH angles; q_cmd (n,) or None; gravity (3,) or None;
        period: time since the previous step started (s); servo_alarm: str or None."""
        valid = np.asarray(valid, dtype=bool)
        self.consec = np.where(valid, 0, self.consec + 1)
        self.drop_hist.append(~valid)
        self.overruns.append(period > OVERRUN_S)
        self.all_dead = self.all_dead + 1 if not valid.any() else 0
        res = []

        if self.all_dead >= BUS_DEAD_STEPS:
            res.append((KILL, "bus dead: no servo answered for %d steps" % self.all_dead))
        if gravity is not None:
            tl = tilt_deg(gravity)
            if not np.isfinite(tl):
                res.append((FAULT, "gravity estimate not finite"))
            elif tl > KILL_TILT and torque_on:
                res.append((KILL, "tilt %.0f deg: fallen" % tl))
            elif tl > TILT_FAULT:
                res.append((FAULT, "tilt %.0f deg" % tl))
            elif tl > TILT_WARN:
                res.append((WARN, "tilt %.0f deg" % tl))
        if self.imu_t is None or t - self.imu_t > IMU_STALE_S:
            res.append((FAULT, "IMU stale %.0f ms" % (1e3 * (t - self.imu_t)) if self.imu_t else "IMU never read"))
        bad = np.nonzero(self.consec > MAX_CONSEC_DROPS)[0]
        if len(bad):
            res.append((FAULT, "servo idx %s: >%d consecutive dropped reads" % (bad.tolist(), MAX_CONSEC_DROPS)))
        if len(self.drop_hist) == DROP_WINDOW:
            rate = self.drop_rate()
            if rate.max() > DROP_RATE_MAX:
                res.append((FAULT, "servo idx %d: %.0f %% dropped reads over 1 s" % (rate.argmax(), 100 * rate.max())))
        if sum(self.overruns) >= OVERRUN_COUNT:
            res.append((FAULT, "loop overrun > %.0f ms on %d of the last %d steps"
                        % (1e3 * OVERRUN_S, sum(self.overruns), len(self.overruns))))
        if torque_on and q_cmd is not None:
            err = np.abs(np.asarray(q_cmd) - np.asarray(q)) * valid
            self.track_t = np.where(err > TRACK_ERR, self.track_t + self.dt, 0.0)
            j = int(self.track_t.argmax())
            if self.track_t[j] >= TRACK_HOLD_S:
                res.append((FAULT, "joint idx %d: tracking error %.2f rad for %.1f s" % (j, err[j], self.track_t[j])))
        else:
            self.track_t[:] = 0.0
        if self.temps:
            tm = self.max_temp
            if tm >= TEMP_FAULT:
                res.append((FAULT, "servo temperature %.0f C" % tm))
            elif tm >= TEMP_WARN:
                res.append((WARN, "servo temperature %.0f C" % tm))
        if len(self.volts) >= 3:
            v = self.v_batt
            if v < V_FAULT:
                res.append((FAULT, "battery %.1f V" % v))
            elif v < V_WARN:
                res.append((WARN, "battery %.1f V" % v))
        if servo_alarm and torque_on:
            res.append((FAULT, "servo alarm: %s" % servo_alarm))
        self.last = max(res, key=lambda r: r[0]) if res else (OK, "")
        return self.last

    @staticmethod
    def check_policy(q_cmd, q_cmd_prev, a):
        """Policy-path checks (no rate limit, only gross errors). Returns (level, reason)."""
        if not np.all(np.isfinite(a)):
            return FAULT, "policy output not finite"
        if q_cmd_prev is not None:
            d = np.abs(np.asarray(q_cmd) - np.asarray(q_cmd_prev))
            if d.max() > POLICY_STEP_MAX:
                return FAULT, "policy q_cmd jump %.2f rad on joint idx %d" % (d.max(), d.argmax())
        return OK, ""

    def check_script(self, q_cmd, q_cmd_prev):
        d = np.abs(np.asarray(q_cmd) - np.asarray(q_cmd_prev)).max()
        assert d <= SCRIPT_RATE_MAX * self.dt + 1e-9, "scripted move too fast: %.3f rad in one step" % d

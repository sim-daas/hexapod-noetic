"""Motion-session checks for the calibrated IMU (pure NumPy, used by imu_viz.py and the tests).

Main check, gyro <-> accel consistency: take the body-frame "up" direction from the accelerometer at time t, rotate it
forward by the gyro alone over TAU seconds, and compare with the accelerometer's "up" at t + TAU. If the calibration's
axes, signs or handedness were wrong for either sensor, the two disagree by tens of degrees; if they agree to a few
degrees while the robot is tilted and rotated about every axis, the body-frame mapping of both sensors is right.
Quasi-static windows only (| |a| - g | small), because the accelerometer measures gravity only when not accelerating.
"""
import numpy as np

from imu_synth import rot_exp

G0 = 9.80665
TAU = 0.5


def _unit_rows(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def analyse(t, ab, w, tau=TAU, g_tol=1.0, w_min=0.15):
    """t (n,), ab (n,3) calibrated body-frame specific force at the sensor, w (n,3) bias-corrected body-frame gyro."""
    t, ab, w = np.asarray(t, float), np.asarray(ab, float), np.asarray(w, float)
    n = len(t)
    res = dict(n=n, duration=float(t[-1] - t[0]) if n > 1 else 0.0)
    if n < 10:
        res.update(windows=0)
        return res
    u = _unit_rows(ab)
    dt = np.diff(t, prepend=t[0])
    quasi = np.abs(np.linalg.norm(ab, axis=1) - G0) < g_tol
    errs = []
    for i in range(0, n, 3):
        j = int(np.searchsorted(t, t[i] + tau))
        if j >= n:
            break
        if not (quasi[i] and quasi[j]) or np.linalg.norm(w[i:j + 1], axis=1).mean() < w_min:
            continue
        v = u[i]
        for k in range(i + 1, j + 1):
            v = rot_exp(-w[k] * dt[k]) @ v  # the up vector rotates by -w dt in the body frame
        errs.append(np.degrees(np.arccos(np.clip(v @ u[j], -1, 1))))
    errs = np.array(errs)
    res["windows"] = len(errs)
    if len(errs):
        res.update(err_med=float(np.median(errs)), err_p95=float(np.percentile(errs, 95)), err_max=float(errs.max()))
    res["peak_w"] = np.abs(w).max(0).tolist()  # rad/s per body axis x, y, z
    res["nose_up_max"] = float(np.degrees(np.arcsin(np.clip(u[:, 0].max(), -1, 1))))
    res["nose_down_max"] = float(np.degrees(np.arcsin(np.clip(-u[:, 0].min(), -1, 1))))
    res["left_up_max"] = float(np.degrees(np.arcsin(np.clip(u[:, 1].max(), -1, 1))))
    res["right_up_max"] = float(np.degrees(np.arcsin(np.clip(-u[:, 1].min(), -1, 1))))
    yaw = np.cumsum(np.einsum("ij,ij->i", w, u) * dt)  # rotation about the vertical, CCW seen from above = +
    res["yaw_ccw_max"], res["yaw_cw_max"] = float(np.degrees(yaw.max())), float(np.degrees(-yaw.min()))
    return res


def verdict(res, max_med=2.0, max_p95=4.0, min_windows=30, min_peak_w=0.5, min_tilt=12.0, min_yaw=60.0):
    """-> [(label, ok, detail)]. 'required' items decide PASS; coverage items say what still needs doing."""
    out = []
    if res.get("windows", 0) < min_windows:
        out.append(("gyro<->accel consistency", None, "need more motion (%d windows, want %d)" %
                    (res.get("windows", 0), min_windows)))
    else:
        ok = res["err_med"] <= max_med and res["err_p95"] <= max_p95
        out.append(("gyro<->accel consistency", ok, "med %.1f p95 %.1f deg, n=%d (limit %.0f/%.0f)" %
                    (res["err_med"], res["err_p95"], res["windows"], max_med, max_p95)))
    if "peak_w" in res:
        for ax, pw in zip("xyz", res["peak_w"]):
            out.append(("rotated about body %s" % ax, pw >= min_peak_w, "peak %.2f rad/s (need >= %.1f)" % (pw, min_peak_w)))
        out.append(("nose up", res["nose_up_max"] >= min_tilt, "%.0f deg" % res["nose_up_max"]))
        out.append(("nose down", res["nose_down_max"] >= min_tilt, "%.0f deg" % res["nose_down_max"]))
        out.append(("left side up", res["left_up_max"] >= min_tilt, "%.0f deg" % res["left_up_max"]))
        out.append(("right side up", res["right_up_max"] >= min_tilt, "%.0f deg" % res["right_up_max"]))
        out.append(("yaw CCW (seen from above)", res["yaw_ccw_max"] >= min_yaw, "%.0f deg" % res["yaw_ccw_max"]))
        out.append(("yaw CW (seen from above)", res["yaw_cw_max"] >= min_yaw, "%.0f deg" % res["yaw_cw_max"]))
    return out


def passed(items):
    """Consistency is required to be OK; coverage items must not be False."""
    return all(ok is True for label, ok, _ in items if label.startswith("gyro")) and all(
        ok is not False for _, ok, _ in items)


def report(res):
    items = verdict(res)
    lines = ["IMU motion check  (%d samples, %.0f s)" % (res["n"], res["duration"])]
    for label, ok, detail in items:
        lines.append("  [%s] %-36s %s" % ({True: "OK", False: "--", None: "??"}[ok], label, detail))
    lines.append("RESULT: " + ("PASS" if passed(items) else "NOT YET (see [--] / [??] items)"))
    return "\n".join(lines)

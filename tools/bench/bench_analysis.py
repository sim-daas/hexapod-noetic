"""Summaries of the servo bench logs (tools/bench/servo_bench.py). NumPy only, Python 3.8; runs on the Pi and the
workstation. analyse(stem) -> dict, format_summary(dict) -> str. Cross-run results (PWM -> torque from a hold with and
without a known mass) are in analyze_bench.py.

Model the numbers feed (hexasim actuators/xl430.py): tau = tau_s(V) (duty - w / w_nl(V)) - tau_c sign(w), duty =
clip(P/128 err_ticks / 885, +-Goal PWM / 885). From the logs:
  speed  w_nl at duty 1 (and V), Coulomb duty d_c (tau_c = d_c tau_s), mechanical time constant t_m = J w_nl / tau_s
  step   dead time (bus + firmware latency), 10-90 % rise, overshoot, final error, peak speed / PWM per (cap, size)
  sine   amplitude ratio and phase lag per frequency (compare hexasim 2.4.2: 0.93 / 0.66 / 0.31 / 0.06 at
         1.5 / 8 / 12.5 / 25 Hz for J 0.004)
  hold   static PWM vs angle (gravity moment, friction = half the up/down gap), P-law check (measured / predicted PWM)
  drops  dropped-read rate per servo and per command step size
  stall  time to the overload alarm per (cap, requested PWM)
  ground static PWM per joint in STAND, the policy stance (both caps) and the tripods; voltage under load
"""
import json
import math

import numpy as np

TICK = 2.0 * math.pi / 4096.0
KP = 1061


def load(stem):
    d = np.load(stem + ".npz")
    a = {k: d[k] for k in d.files}
    meta = json.load(open(stem + ".json"))
    return a, meta


def _r(x, n=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return None if not np.isfinite(x) else round(x, n)


def analyse(stem):
    a, m = load(stem)
    fn = {"speed": speed, "step": step, "sine": sine, "hold": hold, "drops": drops, "stall": stall,
          "ground": ground}[m["test"]]
    out = fn(a, m)
    out.update(test=m["test"], stem=stem, joint=m.get("names", [None] * 18)[m["joint"]] if "joint" in m else None,
               V_median=_r(np.nanmedian(a["V"]) if "V" in a and np.isfinite(a["V"]).any() else float("nan"), 2),
               V_min=_r(np.nanmin(a["V"]) if "V" in a and np.isfinite(a["V"]).any() else float("nan"), 2))
    return out


# ------------------------------------------------------------------------------------------------------ speed
def speed(a, m):
    rows = []
    for sg in m["segs"]:
        sel = (a["seg"] == sg["seg"]) & (a["phase"] == 1)
        if sel.sum() < 5:
            continue
        t, v, q = a["t"][sel], a["vel"][sel] * sg["sign"], a["q"][sel]
        T = t[-1] - t[0]
        late = t >= t[0] + 0.6 * T
        v_ss = float(np.median(v[late])) if late.sum() >= 3 else float("nan")
        dq = np.diff(q) / np.maximum(np.diff(t), 1e-4) * sg["sign"]
        v_pos = float(np.median(dq[late[1:]])) if late[1:].sum() >= 3 else float("nan")
        t63 = float("nan")
        if np.isfinite(v_ss) and v_ss > 0.05:
            hit = np.where(v >= 0.632 * v_ss)[0]
            if len(hit):
                t63 = float(t[hit[0]] - t[0])
        rows.append({"duty": sg["duty"], "sign": sg["sign"], "stuck": sg["stuck"], "v_ss": v_ss, "v_pos": v_pos,
                     "t63": t63, "travel": float(abs(q[-1] - q[0])), "V": float(np.nanmedian(a["V"][sel]))})
    mv = [r for r in rows if not r["stuck"] and np.isfinite(r["v_ss"]) and r["v_ss"] > 0.1 and r["duty"] >= 0.1]
    fit = {}
    if len(set(r["duty"] for r in mv)) >= 2:
        d = np.array([r["duty"] for r in mv])
        v = np.array([r["v_ss"] for r in mv])
        A = np.c_[d, np.ones_like(d)]
        (k, c0), *_ = np.linalg.lstsq(A, v, rcond=None)
        fit = {"w_nl_rad_s": _r(k), "w_nl_rpm": _r(k * 60 / (2 * math.pi), 1), "coulomb_duty": _r(-c0 / k, 4),
               "fit_rms_rad_s": _r(np.sqrt(np.mean((A @ [k, c0] - v) ** 2)))}
        hi = [r for r in mv if r["duty"] >= 0.3 and np.isfinite(r["t63"])]
        if hi:
            fit["t63_s_median_duty>=0.3"] = _r(np.median([r["t63"] for r in hi]), 4)
        for sgn in (1.0, -1.0):
            vs = [r["v_ss"] for r in mv if r["sign"] == sgn and r["duty"] == 1.0]
            if vs:
                fit["v_at_duty1_%s" % ("pos" if sgn > 0 else "neg")] = _r(np.median(vs))
    stuck = sorted(set(r["duty"] for r in rows if r["stuck"]))
    return {"rows": [{k: _r(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows], "fit": fit,
            "no_motion_duties": stuck}


# ------------------------------------------------------------------------------------------------------ step
def step(a, m):
    rows = []
    for sg in m["segs"]:
        sel = a["seg"] == sg["seg"]
        if sel.sum() < 5:
            continue
        t, q, v, p = a["t"][sel], a["q"][sel], a["vel"][sel], a["pwm"][sel]
        g = sg["goal"]
        q0 = sg["from"] if np.isfinite(sg["from"]) else q[0]
        D = g - q0
        r = {"cap": sg["cap"], "amp": sg["amp"], "D": D, "final_err": float(np.mean(g - q[t >= t[-1] - 0.2]))}
        if abs(D) >= 0.015:
            s = np.sign(D)
            prog = (q - q0) * s / abs(D)
            tw = sg["t_write"]
            mv = np.where(np.abs(q - q0) >= 3 * TICK)[0]
            r["dead_ms"] = float((t[mv[0]] - tw) * 1e3) if len(mv) else float("nan")
            i10, i90 = np.where(prog >= 0.1)[0], np.where(prog >= 0.9)[0]
            r["rise_ms"] = float((t[i90[0]] - t[i10[0]]) * 1e3) if len(i10) and len(i90) else float("nan")
            r["t90_ms"] = float((t[i90[0]] - tw) * 1e3) if len(i90) else float("nan")
            r["overshoot"] = float(max(0.0, (prog.max() - 1.0)))
            r["peak_vel"] = float(np.abs(v).max())
            r["peak_pwm"] = float(np.abs(p).max())
        rows.append(r)
    agg = {}
    for cap in sorted(set(r["cap"] for r in rows)):
        for amp in sorted(set(r["amp"] for r in rows)):
            rr = [r for r in rows if r["cap"] == cap and r["amp"] == amp and "dead_ms" in r]
            if not rr:
                continue
            med = lambda k: _r(np.nanmedian([x[k] for x in rr]), 4)  # noqa: E731
            agg["cap%d_amp%.2f" % (cap, amp)] = {"n": len(rr), "dead_ms": med("dead_ms"), "rise_ms": med("rise_ms"),
                                                 "t90_ms": med("t90_ms"), "overshoot": med("overshoot"),
                                                 "final_err_mrad": _r(1e3 * np.median([abs(x["final_err"]) for x in rr]), 2),
                                                 "peak_vel": med("peak_vel"), "peak_pwm": med("peak_pwm")}
    return {"by_cap_amp": agg}


# ------------------------------------------------------------------------------------------------------ sine
def _fit(t, y, f):
    A = np.c_[np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t), np.ones_like(t)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return complex(c[0], -c[1])


def sine(a, m):
    rows = []
    for sg in m["segs"]:
        sel = a["seg"] == sg["seg"]
        t = a["t"][sel]
        if len(t) < 20:
            continue
        keep = t >= t[0] + min(1.0, 0.25 * (t[-1] - t[0]))  # drop the start transient
        t, q, cmd = t[keep] - t[keep][0], a["q"][sel][keep], a["cmd"][sel][keep]
        f = sg["f"]
        Q, C = _fit(t, q, f), _fit(t, cmd, f)
        lag = -math.degrees(np.angle(Q / C)) if abs(C) > 1e-9 else float("nan")
        rows.append({"f": f, "gain_vs_A": _r(abs(Q) / m["A"]), "gain_vs_cmd": _r(abs(Q) / abs(C) if abs(C) else float("nan")),
                     "lag_deg": _r(lag, 1), "lag_ms": _r(lag / 360.0 / f * 1e3, 1),
                     "peak_pwm": _r(np.abs(a["pwm"][sel]).max())})
    return {"rows": rows, "A": m["A"], "cap": m["cap"]}


# ------------------------------------------------------------------------------------------------------ hold
def hold(a, m):
    rows = []
    for sg in m["segs"]:
        sel = a["seg"] == sg["seg"]
        t = a["t"][sel]
        if len(t) < 5:
            continue
        late = t >= t[-1] - 1.0
        err = a["cmd"][sel][late] - a["q"][sel][late]
        p = a["pwm"][sel][late]
        pred = np.clip(KP / 128.0 * np.mean(err) / TICK / 885.0, -1, 1)
        rows.append({"angle": sg["angle"], "from": sg["from"], "arm": sg["arm"], "pwm": float(np.mean(p)),
                     "pwm_std": float(np.std(p)), "err_mrad": float(1e3 * np.mean(err)), "pred_pwm": float(pred)})
    by = {}
    for r in rows:
        by.setdefault(round(r["angle"], 4), {})[r["from"]] = r
    table = []
    for ang in sorted(by):
        lo, hi = by[ang].get(-1.0), by[ang].get(1.0)
        if lo and hi:
            table.append({"angle": ang, "arm_m": _r(lo["arm"], 4), "pwm_mean": _r((lo["pwm"] + hi["pwm"]) / 2, 4),
                          "friction_duty": _r(abs(lo["pwm"] - hi["pwm"]) / 2, 4),
                          "err_mrad_mean": _r((lo["err_mrad"] + hi["err_mrad"]) / 2, 2)})
    p = np.array([r["pwm"] for r in rows])
    pr = np.array([r["pred_pwm"] for r in rows])
    ok = np.abs(pr) > 0.02
    plaw = _r(np.median(p[ok] / pr[ok]), 3) if ok.sum() >= 3 else None
    return {"table": table, "mass": m.get("mass", 0.0), "mass_label": m.get("mass_label", ""),
            "p_law_measured_over_predicted": plaw,
            "friction_duty_median": _r(np.median([r["friction_duty"] for r in table]), 4) if table else None}


# ------------------------------------------------------------------------------------------------------ drops
def drops(a, m):
    rows = []
    for sg in m["segs"]:
        sel = a["seg"] == sg["seg"]
        if sel.sum() == 0:
            continue
        v = a["valid"][sel].astype(bool)
        cul = a["rd"][sel][:, 4].astype(int)
        cc = {}
        for c in cul[cul > 0]:
            cc[int(c)] = cc.get(int(c), 0) + 1
        rows.append({"amp": sg["amp"], "cap": sg["cap"], "steps": int(sel.sum()),
                     "drop_rate": _r(1.0 - v.mean(), 5), "steps_with_drop": _r((~v).any(1).mean(), 4),
                     "chain_breaks": int((cul > 0).sum()), "culprits": dict(sorted(cc.items(), key=lambda kv: -kv[1])[:4]),
                     "V_min": _r(np.nanmin(a["V"][sel]) if np.isfinite(a["V"][sel]).any() else float("nan"), 2)})
    return {"rows": rows}


# ------------------------------------------------------------------------------------------------------ stall
def stall(a, m):
    rows = []
    fem = ~np.isfinite(a["tib_pwm"])
    for sg in m["segs"]:
        sel = (a["seg"] == sg["seg"]) & fem
        r = {"cap": sg["cap"], "req": sg["req"], "outcome": sg["outcome"], "t_trip_s": _r(sg["t_trip"], 2)}
        if sel.sum():
            r.update(pwm_median=_r(np.median(np.abs(a["pwm"][sel]))), T_start=_r(a["T"][sel][0], 0),
                     T_end=_r(a["T"][sel][-1], 0), V_min=_r(np.nanmin(a["V"][sel]), 2),
                     err_mrad=_r(1e3 * np.median(np.abs(a["cmd"][sel] - a["q"][sel])), 1))
        tib = (a["seg"] == sg["seg"]) & ~fem
        if tib.sum():
            r["tibia_pwm_max"] = _r(np.abs(a["tib_pwm"][tib]).max())
        rows.append(r)
    return {"rows": rows}


# ------------------------------------------------------------------------------------------------------ ground
def ground(a, m):
    names = m["names"]
    short = [n.replace("right_", "R").replace("left_", "L").replace("front", "F").replace("middle", "M")
             .replace("back", "B").replace("_", " ") for n in names]
    rows = []
    for ph in m["phases"]:
        sel = (a["t"] >= ph["t0"]) & (a["t"] <= ph["t1"])
        if sel.sum() < 3:
            continue
        t = a["t"][sel]
        late = t >= t[0] + 0.3 * (t[-1] - t[0])
        p = np.abs(a["pwm"][sel][late])
        mean = np.nanmean(p, 0)
        top = np.argsort(-mean)[:4]
        rows.append({"phase": ph["name"], "cap": ph["cap"], "outcome": ph.get("outcome", ""),
                     "pwm_mean_by_type": {"coxa": _r(np.nanmean(mean[:6])), "femur": _r(np.nanmean(mean[6:12])),
                                          "tibia": _r(np.nanmean(mean[12:]))},
                     "top4": {short[k]: _r(mean[k]) for k in top},
                     "pwm_max": _r(np.nanmax(p)), "err_max_mrad": _r(1e3 * np.nanmax(np.abs(a["q_cmd"][sel] - a["q"][sel])), 0),
                     "drop_rate": _r(1 - a["valid"][sel].mean(), 4),
                     "V_min": _r(np.nanmin(a["V"][sel]) if np.isfinite(a["V"][sel]).any() else float("nan"), 2)})
    return {"rows": rows, "tilt_max_deg": _r(m.get("tilt_max", float("nan")), 1)}


# ------------------------------------------------------------------------------------------------------ text
def format_summary(o):
    L = ["== %s %s  (V median %s, min %s)" % (o["test"], o.get("joint") or "", o.get("V_median"), o.get("V_min"))]
    t = o["test"]
    if t == "speed":
        L.append("  fit: %s" % o["fit"])
        if o["no_motion_duties"]:
            L.append("  no motion at duty %s" % o["no_motion_duties"])
        for r in o["rows"]:
            L.append("  duty %4.2f %+2.0f  v_ss %s rad/s  t63 %s s  travel %s" % (r["duty"], r["sign"], r["v_ss"],
                                                                              r["t63"], r["travel"]))
    elif t == "step":
        for k, r in o["by_cap_amp"].items():
            L.append("  %-16s dead %s ms  rise %s ms  t90 %s ms  overshoot %s  final err %s mrad  peak vel %s pwm %s" % (
                k, r["dead_ms"], r["rise_ms"], r["t90_ms"], r["overshoot"], r["final_err_mrad"], r["peak_vel"],
                r["peak_pwm"]))
    elif t == "sine":
        for r in o["rows"]:
            L.append("  f %5.1f Hz  gain %s (vs sent fundamental %s)  lag %s deg = %s ms  peak pwm %s" % (
                r["f"], r["gain_vs_A"], r["gain_vs_cmd"], r["lag_deg"], r["lag_ms"], r["peak_pwm"]))
    elif t == "hold":
        L.append("  mass %s kg %s, P-law measured/predicted %s, friction duty median %s" % (
            o["mass"], o["mass_label"], o["p_law_measured_over_predicted"], o["friction_duty_median"]))
        for r in o["table"]:
            L.append("  angle %+.3f  arm %s m  pwm %s  friction %s  err %s mrad" % (r["angle"], r["arm_m"], r["pwm_mean"],
                                                                                r["friction_duty"], r["err_mrad_mean"]))
    elif t == "drops":
        for r in o["rows"]:
            L.append("  cap %d |dq| %.2f: drop rate %s, steps with a drop %s, chain breaks %d, culprits %s, V min %s" % (
                r["cap"], r["amp"], r["drop_rate"], r["steps_with_drop"], r["chain_breaks"], r["culprits"], r["V_min"]))
    elif t == "stall":
        for r in o["rows"]:
            L.append("  cap %3.0f %% req %3.0f %%: %-16s t_trip %s s  pwm %s  err %s mrad  T %s->%s C  V min %s  tibia %s" % (
                100 * r["cap"], 100 * r["req"], r["outcome"], r["t_trip_s"], r.get("pwm_median"), r.get("err_mrad"),
                r.get("T_start"), r.get("T_end"), r.get("V_min"), r.get("tibia_pwm_max")))
    elif t == "ground":
        L.append("  tilt max %s deg" % o["tilt_max_deg"])
        for r in o["rows"]:
            L.append("  %-34s cap %3d %-14s |pwm| %s  top %s  max %s  err max %s mrad  drops %s  V min %s" % (
                r["phase"], r["cap"], r["outcome"], r["pwm_mean_by_type"], r["top4"], r["pwm_max"], r["err_max_mrad"],
                r["drop_rate"], r["V_min"]))
    return "\n".join(L)

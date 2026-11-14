"""Workstation (env_isaaclab): write config/contract_<tag>.json + the model files for the robot runner, from hexasim
code. Blind (arm B0) S1-family checkpoints: S1 (1,125-D, clock on) and S2 / S1H4 (1,269-D), AX-12A or XL430 model.

    ~/miniconda3/envs/env_isaaclab/bin/python tools/policy/export_contract.py [checkpoint [tag]]

Tag s1h4 (default) writes into models/, any other tag into models/<tag>/ (scripts/joint_map.py: policy_files).
Outputs (all generated, never hand-edited):
  config/contract_<tag>.json  joint order, q_default, clip, scale, slew limit, dt, W, obs layout, servo tick, clock,
                              command set, FK constants, servo IDs + sleep pose (by name, from the vendor pxmark4.yaml)
  control.onnx                fused control graph (tools/s1_export/export_fused.py), hist (1,W,111) + obs_base → a
  fused_io.npz                random I/O sample of the fused graph
  parity_ref.npz              a 120-step sequence run through hexasim's own ProprioReader, S1Aux and actor, with the
                              env's history fill / obs assembly / clock / slew limit: the robot-side NumPy code must
                              reproduce it
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
HEXASIM = Path("/home/ubuntu/gitrepos/hexasim")
CK = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    HEXASIM / "logs/rsl_rl/hexapod_s1h4/2026-10-16_21-12-36_s1h4x_B0_seed43/model_10998.pt")
TAG = sys.argv[2] if len(sys.argv) > 2 else "s1h4"
CONTRACT = REPO / "config" / ("contract_%s.json" % TAG)
MODELS = REPO / "models" if TAG == "s1h4" else REPO / "models" / TAG
VENDOR_YAML = Path("/home/ubuntu/gitrepos/interbotix_ros_crawlers/interbotix_ros_xshexapods/"
                   "interbotix_xshexapod_control/config/pxmark4.yaml")

from hexasim.actuators import ax12a, xl430  # noqa: E402
from hexasim.rl.s1 import kinematics as kin  # noqa: E402
from hexasim.rl.s1.aux import AuxCfg, S1Aux  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import layout  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.proprio import ProprioReader  # noqa: E402


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda l, s, n: None)


def find(d, key):
    """First value under ``key`` anywhere in a nested dict (the dumped env.yaml)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def main():
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    s1 = ck["s1_state"]
    env = yaml.load(open(CK.parent / "params" / "env.yaml"), Loader=_Loader)
    assert s1["gate_iter"] >= 0 and s1["arm"] == "B0", s1
    assert s1["v0_drop_p"] == 1.0, "the runner feeds V_0 as zeros"
    assert s1["clock_drop_p"] in (0.0, 1.0), "the runner feeds the clock always or never"
    clock_on = 1.0 - s1["clock_drop_p"]

    names = ["%s_%s" % (leg, t) for t in ("coxa", "femur", "tibia") for leg in kin.LEGS]
    jp = env["actions"]["joint_pos"]
    assert jp["scale"] == 0.5 and jp["use_default_offset"], jp
    lo, hi = [], []
    for n in names:
        t = n.rsplit("_", 1)[1]
        a, b = jp["clip"][".*_" + t]
        lo.append(a)
        hi.append(b)
    init = env["scene"]["robot"]["init_state"]["joint_pos"]
    q_default = [init[".*_" + n.rsplit("_", 1)[1]] for n in names]
    assert np.allclose(q_default, ck["aux_state_dict"]["q_default"].numpy(), atol=1e-6)
    dt = env["sim"]["dt"] * env["decimation"]
    cmd = env["commands"]["base_velocity"]

    # URDF joint limits (hexasim's URDF, the one the sim and kinematics use).
    import xml.etree.ElementTree as ET
    root = ET.parse(kin.URDF).getroot()
    urdf_lo, urdf_hi = [], []
    for n in names:
        lim = root.find("joint[@name='%s']/limit" % n)
        urdf_lo.append(float(lim.get("lower")))
        urdf_hi.append(float(lim.get("upper")))

    vend = yaml.safe_load(open(VENDOR_YAML))
    servo_id = {n: vend["motors"][n]["ID"] for n in names}
    sleep = dict(zip(vend["joint_order"], vend["sleep_positions"]))

    actor_dim = env["actor_dim"]
    lay = layout.offsets(layout.BLOCKS_BY_DIM[actor_dim])
    assert (lay["v0"], lay["twist_cmd"], lay["z_p"], lay["e_hat"]) == (
        slice(111, 114), slice(114, 117), slice(117, 181), slice(181, 196)), "export_fused.py's slots"
    ((servo_name, sv),) = env["scene"]["robot"]["actuators"].items()
    if servo_name == "ax12a":
        tick = ax12a.TICK
        # Nominal servo settings (DR centre; the run randomised slope over {16, 32, 64}).
        servo = {"compliance_margin": sv["compliance_margin"], "compliance_slope": sv["compliance_slope"],
                 "slope_choices": list(sv["compliance_slope_choices"] or []), "punch": int(round(sv["punch"] * 1023))}
    else:
        assert servo_name == "xl430", servo_name
        tick = xl430.TICK
        # Nominal servo settings (DR centre) the runner must write: position P gain, Goal PWM under the policy.
        servo = {"p_gain": sv["p_gain"], "p_gain_choices": list(sv["p_gain_choices"] or []), "goal_pwm": sv["goal_pwm"],
                 "voltage_range": list(sv["voltage_range"] or []), "delay_steps": [sv["min_delay"], sv["max_delay"]]}
    servo.update(model=servo_name, ids=servo_id)
    rate_limit = float(jp.get("rate_limit") or 0.0)
    contract = {
        "source": {"checkpoint": str(CK), "task_id": s1["task_id"], "arm": s1["arm"], "iter": ck.get("iter")},
        "joint_names": names,
        "leg_names": list(kin.LEGS),
        "q_default": q_default,
        "clip_lo": lo, "clip_hi": hi,
        "urdf_lo": urdf_lo, "urdf_hi": urdf_hi,
        "action_scale": jp["scale"],
        "rate_limit": rate_limit,
        "dt": dt,
        "window": s1["window"],
        "frame_dim": 111,
        "obs_dim": actor_dim,
        "layout": {k: [v.start, v.stop] for k, v in lay.items()},
        "o_p": {k: [v.start, v.stop] for k, v in layout.ACTOR.items() if v.stop <= 111},
        "v0_keep": 0.0, "clock_on": clock_on, "e_gate": 1.0,
        "tick": tick,
        "clock": {"offsets": [0.0, math.pi] * 3, "stride_m": 0.20, "f_min": 0.8, "f_max": 2.5},
        "command": {
            "lin_vel_x": list(cmd["ranges"]["lin_vel_x"]), "lin_vel_y": list(cmd["ranges"]["lin_vel_y"]),
            "ang_vel_z": list(cmd["ranges"]["ang_vel_z"]), "turn_yaw_range": list(cmd["turn_yaw_range"]),
            "rel_standing_envs": cmd["rel_standing_envs"], "rel_turn_envs": cmd["rel_turn_envs"],
            "rel_side_envs": cmd["rel_side_envs"],
        },
        "kinematics": {"hip": kin.HIP.tolist(), "hip_yaw": kin.HIP_YAW.tolist(), "femur_off": kin.FEMUR_OFF.tolist(),
                       "tibia_off": kin.TIBIA_OFF.tolist(), "toe_off": kin.TOE_OFF.tolist()},
        "servo": servo,
        "sleep_positions": [sleep[n] for n in names],
        "proprio": {"drop_p0": env.get("drop_p0"), "drop_k": env.get("drop_k")},
    }
    CONTRACT.parent.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    json.dump(contract, open(CONTRACT, "w"), indent=1)
    print("wrote %s  (%s, obs %d, W %d, dt %.3f, servo %s tick %.6f, clock %s, slew %.2f rad/s)"
          % (CONTRACT.relative_to(REPO), s1["task_id"], actor_dim, s1["window"], dt, servo_name, tick,
             "on" if clock_on else "off", rate_limit))

    subprocess.run([sys.executable, str(REPO / "tools/s1_export/export_fused.py"), str(CK), str(MODELS)], check=True)
    parity(ck, contract)


def parity(ck, c):
    """A sequence through hexasim's own modules, with HexapodS1Env's per-step logic copied 1:1 (refresh_proprio,
    _whist fill, s1_actor_obs assembly, the S0b clock, JointPositionAction / SlewJointPositionAction processing).
    The clock starts at the tripod offsets (phase jitter 0) and the slew starts from the rest pose."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    torch.manual_seed(1)
    T, W = 120, c["window"]
    q0 = torch.tensor(c["q_default"])
    lo, hi = torch.tensor(c["clip_lo"]), torch.tensor(c["clip_hi"])
    t = torch.arange(T, dtype=torch.float32)[:, None] * 0.02
    q_true = q0 + 0.25 * torch.sin(2 * math.pi * (0.7 + 0.05 * torch.arange(18)) * t + torch.arange(18))
    valid = torch.rand(T, 18) >= 0.08
    valid[:3] = True
    imu9 = torch.randn(T, 9) * 0.3 + torch.tensor([0, 0, 9.81, 0, 0, 0, 0, 0, -1.0])
    cmd = torch.zeros(T, 3)
    cmd[40:80] = torch.tensor([0.10, 0.0, 0.0])
    cmd[80:] = torch.tensor([0.15, 0.025, -0.2])

    aux = S1Aux(AuxCfg(), 1, q0, c["dt"], "cpu")
    missing = aux.load_state_dict(ck["aux_state_dict"], strict=False)
    assert not [k for k in missing.missing_keys if not k.startswith(("cur_", "td_"))], missing
    aux.eval()
    actor = MLPModel(TensorDict({"policy": torch.zeros(1, c["obs_dim"])}, [1]), {"actor": ["policy"]}, "actor", 18,
                     [256, 128, 64], "elu", True, {"class_name": "GaussianDistribution", "init_std": 1.0})
    actor.load_state_dict(ck["actor_state_dict"])
    actor.eval()

    pr = ProprioReader(1, 18, c["dt"], "cpu", tick=c["tick"])
    whist = torch.zeros(1, W, 111)
    a_prev, qcmd_prev = torch.zeros(18), q0.clone()  # episode start: raw action 0, target = rest pose
    ck_ = c["clock"]
    phase = torch.tensor(ck_["offsets"])
    max_step = c["rate_limit"] * c["dt"]
    zero_tail = torch.zeros(1, c["obs_dim"] - layout.ACTOR["m_star"].start)
    out = {k: [] for k in ("q", "qd", "mask", "o_p", "obs", "action", "q_cmd")}
    for k in range(T):
        if k == 0:
            pr.reset(slice(None), q_true[0:1])
        pr.update(q_true[k:k + 1], valid[k:k + 1])
        # HexapodS0bEnv.step: cpg_active and the phase advance come from the command held when the step began.
        ref = cmd[max(k - 1, 0)]
        active = bool((ref != 0).any())
        if k > 0 and active:
            f = min(max(float(ref[0]) / ck_["stride_m"], ck_["f_min"]), ck_["f_max"])
            phase = (phase + 2.0 * math.pi * f * c["dt"]) % (2.0 * math.pi)
        clock = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).view(12) * float(active) * c["clock_on"]
        o_p = torch.cat([pr.q[0] - q0, pr.qd[0], qcmd_prev - pr.q[0], a_prev, imu9[k], clock, pr.mask[0]])
        whist = torch.roll(whist, -1, dims=1)
        whist[:, -1] = o_p
        if k == 0:
            whist[:] = o_p
        with torch.no_grad():
            lat = aux.latents(whist)
            obs = torch.cat([o_p[None], torch.zeros(1, 3), cmd[k:k + 1], lat, zero_tail], dim=-1)
            a = actor(TensorDict({"policy": obs}, [1]))[0]
        q_cmd = torch.clamp(q0 + c["action_scale"] * a, lo, hi)
        if max_step > 0:
            q_cmd = qcmd_prev + torch.clamp(q_cmd - qcmd_prev, -max_step, max_step)
        for key, v in (("q", pr.q[0]), ("qd", pr.qd[0]), ("mask", pr.mask[0]), ("o_p", o_p), ("obs", obs[0]),
                       ("action", a), ("q_cmd", q_cmd)):
            out[key].append(v.numpy().copy())
        a_prev, qcmd_prev = a, q_cmd
    np.savez(MODELS / "parity_ref.npz", q_true=q_true.numpy(), valid=valid.numpy(), imu9=imu9.numpy(),
             cmd=cmd.numpy(), **{k: np.array(v) for k, v in out.items()})
    print("wrote %s (%d steps, %.1f %% dropped reads)"
          % ((MODELS / "parity_ref.npz").relative_to(REPO), T, 100 * (1 - valid.float().mean())))


if __name__ == "__main__":
    main()

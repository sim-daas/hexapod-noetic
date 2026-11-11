"""Workstation (env_isaaclab): write config/contract_s1h4.json + models/ for the robot runner, from hexasim code.

    ~/miniconda3/envs/env_isaaclab/bin/python tools/policy/export_contract.py [checkpoint]

Outputs (all generated, never hand-edited):
  config/contract_s1h4.json   joint order, q_default, clip, scale, dt, W, obs layout, command set, FK constants,
                              servo IDs + sleep pose (by name, from the vendor pxmark4.yaml)
  models/control.onnx         fused control graph (tools/s1_export/export_fused.py), hist (1,W,111) + obs_base → a
  models/fused_io.npz         random I/O sample of the fused graph
  models/parity_ref.npz       a 120-step sequence run through hexasim's own ProprioReader, S1Aux and actor, with the
                              env's history fill / obs assembly: the robot-side NumPy code must reproduce it
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
VENDOR_YAML = Path("/home/ubuntu/gitrepos/interbotix_ros_crawlers/interbotix_ros_xshexapods/"
                   "interbotix_xshexapod_control/config/pxmark4.yaml")

from hexasim.actuators.ax12a import TICK  # noqa: E402
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
    assert s1["v0_drop_p"] == 1.0 and s1["clock_drop_p"] == 1.0, "runner feeds V_0 and the clock as zeros"

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
    ax = env["scene"]["robot"]["actuators"]["ax12a"]
    contract = {
        "source": {"checkpoint": str(CK), "task_id": s1["task_id"], "arm": s1["arm"], "iter": ck.get("iter")},
        "joint_names": names,
        "leg_names": list(kin.LEGS),
        "q_default": q_default,
        "clip_lo": lo, "clip_hi": hi,
        "urdf_lo": urdf_lo, "urdf_hi": urdf_hi,
        "action_scale": jp["scale"],
        "dt": dt,
        "window": s1["window"],
        "frame_dim": 111,
        "obs_dim": actor_dim,
        "layout": {k: [v.start, v.stop] for k, v in lay.items()},
        "o_p": {k: [v.start, v.stop] for k, v in layout.ACTOR.items() if v.stop <= 111},
        "v0_keep": 0.0, "clock_on": 0.0, "e_gate": 1.0,
        "tick": TICK,
        "clock": {"offsets": [0.0, math.pi] * 3, "stride_m": 0.20, "f_min": 0.8, "f_max": 2.5},
        "command": {
            "lin_vel_x": list(cmd["ranges"]["lin_vel_x"]), "lin_vel_y": list(cmd["ranges"]["lin_vel_y"]),
            "ang_vel_z": list(cmd["ranges"]["ang_vel_z"]), "turn_yaw_range": list(cmd["turn_yaw_range"]),
            "rel_standing_envs": cmd["rel_standing_envs"], "rel_turn_envs": cmd["rel_turn_envs"],
            "rel_side_envs": cmd["rel_side_envs"],
        },
        "kinematics": {"hip": kin.HIP.tolist(), "hip_yaw": kin.HIP_YAW.tolist(), "femur_off": kin.FEMUR_OFF.tolist(),
                       "tibia_off": kin.TIBIA_OFF.tolist(), "toe_off": kin.TOE_OFF.tolist()},
        # Nominal servo settings (DR centre; the run randomised slope over {16, 32, 64}).
        "servo": {"compliance_margin": ax["compliance_margin"], "compliance_slope": ax["compliance_slope"],
                  "slope_choices": list(ax["compliance_slope_choices"] or []),
                  "punch": int(round(ax["punch"] * 1023)), "ids": servo_id},
        "sleep_positions": [sleep[n] for n in names],
        "proprio": {"drop_p0": env.get("drop_p0"), "drop_k": env.get("drop_k")},
    }
    (REPO / "config").mkdir(exist_ok=True)
    (REPO / "models").mkdir(exist_ok=True)
    json.dump(contract, open(REPO / "config" / "contract_s1h4.json", "w"), indent=1)
    print("wrote config/contract_s1h4.json  (obs %d, W %d, dt %.3f)" % (actor_dim, s1["window"], dt))

    subprocess.run([sys.executable, str(REPO / "tools/s1_export/export_fused.py"), str(CK), str(REPO / "models")],
                   check=True)
    parity(ck, contract)


def parity(ck, c):
    """A sequence through hexasim's own modules, with HexapodS1Env's per-step logic copied 1:1 (refresh_proprio,
    _whist fill, s1_actor_obs assembly, JointPositionAction processing)."""
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

    pr = ProprioReader(1, 18, c["dt"], "cpu")
    whist = torch.zeros(1, W, 111)
    a_prev, qcmd_prev = torch.zeros(18), q0.clone()  # episode start: raw action 0, target = rest pose
    zero_tail = torch.zeros(1, c["obs_dim"] - layout.ACTOR["m_star"].start)
    out = {k: [] for k in ("q", "qd", "mask", "o_p", "obs", "action", "q_cmd")}
    for k in range(T):
        if k == 0:
            pr.reset(slice(None), q_true[0:1])
        pr.update(q_true[k:k + 1], valid[k:k + 1])
        o_p = torch.cat([pr.q[0] - q0, pr.qd[0], qcmd_prev - pr.q[0], a_prev, imu9[k], torch.zeros(12), pr.mask[0]])
        whist = torch.roll(whist, -1, dims=1)
        whist[:, -1] = o_p
        if k == 0:
            whist[:] = o_p
        with torch.no_grad():
            lat = aux.latents(whist)
            obs = torch.cat([o_p[None], torch.zeros(1, 3), cmd[k:k + 1], lat, zero_tail], dim=-1)
            a = actor(TensorDict({"policy": obs}, [1]))[0]
        q_cmd = torch.clamp(q0 + c["action_scale"] * a, lo, hi)
        for key, v in (("q", pr.q[0]), ("qd", pr.qd[0]), ("mask", pr.mask[0]), ("o_p", o_p), ("obs", obs[0]),
                       ("action", a), ("q_cmd", q_cmd)):
            out[key].append(v.numpy().copy())
        a_prev, qcmd_prev = a, q_cmd
    np.savez(REPO / "models" / "parity_ref.npz", q_true=q_true.numpy(), valid=valid.numpy(), imu9=imu9.numpy(),
             cmd=cmd.numpy(), **{k: np.array(v) for k, v in out.items()})
    print("wrote models/parity_ref.npz (%d steps, %.1f %% dropped reads)" % (T, 100 * (1 - valid.float().mean())))


if __name__ == "__main__":
    main()

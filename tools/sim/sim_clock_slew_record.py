"""Run with hexasim env_isaaclab from the hexasim repo: python <this> --headless --ckpt <pt> --out x.npz
Records what the real env feeds a blind S1-family policy, for tools/sim/check_clock_slew.py: per step the twist
command and clock columns of the observation, the env's clock phase, the raw action and the sent joint target.
Flat ground, the task's own command distribution, resampled every 2-3 s so stand <-> walk <-> turn changes occur."""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=32)
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hexasim.rl.load_policy import build_actor, configure, eval_physx_buffers, load_env_state, task_for  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import layout, pin_terrain_level  # noqa: E402
from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from tensordict import TensorDict  # noqa: E402

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
task = task_for(ck)
cfg = parse_env_cfg(task, num_envs=args.n)
configure(cfg, ck)
eval_physx_buffers(cfg)
cfg.curriculum = None
cfg.seed = args.seed
cfg.commands.base_velocity.resampling_time_range = (2.0, 3.0)
cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
    num_rows=1, num_cols=2, difficulty_range=(0.0, 0.0), sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)})
cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.0)
env = gym.make(task, cfg=cfg)
base = env.unwrapped
pin_terrain_level(base, 0)
ckd = torch.load(args.ckpt, map_location=base.device, weights_only=False)
load_env_state(base, ckd)
with torch.inference_mode():
    obs, _ = env.reset()
policy = build_actor(ckd, base.device)
robot = base.scene["robot"]
term_a = base.action_manager.get_term("joint_pos")
A = layout.ACTOR
steps = int(args.seconds / base.step_dt)
rec = {k: [] for k in ("cmd", "clock", "phase", "a", "q_sent", "done")}
with torch.inference_mode():
    for t in range(steps):
        o = obs["policy"]
        rec["cmd"].append(o[:, A["twist_cmd"]].cpu().numpy())  # what the policy sees at step t
        rec["clock"].append(o[:, A["clock"]].cpu().numpy())
        rec["phase"].append(base.cpg_phase.cpu().numpy())
        a = policy(TensorDict({"policy": o}, batch_size=[base.num_envs]))
        obs, _, term, trunc, _ = env.step(a)
        rec["a"].append(a.cpu().numpy())
        rec["q_sent"].append(term_a.processed_actions.cpu().numpy())  # target sent for action a
        rec["done"].append((term | trunc).cpu().numpy())
jp = cfg.actions.joint_pos
np.savez_compressed(args.out, task=task, dt=base.step_dt, joint_names=np.array(robot.joint_names),
                    q_default=robot.data.default_joint_pos.torch[0].cpu().numpy(),
                    rate_limit=float(getattr(jp, "rate_limit", 0.0) or 0.0), scale=jp.scale,
                    **{k: np.stack(v) for k, v in rec.items()})
print("[record] saved", args.out, steps, "steps;", task, "envs done at some point:", int(np.stack(rec["done"]).any(0).sum()))
env.close()
app.close()

"""Run with hexasim env_isaaclab from the hexasim repo: python <this> --headless --ckpt <pt> --out x.npz [--vx 0.05] [--zero_offsets].
Policy onset in sim: S1H4 checkpoint, zero command, flat, N envs; logs actions/joints for the first seconds.
--zero_offsets: reset joints exactly at q_default (the robot's STAND hand-off) instead of +-0.25 rad."""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=64)
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--zero_offsets", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--vx", type=float, default=0.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hexasim.rl.load_policy import build_actor, configure, eval_physx_buffers, load_env_state, task_for  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from tensordict import TensorDict  # noqa: E402

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
base_task = task_for(ck)
play = base_task.replace("-v0", "-Play-v0")
task = play if play in gym.registry else base_task
cfg = parse_env_cfg(task, num_envs=args.n)
configure(cfg, ck)
eval_physx_buffers(cfg)
cfg.curriculum = None
cfg.seed = args.seed
pin_command(cfg, args.vx, 0.0, 0.0)
cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
    num_rows=1, num_cols=2, difficulty_range=(0.0, 0.0), sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)})
cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.0)
if hasattr(cfg, "v0_ramp_steps"):
    cfg.v0_ramp_steps = (0, 1)
if args.zero_offsets:
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
print("[onset] task", task, "reset joint offsets", cfg.events.reset_robot_joints.params["position_range"])
env = gym.make(task, cfg=cfg)
base = env.unwrapped
pin_terrain_level(base, 0)
ckd = torch.load(args.ckpt, map_location=base.device, weights_only=False)
load_env_state(base, ckd)
with torch.inference_mode():
    obs, _ = env.reset()
policy = build_actor(ckd, base.device)
robot = base.scene["robot"]
steps = int(args.seconds / base.step_dt)
rec = {k: [] for k in ("a", "q_true", "q_obs", "q_cmd", "gyro", "o_p", "term", "vel")}
with torch.inference_mode():
    for t in range(steps):
        a = policy(TensorDict({"policy": obs["policy"]}, batch_size=[base.num_envs]))
        rec["o_p"].append(obs["policy"][:, :111].cpu().numpy())
        obs, _, term, trunc, _ = env.step(a)
        rec["a"].append(a.cpu().numpy())
        rec["q_true"].append(robot.data.joint_pos.torch.cpu().numpy())
        rec["q_obs"].append(base.proprio.q.cpu().numpy())
        rec["q_cmd"].append(robot.data.joint_pos_target.torch.cpu().numpy())
        rec["gyro"].append(robot.data.root_ang_vel_b.torch.cpu().numpy())
        rec["term"].append((term | trunc).cpu().numpy())
        rec["vel"].append(robot.data.root_lin_vel_b.torch.cpu().numpy())
np.savez_compressed(args.out, joint_names=np.array(robot.joint_names), dt=base.step_dt,
                    q_default=robot.data.default_joint_pos.torch[0].cpu().numpy(),
                    **{k: np.stack(v) for k, v in rec.items()})
print("[onset] saved", args.out, steps, "steps; terminated envs:", int(np.stack(rec["term"]).any(0).sum()))
env.close()
app.close()

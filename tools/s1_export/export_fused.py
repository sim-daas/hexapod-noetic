"""Fused control graph for a blind S1-family policy (S1 1,125-D, S2 / S1H4 1,269-D): hist (1, W, 111) + obs_base
(1, obs_dim) → action (18). obs_dim is read from the checkpoint's actor.

Exact rewrites of hexasim's inference (verified here against the original modules):
  - TCN evaluated only on its receptive cone of the last time step (same zero padding → identical output).
  - v_kin inside the graph; the 3-parameter plane lstsq becomes the closed-form normal-equation solve.
  - obs_base carries every non-latent block; z_p goes to [117:181], ê = [v_kin + Δv̂, f̂] (gate open) to [181:196].
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from tensordict import TensorDict

S1 = "/home/ubuntu/gitrepos/hexasim/source/hexasim/hexasim/rl/s1/"
CK, OUT = sys.argv[1], Path(sys.argv[2])


def load(name):
    spec = importlib.util.spec_from_file_location(name, S1 + name + ".py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tcn, kin = load("tcn"), load("kinematics")
ck = torch.load(CK, map_location="cpu", weights_only=False)
assert ck["s1_state"]["gate_iter"] >= 0, "ê gate closed in this checkpoint"
W = ck["s1_state"]["window"]
asd = ck["aux_state_dict"]
Z0, E0, E1 = 117, 181, 196  # z_p and ê slots, the same in every S1-family layout (mdp/layout.py; asserted by the caller)
D = ck["actor_state_dict"]["mlp.0.weight"].shape[1]  # actor input width


class ConeTCN(nn.Module):
    """ProprioTCN.forward restricted to the positions the last output depends on."""

    def __init__(self, full: "tcn.ProprioTCN"):
        super().__init__()
        self.full = full
        k = full.blocks[0].conv1.conv.kernel_size[0]
        # Top-down: positions needed at each conv output, then gather tables bottom-up.
        convs = []  # (block_idx, which, dilation)
        for i, d in enumerate(full.dilations):
            convs += [(i, 1, d), (i, 2, d)]
        need = [None] * (len(convs) + 1)  # need[j] = positions required at the input of conv j; need[-1] = output
        need[len(convs)] = [W - 1]
        for j in range(len(convs) - 1, -1, -1):
            d = convs[j][2]
            out = set(need[j + 1])
            if convs[j][1] == 1:  # conv1's output feeds conv2 only; the block input also feeds the skip
                out_c1 = set(need[j + 1])
                ins = {p - t * d for p in out_c1 for t in range(k)} | set(need[j + 2])
            else:
                ins = {p - t * d for p in out for t in range(k)}
            need[j] = sorted(p for p in ins if p >= 0)
        # conv j computes outputs at need[j+1] (for conv1: the positions conv2 needs), reading need[j]
        self.pos = need
        self.idx = []
        for j, (_, _, d) in enumerate(convs):
            src = {p: n for n, p in enumerate(need[j])}
            zero = len(need[j])  # appended zero column = causal padding
            self.idx.append(torch.tensor([[src.get(p - (k - 1 - t) * d, zero) for p in need[j + 1]] for t in range(k)]))
        self.convs, self.k = convs, k
        self.n_full = len(convs) * W
        self.n_cone = sum(len(need[j + 1]) for j in range(len(convs)))

    def conv_at(self, x, conv, idx):
        """x (C, n_in) → (C_out, n_out) at the gathered positions."""
        xp = torch.cat((x, torch.zeros(x.shape[0], 1)), dim=1)
        w = conv.weight  # (C_out, C_in, k)
        y = conv.bias[:, None]
        for t in range(self.k):
            y = y + w[:, :, t] @ xp[:, idx[t]]
        return y

    def forward(self, hist):  # hist (1, W, in) normalised
        x = hist[0].transpose(0, 1)  # (in, W)
        x = x[:, torch.tensor(self.pos[0])] if self.pos[0] != list(range(W)) else x
        j = 0
        for blk in self.full.blocks:
            p_in, p_mid, p_out = self.pos[j], self.pos[j + 1], self.pos[j + 2]
            h = F.elu(self.conv_at(x, blk.conv1.conv, self.idx[j]))
            h = self.conv_at(h, blk.conv2.conv, self.idx[j + 1])
            sel = torch.tensor([p_in.index(p) for p in p_out])
            xs = x[:, sel]
            skip = blk.skip(xs[None])[0] if isinstance(blk.skip, nn.Conv1d) else xs
            x = F.elu(h + skip)
            j += 2
        return self.full.norm(self.full.head(x[:, -1]))[None]


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


def vkin_graph(q, qd, err, gyro, tau_e=0.005, s=0.003, delta=-0.005, eps=1e-3):
    """kinematics.v_kin with lstsq → closed-form 3×3 normal equations (single sample, (1, ·) tensors)."""
    qs = torch.cat((q, q + eps * qd, q - eps * qd), dim=0)  # one FK call for the point and both FD samples
    P = kin.toe_positions(qs)  # (3, 6, 3)
    p, jqd = P[0:1], (P[1:2] - P[2:3]) / (2 * eps)
    g = gyro[:, None, :].expand_as(p)
    cross = torch.stack((g[..., 1] * p[..., 2] - g[..., 2] * p[..., 1],
                         g[..., 2] * p[..., 0] - g[..., 0] * p[..., 2],
                         g[..., 0] * p[..., 1] - g[..., 1] * p[..., 0]), dim=-1)
    v_foot = -(jqd + cross)
    e = err[:, 6:12].abs() + err[:, 12:18].abs()
    A = torch.cat((p[..., :2], torch.ones_like(p[..., :1])), dim=-1)  # (1, 6, 3)
    M = A.transpose(1, 2) @ A  # (1, 3, 3)
    b = A.transpose(1, 2) @ p[..., 2:3]
    a, bb, c, d, ee, f, g2, h, i = [M[:, r, cc] for r in range(3) for cc in range(3)]
    det = a * (ee * i - f * h) - bb * (d * i - f * g2) + c * (d * h - ee * g2)
    adj = torch.stack((torch.stack((ee * i - f * h, c * h - bb * i, bb * f - c * ee), -1),
                       torch.stack((f * g2 - d * i, a * i - c * g2, c * d - a * f), -1),
                       torch.stack((d * h - ee * g2, bb * g2 - a * h, a * ee - bb * d), -1)), -2)
    coef = (adj / det[:, None, None]) @ b
    z_plane = (A @ coef)[..., 0]
    w = torch.sigmoid((e - tau_e) / s) * torch.sigmoid((z_plane - p[..., 2] + delta) / 0.005)
    return (w[..., None] * v_foot).sum(-2) / w.sum(-1, keepdim=True).clamp(min=1e-3)


class Fused(nn.Module):
    def __init__(self, cone, dv, fhat, actor, frame_mean, frame_std, q_default):
        super().__init__()
        self.cone, self.dv, self.fhat, self.actor = cone, dv, fhat, actor
        self.register_buffer("frame_mean", frame_mean)
        self.register_buffer("frame_std", frame_std)
        self.register_buffer("q_default", q_default)

    def forward(self, hist, obs_base):
        z = self.cone((hist - self.frame_mean) / self.frame_std)
        f = hist[:, -1]
        vk = vkin_graph(f[:, 0:18] + self.q_default, f[:, 18:36], f[:, 36:54], f[:, 75:78])
        e = torch.cat((vk + self.dv(z), self.fhat(z)), dim=-1)
        obs = torch.cat((obs_base[:, :Z0], z, e, obs_base[:, E1:]), dim=-1)
        return self.actor(obs)


full = tcn.ProprioTCN(111, W)
full.load_state_dict({k[8:]: v for k, v in asd.items() if k.startswith("encoder.")})
dv, fhat = mlp(64, 3), mlp(64, 12)
dv.load_state_dict({k[3:]: v for k, v in asd.items() if k.startswith("dv.")})
fhat.load_state_dict({k[5:]: v for k, v in asd.items() if k.startswith("fhat.")})
cone = ConeTCN(full)
sd = ck["actor_state_dict"]
actor = MLPModel(TensorDict({"policy": torch.zeros(1, D)}, [1]), {"actor": ["policy"]}, "actor", 18, [256, 128, 64],
                 "elu", True, {"class_name": "GaussianDistribution", "init_std": 1.0})
actor.load_state_dict(sd)
fused = Fused(cone, dv, fhat, actor.as_onnx(verbose=False), asd["frame_mean"], asd["frame_std"], asd["q_default"]).eval()
for m in (full, dv, fhat, actor):
    m.eval()

# ---- verify against the original modules
torch.manual_seed(0)
worst = {"z": 0.0, "vk": 0.0, "action": 0.0}
for _ in range(64):
    hist = torch.randn(1, W, 111) * asd["frame_std"] + asd["frame_mean"]
    obs_base = torch.randn(1, D)
    with torch.no_grad():
        hn = (hist - asd["frame_mean"]) / asd["frame_std"]
        z_ref = full(hn)
        f = hist[:, -1]
        vk_ref, _ = kin.v_kin(f[:, 0:18] + asd["q_default"], f[:, 18:36], f[:, 36:54], f[:, 75:78], 0.005, 0.003, -0.005)
        e_ref = torch.cat((vk_ref + dv(z_ref), fhat(z_ref)), dim=-1)
        obs = obs_base.clone()
        obs[:, Z0:E0], obs[:, E0:E1] = z_ref, e_ref
        a_ref = actor(TensorDict({"policy": obs}, [1]))
        worst["z"] = max(worst["z"], (cone(hn) - z_ref).abs().max().item())
        worst["vk"] = max(worst["vk"], (vkin_graph(f[:, 0:18] + asd["q_default"], f[:, 18:36], f[:, 36:54],
                                                   f[:, 75:78]) - vk_ref).abs().max().item())
        worst["action"] = max(worst["action"], (fused(hist, obs_base) - a_ref).abs().max().item())
print("max |Δ| vs original modules (64 random inputs):", worst)
print(f"TCN conv position-evaluations: cone {cone.n_cone} vs full {cone.n_full}")
assert max(worst.values()) < 1e-4

hist = torch.randn(1, W, 111) * asd["frame_std"] + asd["frame_mean"]
obs_base = torch.randn(1, D)
torch.onnx.export(fused, (hist, obs_base), OUT / "control.onnx", input_names=["hist", "obs_base"],
                  output_names=["action"], opset_version=17, dynamo=False)
with torch.no_grad():
    np.savez(OUT / "fused_io.npz", hist=hist.numpy(), obs_base=obs_base.numpy(), action=fused(hist, obs_base).numpy())
print("exported", OUT / "control.onnx")

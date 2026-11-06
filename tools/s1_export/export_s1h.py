"""Export the blind S1H4 control path (E_p TCN + V heads, actor with folded normaliser) to ONNX for a Pi latency test.

v_kin (lstsq) is left out of the graph: it runs in NumPy on the robot. Also writes random sample I/O for a parity check.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

CK = sys.argv[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
TCN_PY = "/home/ubuntu/gitrepos/hexasim/source/hexasim/hexasim/rl/s1/tcn.py"

spec = importlib.util.spec_from_file_location("tcn", TCN_PY)
tcn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcn)

ck = torch.load(CK, map_location="cpu", weights_only=False)
W = ck["s1_state"]["window"]
aux_sd = ck["aux_state_dict"]


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class AuxInfer(nn.Module):
    """hist (1, W, 111) raw frames → z_p (64), Δv̂ (3), f̂ (12). Same maths as S1Aux.forward minus v_kin."""

    def __init__(self):
        super().__init__()
        self.encoder = tcn.ProprioTCN(111, W)
        self.dv, self.fhat = mlp(64, 3), mlp(64, 12)
        self.register_buffer("frame_mean", torch.zeros(111))
        self.register_buffer("frame_std", torch.ones(111))

    def forward(self, hist):
        z = self.encoder((hist - self.frame_mean) / self.frame_std)
        return z, self.dv(z), self.fhat(z)


aux = AuxInfer()
keep = {k: v for k, v in aux_sd.items() if k.split(".")[0] in ("encoder", "dv", "fhat", "frame_mean", "frame_std")}
missing, unexpected = aux.load_state_dict(keep, strict=True), None
aux.eval()

sd = ck["actor_state_dict"]
obs_dim = sd["obs_normalizer._mean"].shape[1]
actor = MLPModel(
    TensorDict({"policy": torch.zeros(1, obs_dim)}, [1]),
    {"actor": ["policy"]},
    "actor",
    18,
    [256, 128, 64],
    "elu",
    True,
    {"class_name": "GaussianDistribution", "init_std": 1.0},
)
actor.load_state_dict(sd)
actor.eval()
actor_onnx = actor.as_onnx(verbose=False).eval()

hist = torch.randn(1, W, 111) * aux.frame_std + aux.frame_mean
obs = torch.randn(1, obs_dim)
with torch.no_grad():
    z, dv, fh = aux(hist)
    a = actor_onnx(obs)
    a_ref = actor(TensorDict({"policy": obs}, [1]))  # deterministic path
assert torch.allclose(a, a_ref, atol=1e-5), (a - a_ref).abs().max()

torch.onnx.export(aux, (hist,), OUT / "aux.onnx", input_names=["hist"], output_names=["z_p", "dv", "fhat"],
                  opset_version=17, dynamo=False)
torch.onnx.export(actor_onnx, (obs,), OUT / "actor.onnx", input_names=["obs"], output_names=["action"],
                  opset_version=17, dynamo=False)
np.savez(OUT / "sample_io.npz", hist=hist.numpy(), obs=obs.numpy(), z=z.numpy(), dv=dv.numpy(), fhat=fh.numpy(),
         action=a.numpy())
print(f"W={W} obs_dim={obs_dim} TCN dilations={aux.encoder.dilations} RF={aux.encoder.rf}; exported to {OUT}")

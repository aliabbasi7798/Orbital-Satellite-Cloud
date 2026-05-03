
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelSpec:
    model_id: str
    params_billion: float


def build_model_table(cfg: dict) -> Dict[str, ModelSpec]:
    """Return {model_id: ModelSpec} from config."""
    table = {}
    for m in cfg["model"]["models"]:
        table[m["id"]] = ModelSpec(model_id=m["id"],
                                   params_billion=float(m["params_billion"]))
    return table


def active_model_specs(cfg: dict) -> List[ModelSpec]:
    """Return list of ModelSpec for ids in traffic.active_models."""
    table = build_model_table(cfg)
    active_ids = cfg["traffic"]["active_models"]
    out = []
    for mid in active_ids:
        if mid not in table:
            raise ValueError(f"active model '{mid}' not in model.models")
        out.append(table[mid])
    if not out:
        raise ValueError("traffic.active_models must be non-empty")
    return out


@dataclass
class TraceSampler:
    rng: np.random.Generator
    bytes_per_token: int
    flops_per_param_per_token: float
    active_models: List[ModelSpec]

    def sample_batch(self, n: int):
        """Return arrays:
          in_toks, out_toks, flops, in_bytes, out_bytes, model_idx
        model_idx indexes into self.active_models.
        """
        raise NotImplementedError


class SyntheticBurstGPTSampler(TraceSampler):
    """BurstGPT-style mixture for token lengths; uniform model assignment."""

    def sample_batch(self, n: int):
        rng = self.rng
        which_in = rng.random(n) < 0.80
        in_short = rng.lognormal(mean=4.3, sigma=0.9, size=n)
        in_long = rng.lognormal(mean=6.2, sigma=1.0, size=n)
        in_toks = np.where(which_in, in_short, in_long)
        in_toks = np.clip(np.round(in_toks), 1, 8192).astype(np.int32)

        which_out = rng.random(n) < 0.90
        out_short = rng.lognormal(mean=3.4, sigma=1.1, size=n)
        out_long = rng.lognormal(mean=5.8, sigma=0.9, size=n)
        out_toks = np.where(which_out, out_short, out_long)
        out_toks = np.clip(np.round(out_toks), 1, 4096).astype(np.int32)

        M = len(self.active_models)
        model_idx = rng.integers(0, M, size=n).astype(np.int32)
        params_B = np.array([m.params_billion for m in self.active_models])
        req_params = params_B[model_idx]

        total_toks = in_toks.astype(np.int64) + out_toks.astype(np.int64)
        flops = (self.flops_per_param_per_token
                 * req_params * 1e9
                 * total_toks).astype(np.float64)

        in_bytes = in_toks.astype(np.int64) * self.bytes_per_token
        out_bytes = out_toks.astype(np.int64) * self.bytes_per_token

        return in_toks, out_toks, flops, in_bytes, out_bytes, model_idx


class BurstGPTCSVSampler(TraceSampler):
    def __init__(self, csv_path: str, **kw):
        super().__init__(**kw)
        df = pd.read_csv(csv_path)
        self.in_pool = df["input_tokens"].to_numpy(dtype=np.int32)
        self.out_pool = df["output_tokens"].to_numpy(dtype=np.int32)

    def sample_batch(self, n: int):
        idx = self.rng.integers(0, len(self.in_pool), size=n)
        in_toks = self.in_pool[idx]
        out_toks = self.out_pool[idx]
        M = len(self.active_models)
        model_idx = self.rng.integers(0, M, size=n).astype(np.int32)
        params_B = np.array([m.params_billion for m in self.active_models])
        req_params = params_B[model_idx]
        total_toks = in_toks.astype(np.int64) + out_toks.astype(np.int64)
        flops = (self.flops_per_param_per_token
                 * req_params * 1e9
                 * total_toks).astype(np.float64)
        in_bytes = in_toks.astype(np.int64) * self.bytes_per_token
        out_bytes = out_toks.astype(np.int64) * self.bytes_per_token
        return in_toks, out_toks, flops, in_bytes, out_bytes, model_idx


def make_sampler(cfg: dict, rng: np.random.Generator) -> TraceSampler:
    mcfg = cfg["model"]
    tcfg = cfg["traffic"]
    src = tcfg["trace_source"]
    active = active_model_specs(cfg)
    kw = dict(
        rng=rng,
        bytes_per_token=mcfg["bytes_per_token"],
        flops_per_param_per_token=mcfg["flops_per_param_per_token"],
        active_models=active,
    )
    if src == "synthetic_burstgpt":
        return SyntheticBurstGPTSampler(**kw)
    elif src == "burstgpt_csv":
        path = tcfg.get("trace_csv", "data/burstgpt.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"BurstGPT CSV not found: {path}")
        return BurstGPTCSVSampler(csv_path=path, **kw)
    else:
        raise ValueError(f"Unknown trace source: {src}")


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(cfg["traffic"]["trace_seed"])
    s = make_sampler(cfg, rng)
    ins, outs, flops, ib, ob, midx = s.sample_batch(100_000)
    print(f"Active models: {[m.model_id for m in s.active_models]}")
    print(f"Input tokens:  med={np.median(ins):.0f}  p95={np.percentile(ins,95):.0f}")
    print(f"Output tokens: med={np.median(outs):.0f}  p95={np.percentile(outs,95):.0f}")
    for i, m in enumerate(s.active_models):
        mask = midx == i
        frac = mask.mean()
        med_flops = np.median(flops[mask]) if mask.any() else 0
        print(f"  {m.model_id:12s} ({m.params_billion:.1f}B) frac={frac:.3f}  "
              f"med_flops={med_flops:.2e}")

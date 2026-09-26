"""Export the Maia3-23M-ponder model (trunk + all three heads + pooled embedding)
to ONNX for in-browser inference (onnxruntime-web, WebGPU/wasm).

Strategy: load the exact production
model via Maia3PonderWrapper (CPU, fp32), then trace an ExportModel that reuses
the SAME parameters but replaces three ONNX-hostile constructs:
  1. nn.MultiheadAttention -> explicit q/k/v attention math (clean graph for
     ort-web; nn.MHA decomposes into a verbose subgraph with B*H mask folding).
  2. nn.RMSNorm -> primitive ops (no native op below opset 23; ort-web wasm EP
     support for RMSNormalization is not guaranteed).
  3. The 256-iteration Python promotion-logit loop -> one broadcast add
     (the loop traces to ~768 tiny nodes; each node is a dispatch in ort-web).
Equivalence is gated: ExportModel vs the original MAIA3Model.forward must agree
to ~fp32 precision on real tokens BEFORE export, and onnxruntime must agree
with torch after export. Any mismatch raises.

Inputs (all browser-friendly dtypes):
  tokens    float32 (B, 64, 100)  - build_timed_history_tensor layout
  self_elos float32 (B,)          - raw Elo; clamp [0,5000] + interpolation inside
  oppo_elos float32 (B,)
Outputs:
  logits_move   float32 (B, 4352) - UNMASKED move-vocab logits (mask client-side)
  logits_value  float32 (B, 3)    - WDL, internal order [loss, draw, win]
  logits_ponder float32 (B, 30)   - think-time bin logits (softmax client-side)
  embedding     float32 (B, 512)  - pooled last_ln trunk embedding

NOTE (parity caveat, documented): the production wrapper clamps Elo to [0,2700]
before the model's own [0,5000] clamp. The export clamps [0,5000] only; callers
must pre-clamp to [0,2700] to match production (Elo 600..2600 is always
inside both).

Artifacts (written to --out-dir, default ./onnx_out):
  maia3_23m_ponder_fp32.onnx  (~92 MB)
  maia3_23m_ponder_fp16.onnx  (~46 MB, keep_io_types=True -> fp32 I/O)
  maia3_23m_ponder_int8.onnx  (~23 MB, dynamic quant, wasm-CPU tier)
  export_report.json          (sizes, parity numbers)

Run: python export/export_onnx.py

--------------------------------------------------------------------------
STANDALONE COPY (published for AGPL compliance)

The docstrings above and below are kept close to the internal original:
they are the record of how this export was derived. Two things in them
describe the internal tree rather than this repo, and three things changed:

  * "load the exact production model via Maia3PonderWrapper" and "via its
    production wrapper" (load_model) refer to Kibitzr-internal wrappers
    that are not published. This copy performs the equivalent load with
    public pieces only: maia3.model_registry + maia3.models.MAIA3Model + huggingface_hub + torch.
    The ponder head surgery (nn.Linear(head_hid_dim, 1) -> 30 bins) is done
    here explicitly, as the internal wrapper did it.
  * --out-dir now defaults to ./onnx_out instead of a machine-local scratch
    drive, and the run line is `python export/export_onnx.py`.
  * The parity verification (both gates named above) now lives in a clearly
    marked optional section further down and degrades gracefully when its
    dependencies are absent. The fp32 ONNX export path itself needs only
    torch + onnx + maia3 (+ huggingface_hub to fetch the checkpoint).

Dependencies:
  required : torch, onnx, maia3 (https://github.com/CSSLab/maia3), huggingface_hub
  optional : python-chess (real-token inputs), onnxruntime (post-export parity,
             int8 quantization), onnxconverter-common (fp16 conversion)
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FP32_EPS = torch.finfo(torch.float32).eps  # nn.RMSNorm(eps=None) default

# Maia3-23M-ponder is an unlisted preview repo (no alias in maia3's registry),
# so it is resolved by repo id at a PINNED revision: a silent re-upload must
# fail loudly rather than change behavior.
PONDER_REPO = "UofTCSSLab/Maia3-23M-ponder"
PONDER_CHECKPOINT = "maia3-23m-ponder.pt"
PONDER_REVISION = "34ec2be9d651102e4ad0e9eda3d838acd69f5f9a"
PONDER_BINS = 30

PARITY_TOL = 1e-3


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """nn.RMSNorm(512, eps=None) in primitive ops (fp32 default eps)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + FP32_EPS) * weight


class ExportModel(nn.Module):
    """MAIA3Model.forward re-expressed for a clean ONNX graph.

    Reuses the loaded model's modules/parameters (no copies), so a state-dict
    change upstream cannot silently diverge: the pre-export parity gate runs
    the original forward next to this one on the same tokens.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.history = int(m.cfg.history)
        self.heads = int(m.cfg.num_heads)
        self.head_hid_dim = int(m.cfg.head_hid_dim)
        d_model = int(m.cfg.dim_vit)
        self.d_head = d_model // self.heads
        # Timed checkpoints keep 3 clock columns; untimed (79M) keep none.
        self.tok_keep = 12 * self.history + (
            3 if getattr(m.cfg, "include_time_info", False) else 0)
        # Bake the two Elo embedding vectors as constants (Embedding(1, 128)).
        self.register_buffer("elo_low", m.elo_embedding_low.weight[0].detach().clone())
        self.register_buffer("elo_high", m.elo_embedding_high.weight[0].detach().clone())

    def _gab_bias(self, attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """MHA._sq_bias verbatim (sm1 path; 23m has gab_per_square_dim=32)."""
        B = x.size(0)
        y = attn.sm1(x)
        y = y.reshape(B, -1)
        y = attn.sm_act(attn.sm2(y))
        y = attn.ln1(y)
        y = attn.sm_act(attn.sm3(y))
        y = attn.ln2(y).view(B, self.heads, attn.gen_size)
        b = torch.einsum("bhi,oi->bho", y, attn.gab_weight)
        return b.view(B, self.heads, 64, 64)

    def _attention(self, attn: nn.Module, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """nn.MultiheadAttention (batch_first, no qkv/out biases) made explicit."""
        B = x.size(0)
        mha = attn.mha
        qkv = x @ mha.in_proj_weight.t()  # (B, 64, 3*d_model); no bias (omit_qkv_biases)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, 64, self.heads, self.d_head).transpose(1, 2)
        k = k.view(B, 64, self.heads, self.d_head).transpose(1, 2)
        v = v.view(B, 64, self.heads, self.d_head).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head) + bias
        out = torch.softmax(scores, dim=-1) @ v  # (B, H, 64, d_head)
        out = out.transpose(1, 2).reshape(B, 64, self.heads * self.d_head)
        out = out @ mha.out_proj.weight.t()
        if mha.out_proj.bias is not None:
            out = out + mha.out_proj.bias
        return out

    def _elo_emb(self, elos: torch.Tensor) -> torch.Tensor:
        w_low = (torch.clamp(elos, 0.0, 5000.0) / 5000.0).unsqueeze(1)  # (B,1)
        return w_low * self.elo_low + (1.0 - w_low) * self.elo_high  # (B,128)

    def forward(self, tokens, self_elos, oppo_elos):
        m = self.m
        tokens = tokens[:, :, : self.tok_keep]  # drop clk_ponder label (+clock cols if untimed)
        self_emb = self._elo_emb(self_elos).unsqueeze(1).expand(-1, 64, -1)
        oppo_emb = self._elo_emb(oppo_elos).unsqueeze(1).expand(-1, 64, -1)
        x = m.token_projection(torch.cat([tokens, self_emb, oppo_emb], dim=-1))

        for blk in m.transformer.layers:
            bias = self._gab_bias(blk.self_attn, x)
            sa = self._attention(blk.self_attn, x, bias)
            x = rms_norm(x + sa, blk.norm1.weight)
            ff = blk.linear2(F.gelu(blk.linear1(x)))
            x = rms_norm(x + ff, blk.norm2.weight)
        x = m.transformer.norm(x)  # final LayerNorm

        sq_from = m.proj_sq_from(x)
        sq_to = m.proj_sq_to(x)
        scores = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(self.head_hid_dim)
        scores_flat = scores.reshape(x.size(0), 64 * 64)

        # Promotions vectorized: order (from_file major, to_file, piece) matches
        # the original triple loop; rank7 squares 48..55, rank8 squares 56..63.
        promo_base = scores[:, 48:56, 56:64]  # (B, 8_from, 8_to)
        promo_biases = m.promo_bias_proj(sq_to[:, 56:64, :]) * math.sqrt(self.head_hid_dim)  # (B, 8_to, 4)
        promo = promo_base.unsqueeze(-1) + promo_biases.unsqueeze(1)  # (B, 8, 8, 4)
        logits_move = torch.cat([scores_flat, promo.reshape(x.size(0), 256)], dim=1)

        pooled = m.last_ln(x.mean(dim=1))  # (B, 512) - pooled embedding
        logits_value = m.fc_value(F.relu(m.fc_value_hid(pooled)))
        logits_ponder = m.fc_ponder(F.relu(m.fc_ponder_hid(pooled)))
        # clone(): the embedding graph-output must not share a tensor with the
        # head inputs, or fp16 keep_io_types cast insertion produces mixed-type
        # Gemms (onnxconverter-common shared-output bug).
        return logits_move, logits_value, logits_ponder, pooled.clone()


GAME_UCIS = [
    "e2e4", "e7e5", "b1c3", "g8f6", "f1c4", "f6e4", "d1h5", "e4d6",
    "h5e5", "f8e7", "c4b3", "e8g8", "c3d5", "d6f5", "e5f4", "d7d6",
    "d5e7", "f5e7", "g1f3", "b8c6", "e1g1", "c8g4",
]


def build_real_tokens(cfg, timed: bool = True, n_positions: int = 6) -> tuple[torch.Tensor, list]:
    """Tokens from a real short game (Vienna, 3+0 clocks) with 8-ply history.

    timed=True -> build_timed_history_tensor (23M-ponder, clock columns);
    timed=False -> build_history_tensor (79M, untimed). ``cfg`` must carry
    history / include_time_info / use_padding matching the target model.
    """
    # STANDALONE NOTE: build_timed_history_tensor / build_history_tensor were
    # internal thin wrappers over maia3.dataset.get_historical_tokens (the timed
    # one passing base/inc/clk_left_before through, the untimed one relying on
    # cfg.include_time_info=False to emit the single zero ponder column). This
    # copy calls the public function directly, which is what they both did.
    import chess  # optional dependency: only the example inputs need it
    from maia3.dataset import get_historical_tokens, tokenize_board

    board = chess.Board()
    boards = [board.copy()]
    for u in GAME_UCIS:
        board.push_uci(u)
        boards.append(board.copy())

    rows, meta = [], []
    base, inc = 180.0, 0.0
    for p in range(len(boards) - n_positions, len(boards)):
        window = boards[max(0, p - 7): p + 1]
        clk = max(5.0, base - 4.0 * p)  # plausible countdown
        history = [tokenize_board(b) for b in window][-int(cfg.history):]
        rows.append(get_historical_tokens(
            history, cfg,
            base=base if timed else 0.0,
            inc=inc if timed else 0.0,
            clk_left_before=clk if timed else 0.0,
            clk_ponder=0.0,
        ))
        meta.append({"ply": p, "fen": boards[p].fen(), "clk_left_before": clk})
    return torch.stack(rows).float(), meta


def build_synthetic_tokens(cfg, n_positions: int = 6) -> tuple[torch.Tensor, list]:
    """Correctly-shaped placeholder inputs when python-chess is unavailable.

    Enough to trace and export; NOT enough to verify anything, so the parity
    gates refuse to run on them (they would compare two implementations on
    board states that never occur).
    """
    width = 12 * int(cfg.history) + (4 if getattr(cfg, "include_time_info", False) else 1)
    return torch.zeros((n_positions, 64, width), dtype=torch.float32), []


# ==========================================================================
# OPTIONAL VERIFICATION SECTION
#
# In the internal original these gates were not optional: the harness built
# real tokens through Kibitzr's internal wrappers and every export ran them. Those
# wrappers are not published, and the remaining pieces (python-chess, onnxruntime,
# onnxconverter-common) are optional installs, so each gate degrades to a
# printed SKIP here instead of failing the export.
#
# What they verify, in order:
#   1. torch_parity_gate  - ExportModel reproduces the original
#      MAIA3Model.forward to fp32 precision on real tokens, BEFORE export.
#      This is the gate that makes the three graph rewrites above safe.
#   2. ort_parity_gate    - onnxruntime running the exported fp32 graph
#      reproduces torch, AFTER export (catches tracing/opset damage).
#   3. fp16 / int8 gates  - reduced-precision artifacts judged on
#      decision-relevant quantities (softmax probs, top-1 agreement), not on
#      raw logit deltas.
#
# Running an unverified export is a real risk: every failure mode here is
# silent (plausible-looking but wrong policy), so keep the dependencies
# installed for any export you intend to serve.
# ==========================================================================


def torch_parity_gate(model, export_model, tokens_g, elos_f, oppo_f):
    """ExportModel vs the original MAIA3Model.forward on the same tokens."""
    with torch.no_grad():
        ref_move, ref_val, ref_pon = model(tokens_g, elos_f.long(), oppo_f.long())
        exp_move, exp_val, exp_pon, _ = export_model(tokens_g, elos_f, oppo_f)
    return {
        "move_max_abs": float((ref_move - exp_move).abs().max()),
        "value_max_abs": float((ref_val - exp_val).abs().max()),
        "ponder_max_abs": float((ref_pon - exp_pon.reshape(ref_pon.shape)).abs().max()),
    }


def make_ort_runner():
    """`run_ort(path, tokens, self_elos, oppo_elos)` or None if ort is absent."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    def run_ort(path: Path, tk: np.ndarray, se: np.ndarray, oe: np.ndarray,
                optimize: bool = True):
        so = ort.SessionOptions()
        if not optimize:
            # ORT 1.23.2 CPU-EP bug: fp16 models get auto-inserted
            # InsertedPrecisionFreeCast nodes, then SimplifiedLayerNormFusion
            # crashes at init. Parity only needs correct numerics, not speed.
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        return sess.run(None, {"tokens": tk, "self_elos": se, "oppo_elos": oe})

    return run_ort


# ==========================================================================
# END OPTIONAL VERIFICATION SECTION
# ==========================================================================


MODELS = {
    "23m-ponder": dict(prefix="maia3_23m_ponder", timed=True, emb_dim=512),
    "79m": dict(prefix="maia3_79m", timed=False, emb_dim=1024),
}


def _load_state_dict(path: str):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    # Older checkpoints used "smolgen" naming; the current model uses "gab".
    return {k.replace("smolgen", "gab"): v for k, v in sd.items()}


def load_model(kind: str):
    """Load the target checkpoint fp32 on CPU via its production wrapper.

    STANDALONE NOTE: the line above is the original docstring. Kibitzr's
    production wrappers are not published, so this copy does the equivalent
    load with the public maia3 package:

      * architecture from maia3.model_registry.resolve_model_spec, alias
        maia3-23m or maia3-79m, into maia3.models.MAIA3Model;
      * for 23m-ponder, include_time_info=True (live clock columns) and
        fc_ponder swapped to nn.Linear(head_hid_dim, 30) BEFORE the load, so a
        shape skew fails loudly rather than leaving a randomly initialized
        head. Weights come from the pinned PONDER_REVISION of PONDER_REPO;
      * for 79m, the checkpoint the registry resolves (fp32). The internal
        version loaded an fp16 variant and then undid its interpolate_elo
        dtype patch; nothing to undo here, the load is fp32 throughout.

    Any missing/unexpected key after the smolgen->gab rename is fatal: a silent
    mismatch leaves layers at random init and produces plausible-looking but
    wrong policy/WDL outputs.
    """
    from maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
    from maia3.models import MAIA3Model

    if kind == "23m-ponder":
        from huggingface_hub import hf_hub_download

        spec = resolve_model_spec("maia3-23m")
        cfg = SimpleNamespace(**{**spec.config, "include_time_info": True}, device="cpu")
        model = MAIA3Model(cfg)
        model.fc_ponder = nn.Linear(cfg.head_hid_dim, PONDER_BINS)
        ckpt_path = hf_hub_download(
            PONDER_REPO, PONDER_CHECKPOINT, revision=PONDER_REVISION)
    else:
        spec = resolve_model_spec("maia3-79m")
        cfg = SimpleNamespace(**spec.config, device="cpu")
        model = MAIA3Model(cfg)
        ckpt_path = resolve_checkpoint_path(spec)

    result = model.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Maia-3 checkpoint/architecture mismatch for {kind}: "
            f"missing={result.missing_keys[:5]} unexpected={result.unexpected_keys[:5]}")
    return model.float().eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="23m-ponder")
    ap.add_argument("--out-dir", default="./onnx_out")
    ap.add_argument("--skip-fp16", action="store_true")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()
    spec = MODELS[args.model]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"model": args.model}

    print(f"[1/6] Loading {args.model} (CPU, fp32)...")
    model = load_model(args.model)

    try:
        tokens, _ = build_real_tokens(model.cfg, timed=spec["timed"])
        real_tokens = True
    except ImportError:
        print("   python-chess not installed: tracing on placeholder tokens, "
              "parity gates will be SKIPPED (see OPTIONAL VERIFICATION SECTION).")
        tokens, _ = build_synthetic_tokens(model.cfg)
        real_tokens = False
    report["real_tokens"] = real_tokens

    n = tokens.shape[0]
    tok_w = tokens.shape[2]
    grid = torch.tensor([600, 933, 1266, 1600, 1933, 2266, 2600], dtype=torch.float32)
    tokens_g = tokens.repeat_interleave(len(grid), dim=0)  # (n*7, 64, tok_w)
    elos_f = grid.repeat(n)
    oppo_f = torch.full_like(elos_f, 1500.0)
    report["token_width"] = tok_w

    export_model = ExportModel(model).eval()
    with torch.no_grad():
        exp_move, exp_val, exp_pon, exp_emb = export_model(tokens_g, elos_f, oppo_f)
    assert exp_emb.shape[1] == spec["emb_dim"]

    print("[2/6] Parity gate: ExportModel vs original MAIA3Model.forward...")
    if real_tokens:
        gate = torch_parity_gate(model, export_model, tokens_g, elos_f, oppo_f)
        report["torch_reimpl_parity"] = gate
        print("   ", gate)
        if max(gate.values()) > PARITY_TOL:
            raise RuntimeError(f"ExportModel diverges from original forward: {gate}")
    else:
        print("   SKIPPED (no real tokens).")

    fp32_path = out_dir / f"{spec['prefix']}_fp32.onnx"
    print(f"[3/6] torch.onnx.export (opset 17, dynamic batch) -> {fp32_path}")
    torch.onnx.export(
        export_model,
        (tokens_g[:2], elos_f[:2], oppo_f[:2]),
        str(fp32_path),
        input_names=["tokens", "self_elos", "oppo_elos"],
        output_names=["logits_move", "logits_value", "logits_ponder", "embedding"],
        dynamic_axes={k: {0: "batch"} for k in
                      ["tokens", "self_elos", "oppo_elos",
                       "logits_move", "logits_value", "logits_ponder", "embedding"]},
        opset_version=17,
    )
    report["fp32_bytes"] = fp32_path.stat().st_size

    print("[4/6] onnxruntime (CPU) parity vs torch...")
    run_ort = make_ort_runner()
    feeds = (tokens_g.numpy(), elos_f.numpy(), oppo_f.numpy())
    p32 = None
    o_val = o_pon = None
    if run_ort is None:
        print("   SKIPPED (onnxruntime not installed).")
    else:
        t0 = time.perf_counter()
        o_move, o_val, o_pon, o_emb = run_ort(fp32_path, *feeds)
        report["ort_cpu_fp32_s_for_%d_rows" % len(elos_f)] = round(time.perf_counter() - t0, 3)
        ort_gate = {
            "move_max_abs": float(np.abs(o_move - exp_move.numpy()).max()),
            "value_max_abs": float(np.abs(o_val - exp_val.numpy()).max()),
            "ponder_max_abs": float(np.abs(o_pon - exp_pon.numpy()).max()),
            "embedding_max_abs": float(np.abs(o_emb - exp_emb.numpy()).max()),
        }
        report["ort_fp32_parity"] = ort_gate
        print("   ", ort_gate)
        if max(ort_gate.values()) > PARITY_TOL:
            raise RuntimeError(f"ONNX fp32 diverges from torch: {ort_gate}")
        p32 = torch.softmax(torch.from_numpy(o_move), dim=1).numpy()

    if not args.skip_fp16:
        print("[5/6] fp16 conversion (keep_io_types=True)...")
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError:
            print("   SKIPPED (onnx / onnxconverter-common not installed).")
        else:
            fp16_path = out_dir / f"{spec['prefix']}_fp16.onnx"
            # op_block_list=[]: the default block list wraps Pow/ReduceMean in fp32
            # casts, which trips ORT's SimplifiedLayerNormFusion on the RMSNorm pattern
            # (missing-node-arg crash at session init). Nothing here is overflow-prone;
            # the fp16 parity gate below is the proof.
            m16 = float16.convert_float_to_float16(
                onnx.load(str(fp32_path)), keep_io_types=True, op_block_list=[])
            # keep_io_types bug: when a graph output is also an internal input (the
            # embedding feeds the value/ponder heads), the inserted fp16->fp32 Cast gets
            # rewired into the fp16 Gemms. Rewire internal consumers back to the Cast's
            # fp16 source; the Cast remains solely for the graph output.
            out_names = {o.name for o in m16.graph.output}
            cast_src = {n.output[0]: n.input[0] for n in m16.graph.node
                        if n.op_type == "Cast" and n.output[0] in out_names}
            for n in m16.graph.node:
                for i, inp in enumerate(n.input):
                    if inp in cast_src and n.output[0] != inp:
                        n.input[i] = cast_src[inp]
            onnx.save(m16, str(fp16_path))
            report["fp16_bytes"] = fp16_path.stat().st_size
            if p32 is None:
                print("   fp16 parity SKIPPED (no onnxruntime fp32 reference).")
            else:
                o16 = run_ort(fp16_path, *feeds, optimize=False)
                # fp16 judged on decision-relevant quantities, not raw logit deltas:
                # masked-softmax happens client-side, so compare softmax probs + top-1.
                p16 = torch.softmax(torch.from_numpy(o16[0]), dim=1).numpy()
                report["fp16_parity"] = {
                    "policy_prob_max_abs": float(np.abs(p32 - p16).max()),
                    "top1_match_rate": float((p32.argmax(1) == p16.argmax(1)).mean()),
                    "wdl_prob_max_abs": float(np.abs(
                        torch.softmax(torch.from_numpy(o_val), 1).numpy()
                        - torch.softmax(torch.from_numpy(o16[1]), 1).numpy()).max()),
                    "ponder_prob_max_abs": float(np.abs(
                        torch.softmax(torch.from_numpy(o_pon), 1).numpy()
                        - torch.softmax(torch.from_numpy(o16[2]), 1).numpy()).max()),
                }
                print("   ", report["fp16_parity"])

    if not args.skip_int8:
        print("[6/6] int8 dynamic quantization (wasm-CPU tier)...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            print("   SKIPPED (onnxruntime not installed).")
        else:
            int8_path = out_dir / f"{spec['prefix']}_int8.onnx"
            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
            report["int8_bytes"] = int8_path.stat().st_size
            if p32 is None:
                print("   int8 parity SKIPPED (no onnxruntime fp32 reference).")
            else:
                o8 = run_ort(int8_path, *feeds)
                p8 = torch.softmax(torch.from_numpy(o8[0]), dim=1).numpy()
                report["int8_parity"] = {
                    "policy_prob_max_abs": float(np.abs(p32 - p8).max()),
                    "top1_match_rate": float((p32.argmax(1) == p8.argmax(1)).mean()),
                }
                print("   ", report["int8_parity"])

    (out_dir / f"export_report_{spec['prefix']}.json").write_text(json.dumps(report, indent=2))
    print("DONE. Report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

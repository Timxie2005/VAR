import argparse
import time
import os
from statistics import mean, median, stdev
from typing import List

import torch

import dist
from models import build_vae_var
import models.var as var_mamba_mod
import models.var_transformer as var_tf_mod


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark VAR inference: mamba vs transformer")
    p.add_argument("--depth", type=int, default=16, choices=[16, 20, 24, 30])
    p.add_argument("--vae_ckpt", type=str, default="vae_ch160v4096z32.pth")
    p.add_argument("--var_ckpt", type=str, default=None,
                   help="Optional VAR checkpoint to load (not required for speed test)")
    p.add_argument("--B", type=int, nargs="+", default=[1, 2, 4], help="Batch sizes to test (space separated)")
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations per test")
    p.add_argument("--iters", type=int, default=5, help="Measured iterations per test")
    p.add_argument("--use_amp", action="store_true", help="Enable autocast (FP16) on CUDA")
    p.add_argument("--device", type=str, default=None, help="Device string (e.g., cuda:0). If not set, use dist.get_device()")
    return p.parse_args()


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_models(device, depth):
    # Use same VQVAE config as elsewhere in repo
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var_mamba = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=depth, shared_aln=False,
    )

    # instantiate transformer VAR with same vae
    # var_transformer.VAR signature matches VAR(vae_local=..., ...)
    var_tf = var_tf_mod.VAR(
        vae_local=vae,
        num_classes=1000, depth=depth, embed_dim=depth * 64, num_heads=depth,
    )

    return vae, var_mamba, var_tf


def time_inference(model, device, B: int, iters: int, warmup: int, use_amp: bool, seed: int = 0):
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # prepare label_B as None so model randomly samples labels internally (consistent across runs if seed provided)
    times: List[float] = []

    # warmup
    for _ in range(warmup):
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=True):
                out = model.autoregressive_infer_cfg(B=B, label_B=None, g_seed=seed, cfg=1.5, top_k=0, top_p=0.0)
        else:
            out = model.autoregressive_infer_cfg(B=B, label_B=None, g_seed=seed, cfg=1.5, top_k=0, top_p=0.0)
        sync_if_cuda(device)

    # measured runs
    for i in range(iters):
        t0 = time.perf_counter()
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=True):
                out = model.autoregressive_infer_cfg(B=B, label_B=None, g_seed=seed + i, cfg=1.5, top_k=0, top_p=0.0)
        else:
            out = model.autoregressive_infer_cfg(B=B, label_B=None, g_seed=seed + i, cfg=1.5, top_k=0, top_p=0.0)
        sync_if_cuda(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def summarize_times(times: List[float], B: int):
    tot = [t for t in times]
    per_image = [t / B for t in tot]
    return {
        "mean_total": mean(tot),
        "median_total": median(tot),
        "std_total": stdev(tot) if len(tot) > 1 else 0.0,
        "mean_per_image": mean(per_image),
        "images_per_sec": 1.0 / mean(per_image) if mean(per_image) > 0 else float("inf"),
    }


def main():
    args = parse_args()

    # init device
    dist.initialize(fork=False)
    device = torch.device(args.device) if args.device else dist.get_device()
    print(f"Using device: {device}")

    vae, var_mamba, var_tf = build_models(device, args.depth)

    # load VAE weights if available (not strictly required for timing, but closer to real run)
    if os.path.exists(args.vae_ckpt):
        print(f"Loading VAE weights from {args.vae_ckpt} (map to cpu then to device)")
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=True)

    # move models to device
    var_mamba = var_mamba.to(device).eval()
    var_tf = var_tf.to(device).eval()

    # disable grad
    for m in (var_mamba, var_tf, vae):
        for p in m.parameters():
            p.requires_grad_(False)

    # Ensure CUDA is ready
    sync_if_cuda(device)

    results = {}

    for B in args.B:
        print(f"\nBenchmarking B={B}  (warmup={args.warmup}, iters={args.iters}, use_amp={args.use_amp})")

        t_mamba = time_inference(var_mamba, device, B=B, iters=args.iters, warmup=args.warmup, use_amp=args.use_amp)
        summary_m = summarize_times(t_mamba, B)
        print(f"Mamba: mean_total={summary_m['mean_total']:.3f}s mean_per_image={summary_m['mean_per_image']:.3f}s images/s={summary_m['images_per_sec']:.2f}")

        t_tf = time_inference(var_tf, device, B=B, iters=args.iters, warmup=args.warmup, use_amp=args.use_amp)
        summary_t = summarize_times(t_tf, B)
        print(f"Transformer: mean_total={summary_t['mean_total']:.3f}s mean_per_image={summary_t['mean_per_image']:.3f}s images/s={summary_t['images_per_sec']:.2f}")

        results[B] = {"mamba": summary_m, "transformer": summary_t}

    print('\nFull results:')
    for B, r in results.items():
        print(f'B={B}:')
        for k in ("mamba", "transformer"):
            s = r[k]
            print(f"  {k}: mean_total={s['mean_total']:.3f}s mean_per_image={s['mean_per_image']:.3f}s images/s={s['images_per_sec']:.2f}")


if __name__ == "__main__":
    main()

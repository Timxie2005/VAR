import argparse
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image

import dist
from models import build_vae_var


def parse_args():
    p = argparse.ArgumentParser(description="VAR demo: generate sample images and save them locally")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to VAR training checkpoint (defaults to local_output/ar-ckpt-last-*.pth or the latest ar-ckpt*.pth in the directory)")
    p.add_argument("--vae_ckpt", type=str, default="vae_ch160v4096z32.pth",
                   help="VAE checkpoint path (default: prepackaged file in repo root)")
    p.add_argument("--outdir", type=str, default="samples_mamba",
                   help="Output directory for generated images")
    p.add_argument("--B", type=int, default=4, help="Number of images to generate in one run")
    p.add_argument("--cls", type=int, default=-1,
                   help="Class label (0..1000). Use -1 for random/unconditional mixing (useful for CFG)")
    p.add_argument("--cfg", type=float, default=1.5, help="Classifier-Free Guidance strength")
    p.add_argument("--top_k", type=int, default=0, help="top-k sampling (0 disables)")
    p.add_argument("--top_p", type=float, default=0.0, help="top-p sampling (0 disables)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--patch_nums", type=str, default="1_2_3_4_5_6_8_10_13_16",
                   help="Multi-scale token map configuration (should match training). Example: 1_2_3_4_5_6_8_10_13_16")
    return p.parse_args()


def _find_default_ckpt() -> str:
    # Prefer ar-ckpt-last-*.pth first, then fall back to the most recent ar-ckpt*.pth
    cand = [
        os.path.join("local_output", n)
        for n in os.listdir("local_output") if n.startswith("ar-ckpt-last-") and n.endswith(".pth")
    ] if os.path.isdir("local_output") else []
    if cand:
        # 任取其一（通常每种模型类型只有一个 last）
        return sorted(cand)[-1]
    cand = [
        os.path.join("local_output", n)
        for n in os.listdir("local_output") if n.startswith("ar-ckpt") and n.endswith(".pth")
    ] if os.path.isdir("local_output") else []
    return sorted(cand, key=lambda p: os.path.getmtime(p))[-1] if cand else None


def _parse_pn(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.replace("-", "_").split("_"))


def tensor_to_pil(img_chw: torch.Tensor) -> Image.Image:
    # img_chw: 3xHxW, values in [0,1]
    img = img_chw.detach().clamp(0, 1).cpu().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # HWC
    return Image.fromarray(img)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Initialize device (single machine, single GPU assumed)
    dist.initialize(fork=False)
    device = dist.get_device()

    # Build VAE/VAR (architecture should match training)
    patch_nums = _parse_pn(args.patch_nums)
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    )

    # Load VAE weights
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_ckpt}. Please download it to the repo root or provide --vae_ckpt")
    vae_local.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)

    # Load VAR checkpoint (if available)
    ckpt_path = args.ckpt or _find_default_ckpt()
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[demo] Loading VAR checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("trainer", {})
        sd = state.get("var_wo_ddp", None)
        if sd is None:
            print("[demo][warn] 'trainer.var_wo_ddp' not found in checkpoint; the model will remain randomly initialized (outputs may be poor)")
        else:
            missing, unexpected = var_wo_ddp.load_state_dict(sd, strict=False)
            if missing:
                print(f"[demo][load_state_dict] missing: {missing}")
            if unexpected:
                print(f"[demo][load_state_dict] unexpected: {unexpected}")
    else:
        print("[demo][warn] No checkpoint found; using randomly initialized model (outputs may be poor). Provide --ckpt to specify a checkpoint path.")

    var_wo_ddp = var_wo_ddp.to(device).eval()

    # Sampling and saving
    B = int(args.B)
    label = None if args.cls < 0 else int(args.cls)
    img_B3HW = var_wo_ddp.autoregressive_infer_cfg(
        B=B, label_B=label, g_seed=args.seed, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p
    )  # [0,1]

    saved = []
    for i in range(B):
        pil = tensor_to_pil(img_B3HW[i])
        out = os.path.join(args.outdir, f"gen_{i:03d}.png")
        pil.save(out)
        saved.append(out)
        print(f"[demo] saved: {out}")

    # Finalize
    dist.finalize()
    print("[demo] Done.")


if __name__ == "__main__":
    main()

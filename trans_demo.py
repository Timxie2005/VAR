import argparse
import os
import os.path as osp
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image

import dist
from models import build_vae_var


def parse_args():
    p = argparse.ArgumentParser(description="VAR demo (script): generate images without torchvision dependency")
    p.add_argument("--vae_ckpt", type=str, default="vae_ch160v4096z32.pth",
                   help="VAE checkpoint path (default expects file in repo root)")
    p.add_argument("--var_ckpt", type=str, default=None,
                   help="VAR checkpoint path (optional). If not provided the script will try to find a suitable checkpoint in local_output.")
    p.add_argument("--outdir", type=str, default="samples_transformer",
                   help="output directory to save generated images")
    p.add_argument("--depth", type=int, default=16, choices=[16, 20, 24, 30],
                   help="model depth to match VAR checkpoints")
    p.add_argument("--B", type=int, default=8, help="number of images to generate")
    p.add_argument("--cls", type=str, default="980,980,437,437,22,22,562,562",
                   help="comma-separated class labels per sample; use -1 for unconditional (and will be converted to None). If fewer than B labels are provided, labels will be repeated.")
    p.add_argument("--cfg", type=float, default=4.0, help="classifier-free guidance strength")
    p.add_argument("--top_k", type=int, default=900, help="top-k sampling, 0 disables")
    p.add_argument("--top_p", type=float, default=0.95, help="top-p sampling, 0 disables")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--nrow", type=int, default=8, help="nrow for grid layout when saving a combined image")
    p.add_argument("--more_smooth", action="store_true", help="use more smooth option (as in notebook)")
    return p.parse_args()


def _find_default_var_ckpt() -> str:
    if not osp.isdir("local_output"):
        return None
    cand = [osp.join("local_output", n) for n in os.listdir("local_output") if n.startswith("ar-ckpt-last-") and n.endswith(".pth")]
    if cand:
        return sorted(cand)[-1]
    cand = [osp.join("local_output", n) for n in os.listdir("local_output") if n.startswith("ar-ckpt") and n.endswith(".pth")]
    if not cand:
        return None
    return sorted(cand, key=lambda p: osp.getmtime(p))[-1]


def _parse_class_labels(s: str, B: int) -> List[int]:
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    if not parts:
        return [None] * B
    labels = []
    for p in parts:
        try:
            v = int(p)
            labels.append(None if v < 0 else v)
        except Exception:
            # ignore unparsable tokens
            continue
    if not labels:
        return [None] * B
    # repeat to length B
    out = [labels[i % len(labels)] for i in range(B)]
    return out


def tensor_to_pil(img_chw: torch.Tensor) -> Image.Image:
    # img_chw: 3xHxW, values in [0,1]
    img = img_chw.detach().clamp(0, 1).cpu().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # HWC
    return Image.fromarray(img)


def make_grid_pil(images: List[Image.Image], nrow: int = 8, padding: int = 0, pad_value=255) -> Image.Image:
    # images: list of PIL Images (mode RGB)
    if not images:
        raise ValueError("no images to make grid")
    w, h = images[0].size
    nrow = max(1, int(nrow))
    ncol = (len(images) + nrow - 1) // nrow
    grid_w = nrow * w + padding * (nrow - 1)
    grid_h = ncol * h + padding * (ncol - 1)
    grid = Image.new("RGB", (grid_w, grid_h), color=(pad_value, pad_value, pad_value))
    for idx, img in enumerate(images):
        r = idx % nrow
        c = idx // nrow
        x = r * (w + padding)
        y = c * (h + padding)
        grid.paste(img, (x, y))
    return grid


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # init device
    dist.initialize(fork=False)
    device = dist.get_device()

    # build models
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        num_classes=1000, depth=args.depth, shared_aln=False,
    )

    # load VAE
    if not osp.exists(args.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_ckpt}. Please download or provide path via --vae_ckpt")
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # load VAR checkpoint if available
    var_ckpt = args.var_ckpt or _find_default_var_ckpt()
    if var_ckpt and osp.exists(var_ckpt):
        print(f"[demo] loading VAR checkpoint: {var_ckpt}")
        ckpt = torch.load(var_ckpt, map_location="cpu")
        # checkpoint format may vary; try common keys
        if isinstance(ckpt, dict) and "trainer" in ckpt:
            state = ckpt.get("trainer", {})
            sd = state.get("var_wo_ddp", None)
            if sd is None:
                print("[demo][warn] 'trainer.var_wo_ddp' not found in checkpoint; attempting to load top-level state dict")
                try:
                    var.load_state_dict(ckpt, strict=False)
                except Exception as e:
                    print(f"[demo][warn] failed to load checkpoint: {e}")
            else:
                missing, unexpected = var.load_state_dict(sd, strict=False)
                if missing:
                    print(f"[demo][load_state_dict] missing: {missing}")
                if unexpected:
                    print(f"[demo][load_state_dict] unexpected: {unexpected}")
        else:
            try:
                var.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f"[demo][warn] failed to load checkpoint: {e}")
    else:
        print("[demo][warn] no VAR checkpoint found; using random-initialized model (results may be poor). You can provide --var_ckpt or put checkpoint into local_output/")

    var = var.to(device).eval()
    torch.manual_seed(args.seed)

    # prepare labels
    labels = _parse_class_labels(args.cls, args.B)
    label_B = None
    if any(l is not None for l in labels):
        label_B = torch.tensor([(-1 if l is None else l) for l in labels], dtype=torch.long, device=device)
    else:
        label_B = None

    # run sampling (inference mode)
    with torch.inference_mode():
        # choose autocast if CUDA available
        if str(device).startswith("cuda"):
            # use amp for speed if available
            with torch.autocast(device_type="cuda", enabled=True):
                img_B3HW = var.autoregressive_infer_cfg(
                    B=args.B, label_B=label_B, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p, g_seed=args.seed, more_smooth=args.more_smooth
                )
        else:
            img_B3HW = var.autoregressive_infer_cfg(
                B=args.B, label_B=label_B, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p, g_seed=args.seed, more_smooth=args.more_smooth
            )

    # img_B3HW is expected to be [B, 3, H, W] with values in [0,1]
    pil_images = [tensor_to_pil(img_B3HW[i]) for i in range(img_B3HW.shape[0])]

    # save individual images
    saved = []
    for i, pil in enumerate(pil_images):
        out = osp.join(args.outdir, f"gen_{i:03d}.png")
        pil.save(out)
        saved.append(out)
        print(f"[demo] saved: {out}")

    # also save a grid if more than 1 image
    if len(pil_images) > 1:
        grid = make_grid_pil(pil_images, nrow=args.nrow, padding=0, pad_value=255)
        gpath = osp.join(args.outdir, "grid.png")
        grid.save(gpath)
        print(f"[demo] saved grid: {gpath}")

    dist.finalize()
    print("[demo] done.")


if __name__ == "__main__":
    main()

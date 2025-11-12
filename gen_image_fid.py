# gen_images.py
import os, torch
from PIL import Image
import numpy as np
import dist
from models import build_vae_var
import models.var_transformer as var_tf_mod

@torch.no_grad()
def load_models(depth=16, device=None):
    dist.initialize(fork=False)
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    patch_nums = (1,2,3,4,5,6,8,10,13,16)
    vae, var_mamba = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums, num_classes=1000, depth=depth, shared_aln=False
    )
    var_tf = var_tf_mod.VAR(vae_local=vae, num_classes=1000, depth=depth, embed_dim=depth*64, num_heads=depth, patch_nums=patch_nums)
    if hasattr(var_tf, 'init_weights'): var_tf.init_weights(init_std=-1)

    return device, vae.eval(), var_mamba.eval(), var_tf.eval()

@torch.no_grad()
def maybe_load_ckpt(model, path):
    if path and os.path.exists(path):
        sd = torch.load(path, map_location='cpu')
        # 兼容只含 "model"/"state_dict" 键或直接是权重字典的情况
        if isinstance(sd, dict) and any(k in sd for k in ['model','state_dict']): 
            sd = sd.get('model', sd.get('state_dict', sd))
        model.load_state_dict(sd, strict=False)

def tensor_to_pil_and_save(tensor, path):
    """Convert a single image tensor [C,H,W] in [0,1] to PIL and save as PNG."""
    # tensor: (3, H, W) float32 in [0, 1]
    arr = (tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (3, H, W)
    arr = arr.transpose(1, 2, 0)  # (H, W, 3)
    Image.fromarray(arr, mode='RGB').save(path, format='PNG')

def generate_and_save(model, out_dir, total=1000, B=16, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    n_saved, step = 0, 0
    while n_saved < total:
        cur_B = min(B, total - n_saved)
        imgs = model.autoregressive_infer_cfg(B=cur_B, label_B=None, g_seed=seed+step, cfg=1.5, top_k=0, top_p=0.0)
        # imgs: [0,1] float32, shape (B,3,H,W)
        for i in range(cur_B):
            tensor_to_pil_and_save(imgs[i], os.path.join(out_dir, f"{n_saved+i:06d}.png"))
        n_saved += cur_B
        step += 1
        if step % 10 == 0:
            print(f"Generated {n_saved}/{total} images...")

if __name__ == "__main__":
    device, vae, var_mamba, var_tf = load_models(depth=16)
    # 加载权重（按你的实际路径修改）
    maybe_load_ckpt(vae, "vae_ch160v4096z32.pth")
    maybe_load_ckpt(var_mamba, "local_output/ar-ckpt-best-mamba.pth")
    maybe_load_ckpt(var_tf, "local_output/ar-ckpt-best-transformer.pth")

    var_mamba.to(device); var_tf.to(device)
    for m in (vae, var_mamba, var_tf):
        for p in m.parameters(): p.requires_grad_(False)

    # 生成各 1000 张示例（正式评估建议更大数量）
    generate_and_save(var_mamba, "samples_mamba_fid", total=1000, B=16, seed=123)
    generate_and_save(var_tf,    "samples_transformer_fid", total=1000, B=16, seed=123)
import os
from pathlib import Path
from typing import Iterable
import torch_fidelity
from PIL import Image
from datetime import datetime

# ---------------- 基础路径 ----------------
real_dir = "datasets/imagenet-mini/val"              # 原始多子目录结构
real_resized_dir = "datasets/imagenet-mini/val_resized_299"  # 统一尺寸缓存目录
mamba_dir = "samples_mamba_fid"
transformer_dir = "samples_transformer_fid"

TARGET_SIZE = 299  # Inception 输入常用尺寸（内部会再做 299 预处理，这里确保统一）
VALID_EXT = {"jpg", "jpeg", "png", "JPEG", "JPG", "PNG"}


def iter_image_paths(root: str) -> Iterable[Path]:
    root_p = Path(root)
    for p in root_p.rglob("*"):
        if p.is_file() and p.suffix.lstrip('.').lower() in VALID_EXT:
            yield p


def build_resized_cache(src: str, dst: str, size: int = TARGET_SIZE):
    os.makedirs(dst, exist_ok=True)
    src_paths = list(iter_image_paths(src))
    # 若已存在并且文件数一致，直接跳过
    existing = list(iter_image_paths(dst))
    if len(existing) == len(src_paths) and len(src_paths) > 0:
        print(f"[cache] 发现已存在的统一尺寸目录 {dst} (共 {len(existing)} 张)，跳过重建。")
        return
    print(f"[cache] 重建统一尺寸目录 {dst}，共 {len(src_paths)} 张原始图像。")
    for i, p in enumerate(src_paths, 1):
        try:
            img = Image.open(p).convert('RGB')
            # 直接双线性缩放到 size x size
            img = img.resize((size, size), Image.BILINEAR)
            rel = p.relative_to(src)
            out_path = Path(dst) / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format='JPEG', quality=95)
        except Exception as e:
            print(f"[warn] 跳过 {p}，原因: {e}")
        if i % 500 == 0:
            print(f"  进度: {i}/{len(src_paths)}")
    print(f"[cache] 完成。共写入 {len(list(iter_image_paths(dst)))} 张。")


def main():
    # 1. 准备真实集统一尺寸缓存
    build_resized_cache(real_dir, real_resized_dir, TARGET_SIZE)

    # 2. 计算 Mamba FID
    print("计算 Mamba 模型的 FID...")
    metrics_mamba = torch_fidelity.calculate_metrics(
        input1=mamba_dir,
        input2=real_resized_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        verbose=True,
        samples_find_deep=True,
    )
    fid_mamba = metrics_mamba['frechet_inception_distance']
    print(f"\n✓ Mamba FID: {fid_mamba:.2f}\n")

    # 3. 计算 Transformer FID
    print("=" * 60)
    print("计算 Transformer 模型的 FID...")
    metrics_tf = torch_fidelity.calculate_metrics(
        input1=transformer_dir,
        input2=real_resized_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        verbose=True,
        samples_find_deep=True,
    )
    fid_tf = metrics_tf['frechet_inception_distance']
    print(f"\n✓ Transformer FID: {fid_tf:.2f}\n")

    # 4. 汇总
    print("=" * 60)
    print("结果汇总:")
    print(f"  Mamba FID:       {fid_mamba:.2f}")
    print(f"  Transformer FID: {fid_tf:.2f}")
    print(f"  差值 (Δ):        {fid_mamba - fid_tf:+.2f}")
    if fid_mamba < fid_tf:
        print("  → Mamba 生成质量更优 (FID 更低)")
    elif fid_tf < fid_mamba:
        print("  → Transformer 生成质量更优 (FID 更低)")
    else:
        print("  → 两者相当")
    print("=" * 60)

    # 写入结果到文件（追加），同时保存基本计数信息
    try:
        n_real = len(list(iter_image_paths(real_resized_dir)))
    except Exception:
        n_real = -1
    try:
        n_mamba = len(list(iter_image_paths(mamba_dir)))
    except Exception:
        n_mamba = -1
    try:
        n_tf = len(list(iter_image_paths(transformer_dir)))
    except Exception:
        n_tf = -1

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary = (
        f"Timestamp: {ts}\n"
        f"Real dir (resized): {real_resized_dir}  (count={n_real})\n"
        f"Mamba samples dir:   {mamba_dir}  (count={n_mamba})\n"
        f"Transformer samples:  {transformer_dir}  (count={n_tf})\n"
        f"TARGET_SIZE: {TARGET_SIZE}\n"
        f"Mamba FID: {fid_mamba:.6f}\n"
        f"Transformer FID: {fid_tf:.6f}\n"
        f"Delta (mamba - transformer): {fid_mamba - fid_tf:+.6f}\n"
    )

    out_file = 'fid_results.txt'
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write(summary)
        f.write('-' * 60 + '\n')

    print(f"已将结果追加写入 {out_file}")


if __name__ == "__main__":
    main()

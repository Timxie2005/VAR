import os
import os.path as osp
import random
from typing import Callable, List, Sequence, Tuple

import PIL.Image as PImage
import torch

# ---- Remove dependency on torchvision by providing minimal equivalents ----

# Common image extensions (aligned with torchvision.datasets.folder)
IMG_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"
)


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {inner}\n])"


class Resize:
    """
    Resize so that the shorter edge == size (int), keeping aspect ratio.
    Interpolation uses PIL.Image.LANCZOS by default.
    """

    def __init__(self, size: int, interpolation=PImage.LANCZOS):
        assert isinstance(size, int) and size > 0
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: PImage.Image) -> PImage.Image:
        w, h = img.size
        if w <= 0 or h <= 0:
            return img
        if h < w:
            new_h = self.size
            new_w = int(round(w * self.size / h))
        else:
            new_w = self.size
            new_h = int(round(h * self.size / w))
        return img.resize((new_w, new_h), self.interpolation)

    def __repr__(self):
        return f"Resize(size={self.size}, interpolation=LANCZOS)"


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: PImage.Image) -> PImage.Image:
        if random.random() < self.p:
            return img.transpose(PImage.FLIP_LEFT_RIGHT)
        return img

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class CenterCrop:
    def __init__(self, size: Tuple[int, int]):
        self.th, self.tw = int(size[0]), int(size[1])

    def __call__(self, img: PImage.Image) -> PImage.Image:
        w, h = img.size
        left = max((w - self.tw) // 2, 0)
        top = max((h - self.th) // 2, 0)
        right = min(left + self.tw, w)
        bottom = min(top + self.th, h)
        return img.crop((left, top, right, bottom))

    def __repr__(self):
        return f"CenterCrop(size=({self.th},{self.tw}))"


class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.th, self.tw = int(size[0]), int(size[1])

    def __call__(self, img: PImage.Image) -> PImage.Image:
        w, h = img.size
        if w == self.tw and h == self.th:
            return img
        if w < self.tw or h < self.th:
            # Pad by symmetric padding if smaller than target
            pad_w = max(self.tw - w, 0)
            pad_h = max(self.th - h, 0)
            img = img.crop((-pad_w // 2, -pad_h // 2, w + (pad_w - pad_w // 2), h + (pad_h - pad_h // 2)))
            w, h = img.size
        i = 0 if h == self.th else random.randint(0, h - self.th)
        j = 0 if w == self.tw else random.randint(0, w - self.tw)
        return img.crop((j, i, j + self.tw, i + self.th))

    def __repr__(self):
        return f"RandomCrop(size=({self.th},{self.tw}))"


class ToTensor:
    def __call__(self, img: PImage.Image) -> torch.Tensor:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x = torch.from_numpy(__import__('numpy').array(img, dtype='uint8'))  # HWC, uint8
        x = x.permute(2, 0, 1).contiguous().float() / 255.0  # CHW, float32 in [0,1]
        return x

    def __repr__(self):
        return "ToTensor()"


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations (torchvision-free)
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug_list = [
        Resize(mid_reso),  # resize shorter edge to mid_reso
        RandomCrop((final_reso, final_reso)),
        ToTensor(), normalize_01_into_pm1,
    ]
    if hflip:
        train_aug_list.insert(0, RandomHorizontalFlip())
    val_aug_list = [
        Resize(mid_reso),
        CenterCrop((final_reso, final_reso)),
        ToTensor(), normalize_01_into_pm1,
    ]
    train_aug, val_aug = Compose(train_aug_list), Compose(val_aug_list)

    # build dataset
    train_set = SimpleImageFolder(root=osp.join(data_path, 'train'), transform=train_aug, extensions=IMG_EXTENSIONS)
    val_set = SimpleImageFolder(root=osp.join(data_path, 'val'), transform=val_aug, extensions=IMG_EXTENSIONS)
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')

    return num_classes, train_set, val_set


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


class SimpleImageFolder(torch.utils.data.Dataset):
    """A simplified replacement for torchvision.datasets.DatasetFolder specialized for images.
    Expects directory structure root/class_x/xxx.ext
    """

    def __init__(self, root: str, transform: Callable = None, extensions: Tuple[str, ...] = IMG_EXTENSIONS):
        self.root = root
        self.transform = transform
        self.extensions = extensions

        classes = [d for d in sorted(os.listdir(root)) if osp.isdir(osp.join(root, d))]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples: List[Tuple[str, int]] = []
        for c in classes:
            cdir = osp.join(root, c)
            for r, _, files in os.walk(cdir):
                for fn in files:
                    if _is_image_file(fn):
                        samples.append((osp.join(r, fn), class_to_idx[c]))
        if len(samples) == 0:
            raise RuntimeError(f"Found 0 images in: {root}. Supported extensions: {extensions}")
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            img: PImage.Image = PImage.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "albumentations==2.0.8",
#     "kornia==0.8.3",
#     "timm==1.0.27",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 — Package Installation
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _():
    import subprocess
    import sys

    print("📦 Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "albumentations", "timm", "kornia", "openmim",
        "pandas", "matplotlib", "seaborn", "scikit-image", "scikit-learn",
    ], check=True)
    print("✅ All packages installed!")
    return subprocess, sys


# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — Imports and Global Setup
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import os
    import logging
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import cv2
    import numpy as np
    import random
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import Dataset, DataLoader, Subset
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import timm
    import kornia

    # Kaggle credentials — needed for dataset download and checkpoint backup.
    # Option 1: set KAGGLE_API_TOKEN and KAGGLE_USERNAME as environment variables.
    # Option 2: place a KGAT_... token in ~/.kaggle/access_token
    #           and set KAGGLE_USERNAME separately.
    _kgat_file = os.path.expanduser("~/.kaggle/access_token")
    if os.path.exists(_kgat_file) and "KAGGLE_API_TOKEN" not in os.environ:
        with open(_kgat_file) as _f:
            _tok = _f.read().strip()
        if _tok:
            os.environ["KAGGLE_API_TOKEN"] = _tok
            os.environ["KAGGLE_KEY"]       = _tok

    for _n in ["torch._inductor.select_algorithm", "torch._inductor.fx_passes",
               "torch._inductor.compile_fx", "torch.distributed"]:
        logging.getLogger(_n).setLevel(logging.ERROR)

    _inductor_cache = os.path.abspath("./inductor_cache")
    os.makedirs(_inductor_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _inductor_cache
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    SAVE_DIR = './ST_LaneNet_Weights/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs('./report_figures', exist_ok=True)

    plt.rcParams.update({'figure.dpi': 100})

    return (
        A, DataLoader, Dataset, F, SAVE_DIR, Subset, ToTensorV2,
        cv2, device, json, kornia, mo, nn, np, optim, os, pd, plt,
        random, sns, timm, torch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — Model Architecture
# Attribute names kept identical across training and evaluation so state_dicts load.
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(F, kornia, nn, timm, torch):

    class IPMTransform(nn.Module):
        def __init__(self, src_points, dst_points, output_size):
            super().__init__()
            self.output_size = output_size
            self.register_buffer(
                'M', kornia.geometry.transform.get_perspective_transform(src_points, dst_points)
            )
        def forward(self, x):
            return kornia.geometry.transform.warp_perspective(x, self.M, dsize=self.output_size)

    class PaperDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False)
            self.bn   = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x_256):
            new_64 = self.relu(self.bn(self.conv(x_256)))
            return torch.cat([x_256, new_64], dim=1), new_64

    class ImprovementBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(320, 64, kernel_size=3, padding=1, bias=False)
            self.bn   = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x_320, earlier_64):
            return self.relu(self.bn(self.conv(x_320))) + earlier_64

    class SpatialPriorHead(nn.Module):
        def __init__(self, in_channels=3):
            super().__init__()
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.dense_block       = PaperDenseBlock()
            self.improvement_block = ImprovementBlock()
            self.final_conv = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )
        def forward(self, x):
            orig_size = x.shape[2:]
            x_256 = self.init_conv(x)
            x_320, earlier_64 = self.dense_block(x_256)
            x_improved = self.improvement_block(x_320, earlier_64)
            return F.interpolate(self.final_conv(x_improved), size=orig_size,
                                 mode='bilinear', align_corners=True)

    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, dilation=1):
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, groups=in_channels, bias=False
            )
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn   = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            return self.relu(self.bn(self.pointwise(self.depthwise(x))))

    class EdgeEncoder(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.layer1 = DepthwiseSeparableConv(in_channels, 64,  stride=2, dilation=1)
            self.layer2 = DepthwiseSeparableConv(64,  128, stride=2, dilation=2)
            self.layer3 = DepthwiseSeparableConv(128, 256, stride=2, dilation=4)
        def forward(self, x):
            x1 = self.layer1(x); x2 = self.layer2(x1); x3 = self.layer3(x2)
            return x1, x2, x3

    class EfficientPyTorchCARAFE(nn.Module):
        def __init__(self, in_channels, out_channels, scale_factor=2, up_kernel=5, encoder_kernel=3):
            super().__init__()
            self.scale_factor  = scale_factor
            self.up_kernel     = up_kernel
            self.compress      = nn.Conv2d(in_channels, 64, kernel_size=1)
            kernel_channels    = (scale_factor ** 2) * (up_kernel ** 2)
            self.encoder       = nn.Conv2d(64, kernel_channels, kernel_size=encoder_kernel,
                                           padding=encoder_kernel // 2)
            self.pixel_shuffle = nn.PixelShuffle(scale_factor)
            self.final_conv    = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                                  if in_channels != out_channels else nn.Identity())
        def forward(self, x):
            B, C, H, W = x.size()
            kernels = F.softmax(self.pixel_shuffle(self.encoder(self.compress(x))), dim=1)
            pad     = self.up_kernel // 2
            patches = F.unfold(F.pad(x, [pad]*4), kernel_size=self.up_kernel, stride=1)
            patches = patches.view(B, C, self.up_kernel**2, H, W).permute(0,3,4,1,2).reshape(-1, C, self.up_kernel**2)
            kern    = kernels.view(B, self.up_kernel**2, H, self.scale_factor, W, self.scale_factor)
            kern    = kern.permute(0,2,4,1,3,5).reshape(-1, self.up_kernel**2, self.scale_factor**2)
            out     = torch.bmm(patches, kern).view(B, H, W, C, self.scale_factor, self.scale_factor)
            return self.final_conv(
                out.permute(0,3,1,4,2,5).contiguous().view(B, C, H*self.scale_factor, W*self.scale_factor)
            )

    class EdgeDecoder(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.carafe1 = EfficientPyTorchCARAFE(in_channels,    in_channels//2, scale_factor=2, up_kernel=5)
            self.carafe2 = EfficientPyTorchCARAFE(in_channels//2, in_channels//4, scale_factor=2, up_kernel=5)
            self.carafe3 = EfficientPyTorchCARAFE(in_channels//4, out_channels,   scale_factor=2, up_kernel=5)
        def forward(self, features):
            x1, x2, x3 = features
            up1 = self.carafe1(x3) + x2
            up2 = self.carafe2(up1) + x1
            return self.carafe3(up2)

    class SwinLocalizationBranch(nn.Module):
        def __init__(self, pretrained=False):
            super().__init__()
            self.backbone = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained, features_only=True, img_size=(368, 640)
            )
        def forward(self, x):
            f = self.backbone(x)[-1]
            if f.dim() == 4 and f.shape[-1] == 768:
                f = f.permute(0,3,1,2).contiguous()
            return f

    class STLaneNet_Seq(nn.Module):
        def __init__(self, num_classes=1):
            super().__init__()
            src_pts = torch.tensor([[280,200],[360,200],[640,368],[0,368]], dtype=torch.float32).unsqueeze(0)
            dst_pts = torch.tensor([[0,0],[640,0],[640,368],[0,368]],       dtype=torch.float32).unsqueeze(0)
            self.ipm           = IPMTransform(src_pts, dst_pts, output_size=(368, 640))
            self.register_buffer('M_inv', torch.linalg.inv(self.ipm.M))
            self.spatial_prior = SpatialPriorHead(in_channels=3)
            self.edge_encoder  = EdgeEncoder(in_channels=1)
            self.edge_decoder  = EdgeDecoder(in_channels=256, out_channels=64)
            self.localization  = SwinLocalizationBranch(pretrained=False)
            # 768→128: interpolating 768×12×20 → 768×368×640 at batch=32 exceeds CUDA 32-bit indexing
            self.loc_reduce    = nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.final_conv = nn.Sequential(
                nn.Conv2d(64 + 128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1),
            )
        def forward(self, x_front_view):
            # .detach() is mandatory: H is a fixed buffer with no learnable params;
            # kornia's backward tries to compute its Jacobian and crashes with CUDA error.
            x_top_down   = self.ipm(x_front_view).detach()
            prior_logits = self.spatial_prior(x_top_down)
            binary_mask  = torch.sigmoid(prior_logits)
            edge_feat    = self.edge_decoder(self.edge_encoder(binary_mask))
            edge_front   = kornia.geometry.transform.warp_perspective(
                edge_feat, self.M_inv, dsize=x_front_view.shape[2:]
            )
            loc = self.loc_reduce(self.localization(x_front_view))
            loc = F.interpolate(loc, size=edge_front.shape[2:], mode='bilinear', align_corners=True)
            return self.final_conv(torch.cat([edge_front, loc], dim=1)), prior_logits

    class STLaneNet_Par(nn.Module):
        def __init__(self, num_classes=1):
            super().__init__()
            src_pts = torch.tensor([[280,200],[360,200],[640,368],[0,368]], dtype=torch.float32).unsqueeze(0)
            dst_pts = torch.tensor([[0,0],[640,0],[640,368],[0,368]],       dtype=torch.float32).unsqueeze(0)
            self.ipm           = IPMTransform(src_pts, dst_pts, output_size=(368, 640))
            self.register_buffer('M_inv', torch.linalg.inv(self.ipm.M))
            self.spatial_prior = SpatialPriorHead(in_channels=3)
            self.edge_encoder  = EdgeEncoder(in_channels=3)
            self.edge_decoder  = EdgeDecoder(in_channels=256, out_channels=64)
            self.localization  = SwinLocalizationBranch(pretrained=False)
            self.loc_reduce    = nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.final_conv = nn.Sequential(
                nn.Conv2d(64 + 128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1),
            )
        def forward(self, x_front_view):
            x_top_down   = self.ipm(x_front_view).detach()
            prior_logits = self.spatial_prior(x_top_down)
            binary_mask  = torch.sigmoid(prior_logits)
            edge_feat    = self.edge_decoder(self.edge_encoder(x_top_down)) * binary_mask
            edge_front   = kornia.geometry.transform.warp_perspective(
                edge_feat, self.M_inv, dsize=x_front_view.shape[2:]
            )
            loc = self.loc_reduce(self.localization(x_front_view))
            loc = F.interpolate(loc, size=edge_front.shape[2:], mode='bilinear', align_corners=True)
            return self.final_conv(torch.cat([edge_front, loc], dim=1)), prior_logits

    return STLaneNet_Par, STLaneNet_Seq


# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — TuSimple Dataset Download
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo, os, subprocess, sys):
    DATASET_DIR = "tusimple"
    if os.path.exists(DATASET_DIR) and os.listdir(DATASET_DIR):
        dl_status = mo.md("✅ TuSimple dataset already present. Ready to go!")
    else:
        with mo.status.spinner(title="Downloading TuSimple from Kaggle..."):
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
                subprocess.run(["kaggle", "datasets", "download", "-d",
                                "manideep1108/tusimple", "-p", "./"], check=True)
                subprocess.run(["unzip", "-o", "-q", "tusimple.zip", "-d", DATASET_DIR], check=True)
                if os.path.exists("tusimple.zip"):
                    os.remove("tusimple.zip")
                dl_status = mo.md("✅ Dataset downloaded and extracted successfully.")
            except Exception as _e:
                dl_status = mo.md(f"⚠️ Download failed: `{_e}`")
    return (dl_status,)


@app.cell
def _(dl_status):
    dl_status
    return


# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — Dataset & DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(A, DataLoader, Dataset, Subset, ToTensorV2, cv2, json, np, os, random, torch):
    from pathlib import Path as _Path

    class TuSimpleDataset(Dataset):
        def __init__(self, image_dir, json_paths, transforms=None, line_thickness=5):
            self.image_dir      = image_dir
            self.transforms     = transforms
            self.line_thickness = line_thickness
            self.data           = self._parse_jsons(json_paths)

        def _parse_jsons(self, json_paths):
            data = []
            for p in json_paths:
                if not os.path.exists(p):
                    continue
                with open(p) as f:
                    for line in f:
                        data.append(json.loads(line))
            return data

        def _make_mask(self, lanes, h_samples, shape):
            mask = np.zeros(shape[:2], dtype=np.uint8)
            for lane in lanes:
                pts = [(x, y) for x, y in zip(lane, h_samples) if x != -2]
                if len(pts) > 1:
                    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, self.line_thickness)
            return mask

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            raw  = item['raw_file']
            if 'clips/' in raw:
                raw = raw[raw.find('clips/'):]
            img = cv2.imread(os.path.join(self.image_dir, raw))
            if img is None:
                img = np.zeros((368, 640, 3), np.uint8)
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = self._make_mask(item['lanes'], item['h_samples'], img.shape)
            if self.transforms:
                aug = self.transforms(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']
            sem = (mask.clone().detach() if isinstance(mask, torch.Tensor)
                   else torch.tensor(mask, dtype=torch.long))
            return img, sem, sem.clone().unsqueeze(0).float(), idx

    _train_tf = A.Compose([
        A.Affine(scale=(1.0, 1.0), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(height=368, width=640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    _val_tf = A.Compose([
        A.Resize(height=368, width=640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def _find_dir(name, clip_sub):
        for c in [_Path(f"tusimple/{name}"), _Path(f"tusimple/TUSimple/{name}"),
                  _Path(f"TUSimple/{name}"), _Path(name)]:
            if c.is_dir() and (c / clip_sub).exists():
                return c
        for root in [_Path("tusimple"), _Path(".")]:
            if root.exists():
                for p in root.rglob(name):
                    if p.is_dir() and (p / clip_sub).exists():
                        return p
        return None

    def _find_file(names):
        for name in names:
            for root in [_Path("tusimple"), _Path("tusimple/TUSimple"), _Path("TUSimple"), _Path(".")]:
                p = root / name
                if p.is_file():
                    return str(p)
                if root.exists():
                    hits = list(root.rglob(name))
                    if hits:
                        return str(hits[0])
        return None

    _train_dir = _find_dir("train_set", "clips")
    _test_dir  = _find_dir("test_set",  "clips")
    if not _train_dir:
        raise FileNotFoundError("Cannot locate 'train_set' directory with 'clips' subfolder!")
    if not _test_dir:
        raise FileNotFoundError("Cannot locate 'test_set' directory with 'clips' subfolder!")

    _train_jsons = [p for p in (
        _find_file([j]) for j in ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]
    ) if p is not None]
    _test_json = _find_file(["test_label_new.json", "test_label.json", "test_tasks_0627.json"])

    print(f"Train dir : {_train_dir}")
    print(f"Test  dir : {_test_dir}")

    train_dataset = TuSimpleDataset(str(_train_dir), _train_jsons, _train_tf)
    val_dataset   = TuSimpleDataset(str(_test_dir), [_test_json] if _test_json else [], _val_tf)

    random.seed(42)
    BATCH_SIZE  = 32
    NUM_WORKERS = 0

    train_loader = DataLoader(
        Subset(train_dataset, list(range(len(train_dataset)))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, list(range(len(val_dataset)))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Dataloader ready: {len(train_dataset)} train / {len(val_dataset)} val | "
          f"batch={BATCH_SIZE} | workers={NUM_WORKERS}")
    return TuSimpleDataset, train_loader, val_dataset, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 — Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(F, nn, torch):
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            targets = targets.clamp(0.0, 1.0)
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss).clamp(0.0, 1.0)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            if self.gamma == 2.0 or self.gamma == 2:
                focal_loss = alpha_t * (1 - pt) * (1 - pt) * bce_loss
            else:
                focal_loss = alpha_t * torch.pow(1 - pt, self.gamma) * bce_loss
            if self.reduction == 'mean': return focal_loss.mean()
            elif self.reduction == 'sum': return focal_loss.sum()
            else: return focal_loss

    class STLaneNetLoss(nn.Module):
        def __init__(self, alpha=0.75, gamma=2.0, prior_weight=0.4):
            super().__init__()
            self.focal_loss   = FocalLoss(alpha=alpha, gamma=gamma)
            self.prior_weight = prior_weight

        def forward(self, preds, targets):
            final_out, pred_prior = preds
            target_semantic, target_prior = targets
            target_semantic = target_semantic.unsqueeze(1).float()
            loss_final = self.focal_loss(final_out, target_semantic)
            loss_prior = self.focal_loss(pred_prior, target_prior)
            total_loss = loss_final + (self.prior_weight * loss_prior)
            return total_loss, loss_final, loss_prior

    def calculate_intersection_union(preds, labels, threshold=0.5):
        probs         = torch.sigmoid(preds)
        preds_binary  = (probs > threshold).float()
        labels_f      = labels.unsqueeze(1).float()
        intersection  = (preds_binary * labels_f).sum()
        union         = preds_binary.sum() + labels_f.sum() - intersection
        return intersection, union

    return STLaneNetLoss, calculate_intersection_union


# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 — Training Utilities
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(calculate_intersection_union, kornia, mo, np, os, torch):
    def auto_backup_to_kaggle(save_dir, epoch):
        import json as _json
        import subprocess as _sp
        import shutil

        username     = os.environ.get('KAGGLE_USERNAME', '')
        token        = os.environ.get('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_KEY', ''))
        dataset_slug = f"{username}/st-lanenet-weights-backup"

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        atp = os.path.join(kaggle_dir, "access_token")
        with open(atp, 'w') as _f:
            _f.write(token)
        os.chmod(atp, 0o600)

        meta_path = os.path.join(save_dir, "dataset-metadata.json")
        with open(meta_path, 'w') as _f:
            _json.dump({"title": "ST LaneNet Weights Backup",
                        "id": dataset_slug,
                        "licenses": [{"name": "CC0-1.0"}]}, _f)

        if not shutil.which("kaggle"):
            print("⚠️ 'kaggle' CLI not found. Skipping sync.")
            return

        res = _sp.run(
            ["kaggle", "datasets", "version", "-p", save_dir,
             "-m", f"Epoch {epoch} backup", "-q"],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            print(f"☁️ Kaggle backup updated at epoch {epoch}!")
            return

        print(f"  version error: {(res.stderr or res.stdout).strip()}")
        res_create = _sp.run(
            ["kaggle", "datasets", "create", "-p", save_dir, "-q"],
            capture_output=True, text=True
        )
        if res_create.returncode == 0:
            print(f"☁️ Kaggle dataset created at epoch {epoch}!")
        else:
            print(f"  create error: {(res_create.stderr or res_create.stdout).strip()}")
            print("⚠️ Kaggle sync skipped.")

    def train_model_production(model, model_name, train_loader, val_loader, criterion,
                               optimizer, scheduler, device, save_dir,
                               start_epoch=0, num_epochs=80, accumulation_steps=1, best_iou=0.0):
        abs_save_dir = os.path.abspath(save_dir)
        os.makedirs(abs_save_dir, exist_ok=True)
        model = model.to(device)

        def _unwrap_state_dict(m):
            raw = m._orig_mod if hasattr(m, '_orig_mod') else m
            return raw.state_dict()

        src_pts    = torch.tensor([[280,200],[360,200],[640,368],[0,368]], dtype=torch.float32).unsqueeze(0).to(device)
        dst_pts    = torch.tensor([[0,0],[640,0],[640,368],[0,368]],       dtype=torch.float32).unsqueeze(0).to(device)
        target_ipm = kornia.geometry.transform.get_perspective_transform(src_pts, dst_pts)

        history = {'train_loss': [], 'val_iou': [], 'val_acc': []}

        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            with mo.status.progress_bar(total=len(train_loader),
                                        title=f"Epoch {epoch+1}/{num_epochs} [{model_name}] TRAIN") as pbar:
                for i, (images, semantic_masks, binary_masks, _) in enumerate(train_loader):
                    images         = images.to(device)
                    semantic_masks = semantic_masks.to(device)
                    binary_masks   = binary_masks.to(device)

                    with torch.no_grad():
                        semantic_masks_warp = kornia.geometry.transform.warp_perspective(
                            semantic_masks.unsqueeze(1).float(), target_ipm, dsize=(368, 640)
                        ).squeeze(1)
                        # Threshold after warp: bilinear interpolation produces fractional
                        # values that .long() truncation would destroy.
                        semantic_masks_warp = (semantic_masks_warp > 0.5).long()
                        binary_masks_warp   = kornia.geometry.transform.warp_perspective(
                            binary_masks, target_ipm, dsize=(368, 640)
                        ).clamp(0.0, 1.0)

                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        preds = model(images)
                        total_loss, _, _ = criterion(preds, (semantic_masks, binary_masks_warp))

                    loss = total_loss.float() / accumulation_steps
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  ⚠️  NaN/Inf loss at batch {i} — skipping.")
                        optimizer.zero_grad()
                        pbar.update(1)
                        continue

                    loss.backward()
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    running_loss += total_loss.item()
                    pbar.update(1)

            model.eval()
            total_intersection = 0.0
            total_union        = 0.0
            total_n_gt         = 0
            total_n_pred       = 0

            with mo.status.progress_bar(total=len(val_loader),
                                        title=f"Epoch {epoch+1}/{num_epochs} [{model_name}] VAL") as val_pbar:
                with torch.no_grad():
                    for images, semantic_masks, _, idxs in val_loader:
                        images         = images.to(device)
                        semantic_masks = semantic_masks.to(device)

                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            preds = model(images)

                        inter, uni = calculate_intersection_union(preds[0], semantic_masks)
                        total_intersection += inter.item()
                        total_union        += uni.item()

                        probs = torch.sigmoid(preds[0]).squeeze(1).float().cpu().numpy()
                        for b, idx in enumerate(idxs):
                            item      = val_loader.dataset.dataset.data[idx.item()]
                            h_samples = item['h_samples']
                            lanes     = item['lanes']
                            prob_mask = probs[b] > 0.5
                            for lane in lanes:
                                for x_gt, y_gt in zip(lane, h_samples):
                                    if x_gt != -2:
                                        total_n_gt += 1
                                        y_down = min(int(y_gt * 368 / 720), 367)
                                        x_preds = np.where(prob_mask[y_down])[0] * (1280.0 / 640.0)
                                        if len(x_preds) > 0 and np.min(np.abs(x_preds - x_gt)) <= 20:
                                            total_n_pred += 1
                        val_pbar.update(1)

            epoch_val_iou = total_intersection / (total_union + 1e-6)
            epoch_val_acc = total_n_pred / (total_n_gt + 1e-6)
            epoch_loss    = running_loss / len(train_loader)

            history['train_loss'].append(epoch_loss)
            history['val_iou'].append(epoch_val_iou)
            history['val_acc'].append(epoch_val_acc)

            print(f"Epoch {epoch+1} → Loss: {epoch_loss:.4f} | IoU: {epoch_val_iou:.4f} | Acc: {epoch_val_acc:.4f}")

            csv_path   = os.path.join(abs_save_dir, f"{model_name}_metrics.csv")
            current_lr = optimizer.param_groups[0]['lr']
            with open(csv_path, 'a') as _f:
                if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
                    _f.write("epoch,train_loss,val_iou,val_acc,learning_rate\n")
                _f.write(f"{epoch+1},{epoch_loss:.6f},{epoch_val_iou:.6f},{epoch_val_acc:.6f},{current_lr:.6e}\n")

            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': _unwrap_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
            }
            if epoch_val_iou > best_iou:
                best_iou = epoch_val_iou
                ckpt['best_iou'] = best_iou
                save_path = os.path.join(abs_save_dir, f"{model_name}_best.pth")
                torch.save(ckpt, save_path)
                print(f"--> New best saved: {save_path} (IoU: {best_iou:.4f})")

            torch.save(ckpt, os.path.join(abs_save_dir, f"{model_name}_latest.pth"))
            auto_backup_to_kaggle(abs_save_dir, epoch + 1)
            scheduler.step()

            try:
                import matplotlib.pyplot as _plt
                _mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(images.device)
                _std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(images.device)
                _img_vis  = ((images[0] * _std + _mean).cpu().permute(1,2,0)).clamp(0,1)
                _prob_vis = torch.sigmoid(preds[0][0].detach().float()).squeeze().cpu()
                _fig, _ax = _plt.subplots(1, 3, figsize=(15, 5))
                _ax[0].imshow(_img_vis.numpy()); _ax[0].set_title(f"Epoch {epoch+1} Input")
                _ax[1].imshow(semantic_masks[0].cpu().numpy(), cmap='gray'); _ax[1].set_title("GT Mask")
                _ax[2].imshow(_prob_vis.numpy(), cmap='hot', vmin=0, vmax=1); _ax[2].set_title("Prediction")
                _plt.tight_layout(); _plt.show(); _plt.close(_fig)
            except Exception as _vis_err:
                print(f"  Visualization skipped: {_vis_err}")

        return history

    return (train_model_production,)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 8 — Training Execution (Phase 1: Par, Phase 2: Seq)
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(
    SAVE_DIR,
    STLaneNetLoss,
    STLaneNet_Par,
    STLaneNet_Seq,
    device,
    optim,
    os,
    torch,
    train_loader,
    train_model_production,
    val_loader,
):
    def _pull_from_kaggle(save_dir):
        import subprocess as _sp
        dataset_slug = f"{os.environ.get('KAGGLE_USERNAME')}/st-lanenet-weights-backup"
        os.makedirs(save_dir, exist_ok=True)
        if all(os.path.exists(os.path.join(save_dir, f))
               for f in ["STLaneNet_Par_latest.pth", "STLaneNet_Seq_latest.pth"]):
            print("☑️  Local checkpoints found — skipping Kaggle download.")
            return
        print(f"Checking Kaggle for previous backups ({dataset_slug})...")
        res = _sp.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug,
             "-p", save_dir, "--unzip", "-q"],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            print("☁️ Pulled previous weights from Kaggle!")
        else:
            print("Notice: No previous backups found — starting fresh.")

    _pull_from_kaggle(SAVE_DIR)

    criterion    = STLaneNetLoss(alpha=0.75, gamma=2.0, prior_weight=0.4)
    total_epochs = 50

    def _resume(model, model_name, optimizer, scheduler):
        latest = os.path.join(SAVE_DIR, f"{model_name}_latest.pth")
        best   = os.path.join(SAVE_DIR, f"{model_name}_best.pth")
        path   = latest if os.path.exists(latest) else (best if os.path.exists(best) else None)
        start_epoch, best_iou = 0, 0.0
        if path:
            print(f"Loading checkpoint: {path}")
            ckpt = torch.load(path, map_location=device)
            try:
                model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                start_epoch = ckpt['epoch']
                best_iou    = ckpt.get('best_iou', 0.0)
                print(f"  Resuming from epoch {start_epoch} (best IoU: {best_iou:.4f})")
            except RuntimeError as _e:
                print(f"⚠️  Architecture mismatch — starting fresh.\n   {_e}")
                start_epoch, best_iou = 0, 0.0
        return start_epoch, best_iou

    print("=" * 50)
    print("PHASE 1: TRAINING PARALLEL MODEL")
    print("=" * 50)
    _model_par     = STLaneNet_Par(num_classes=1).to(device)
    _optimizer_par = optim.AdamW(_model_par.parameters(), lr=5e-4, weight_decay=0.005)
    _scheduler_par = optim.lr_scheduler.PolynomialLR(_optimizer_par, total_iters=total_epochs, power=0.9)
    _start_par, _best_iou_par = _resume(_model_par, "STLaneNet_Par", _optimizer_par, _scheduler_par)

    if _start_par < total_epochs:
        train_model_production(
            model=_model_par, model_name="STLaneNet_Par",
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=_optimizer_par, scheduler=_scheduler_par,
            device=device, save_dir=SAVE_DIR,
            start_epoch=_start_par, num_epochs=total_epochs, best_iou=_best_iou_par,
        )
    else:
        print("Parallel model already fully trained.")

    del _model_par, _optimizer_par, _scheduler_par
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("PHASE 2: TRAINING SEQUENTIAL MODEL")
    print("=" * 50)
    _model_seq     = STLaneNet_Seq(num_classes=1).to(device)
    _optimizer_seq = optim.AdamW(_model_seq.parameters(), lr=5e-4, weight_decay=0.005)
    _scheduler_seq = optim.lr_scheduler.PolynomialLR(_optimizer_seq, total_iters=total_epochs, power=0.9)
    _start_seq, _best_iou_seq = _resume(_model_seq, "STLaneNet_Seq", _optimizer_seq, _scheduler_seq)

    if _start_seq < total_epochs:
        train_model_production(
            model=_model_seq, model_name="STLaneNet_Seq",
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=_optimizer_seq, scheduler=_scheduler_seq,
            device=device, save_dir=SAVE_DIR,
            start_epoch=_start_seq, num_epochs=total_epochs, best_iou=_best_iou_seq,
        )
    else:
        print("Sequential model already fully trained.")
    return


# ─────────────────────────────────────────────────────────────────────────────
# Cell 9 — LanePostProcessor
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(cv2, np):
    class LanePostProcessor:
        def __init__(self, threshold=0.5, min_area=100):
            self.threshold = threshold
            self.min_area  = min_area

        def __call__(self, prob_map: np.ndarray) -> np.ndarray:
            binary   = (prob_map >= self.threshold).astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            cleaned  = np.zeros_like(binary)
            for lbl in range(1, n_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= self.min_area:
                    cleaned[labels == lbl] = 1
            return cleaned.astype(bool)

    postprocessor = LanePostProcessor(threshold=0.5, min_area=100)
    print("✅ LanePostProcessor ready  (threshold=0.5 | min_area=100)")
    return LanePostProcessor, postprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Cell 10 — Pull Model Weights from Kaggle (skipped if already trained locally)
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(SAVE_DIR, mo, os, subprocess, sys):
    def _pull_weights(save_dir):
        username = os.environ.get('KAGGLE_USERNAME', '')
        token    = os.environ.get('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_KEY', ''))
        slug     = f"{username}/st-lanenet-weights-backup"

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        atp = os.path.join(kaggle_dir, "access_token")
        with open(atp, 'w') as _f:
            _f.write(token)
        os.chmod(atp, 0o600)

        if all(os.path.exists(os.path.join(save_dir, p))
               for p in ["STLaneNet_Par_best.pth", "STLaneNet_Seq_best.pth"]):
            print("☑️  Best checkpoints already present — skipping download.")
            return True

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
        print(f"⬇️  Downloading weights from {slug}...")
        r = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", save_dir, "--unzip", "-q"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print("✅ Weights downloaded successfully!")
            return True
        print(f"⚠️  Kaggle download failed: {(r.stderr or r.stdout).strip()}")
        return False

    with mo.status.spinner(title="Pulling model weights from Kaggle backup..."):
        weights_ok = _pull_weights(SAVE_DIR)

    mo.md(f"{'✅ Model weights are ready.' if weights_ok else '⚠️ Could not download weights — check credentials or upload manually.'}")
    return (weights_ok,)


# ─────────────────────────────────────────────────────────────────────────────
# Section A — Load Training Metrics CSVs
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(SAVE_DIR, os, pd):
    def _load_csv(model_name):
        for path in [
            os.path.join(SAVE_DIR, f"{model_name}_metrics.csv"),
            f"{model_name}_metrics.csv",
        ]:
            if os.path.exists(path):
                return pd.read_csv(path)
        raise FileNotFoundError(f"Metrics CSV not found for {model_name}.")

    df_par = _load_csv("STLaneNet_Par")
    df_seq = _load_csv("STLaneNet_Seq")
    print(f"Loaded — Par: {len(df_par)} epochs | Seq: {len(df_seq)} epochs")
    return df_par, df_seq


@app.cell
def _(df_par, df_seq, mo):
    mo.vstack([
        mo.md("## 📋 Section A — Training Metrics Data"),
        mo.md("**STLaneNet_Par** (Parallel)"),
        mo.ui.table(df_par.round(6), selection=None),
        mo.md("**STLaneNet_Seq** (Sequential)"),
        mo.ui.table(df_seq.round(6), selection=None),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Section B — Training Curves
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_par, df_seq, mo, plt):
    _fig_b, _ax = plt.subplots(2, 2, figsize=(16, 10))
    _fig_b.suptitle('ST-LaneNet — Training History', fontsize=14, fontweight='bold', y=1.01)

    _ep_p     = df_par['epoch'].values
    _ep_s     = df_seq['epoch'].values
    _best_p   = int(df_par.loc[df_par['val_iou'].idxmax(), 'epoch'])
    _best_s   = int(df_seq.loc[df_seq['val_iou'].idxmax(), 'epoch'])

    _ax[0,0].plot(_ep_p, df_par['train_loss'], lw=2.0, label='Par')
    _ax[0,0].plot(_ep_s, df_seq['train_loss'], lw=2.0, ls='--', label='Seq')
    _ax[0,0].set_title('Training Loss (Focal)'); _ax[0,0].set_xlabel('Epoch')
    _ax[0,0].set_ylabel('Loss'); _ax[0,0].legend(); _ax[0,0].grid(True)

    _ax[0,1].plot(_ep_p, df_par['val_iou'], lw=2.0, label='Par')
    _ax[0,1].plot(_ep_s, df_seq['val_iou'], lw=2.0, ls='--', label='Seq')
    _ax[0,1].axvline(_best_p, ls=':', alpha=0.6, label=f'Par best (ep {_best_p})')
    _ax[0,1].axvline(_best_s, ls=':', alpha=0.6, label=f'Seq best (ep {_best_s})')
    _ax[0,1].set_title('Validation IoU'); _ax[0,1].set_xlabel('Epoch')
    _ax[0,1].set_ylabel('IoU'); _ax[0,1].legend(fontsize=9); _ax[0,1].grid(True)

    _ax[1,0].plot(_ep_p, df_par['val_acc'], lw=2.0, label='Par')
    _ax[1,0].plot(_ep_s, df_seq['val_acc'], lw=2.0, ls='--', label='Seq')
    _ax[1,0].set_title('Validation Accuracy (TuSimple point-based)')
    _ax[1,0].set_xlabel('Epoch'); _ax[1,0].set_ylabel('Accuracy')
    _ax[1,0].legend(); _ax[1,0].grid(True)

    _ax[1,1].plot(_ep_p, df_par['learning_rate'], lw=2.0, label='Par')
    _ax[1,1].plot(_ep_s, df_seq['learning_rate'], lw=2.0, ls='--', label='Seq')
    _ax[1,1].set_title('Learning Rate (PolynomialLR, power=0.9)')
    _ax[1,1].set_xlabel('Epoch'); _ax[1,1].set_ylabel('LR')
    _ax[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    _ax[1,1].legend(); _ax[1,1].grid(True)

    plt.tight_layout()
    _fig_b.savefig('./report_figures/training_curves.png', dpi=150, bbox_inches='tight')
    _html_b = mo.as_html(_fig_b)
    plt.close(_fig_b)

    mo.vstack([mo.md("## Section B — Training Curves"), _html_b])


# ─────────────────────────────────────────────────────────────────────────────
# Section C — Best Metrics Comparison
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_par, df_seq, mo, np, pd, plt):
    _bp = df_par.loc[df_par['val_iou'].idxmax()]
    _bs = df_seq.loc[df_seq['val_iou'].idxmax()]

    _summary = pd.DataFrame({
        'Model':           ['STLaneNet_Par', 'STLaneNet_Seq'],
        'Best Val IoU':    [round(float(_bp['val_iou']), 4), round(float(_bs['val_iou']), 4)],
        'Best Val Acc':    [round(float(_bp['val_acc']), 4), round(float(_bs['val_acc']), 4)],
        'Best IoU Epoch':  [int(_bp['epoch']),               int(_bs['epoch'])],
        'Final Train Loss':[round(float(df_par['train_loss'].iloc[-1]), 6),
                            round(float(df_seq['train_loss'].iloc[-1]), 6)],
        'Final LR':        [f"{df_par['learning_rate'].iloc[-1]:.3e}",
                            f"{df_seq['learning_rate'].iloc[-1]:.3e}"],
    })

    _metrics = ['Best Val IoU', 'Best Val Acc']
    _x = np.arange(len(_metrics))
    _w = 0.30
    _fig_c, _ax_c = plt.subplots(figsize=(9, 5))
    _b1 = _ax_c.bar(_x - _w/2, [_summary.loc[0, m] for m in _metrics], width=_w, alpha=0.85, label='Par')
    _b2 = _ax_c.bar(_x + _w/2, [_summary.loc[1, m] for m in _metrics], width=_w, alpha=0.85, label='Seq')
    _ax_c.bar_label(_b1, fmt='%.4f', padding=4, fontsize=10)
    _ax_c.bar_label(_b2, fmt='%.4f', padding=4, fontsize=10)
    _ax_c.set_xticks(_x); _ax_c.set_xticklabels(_metrics, fontsize=12)
    _ax_c.set_ylim(0, 1.05); _ax_c.set_ylabel('Score')
    _ax_c.set_title('Best Validation Metrics — Par vs. Seq', fontsize=13)
    _ax_c.legend(); _ax_c.grid(True, axis='y')

    plt.tight_layout()
    _html_c = mo.as_html(_fig_c)
    plt.close(_fig_c)

    mo.vstack([
        mo.md("## Section C — Best Metrics Comparison"),
        _html_c,
        mo.md("### Summary Table"),
        mo.ui.table(_summary, selection=None),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Section D — Load Both Best Models
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(SAVE_DIR, STLaneNet_Par, STLaneNet_Seq, device, mo, os, torch, weights_ok):
    _ = weights_ok  # ordering guard

    def _load_best(ModelClass, model_name, save_dir, dev):
        ckpt_path = os.path.join(save_dir, f"{model_name}_best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = ModelClass(num_classes=1).to(dev)
        ckpt  = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  ✅ {model_name:<20} | epoch {ckpt.get('epoch','?')} | best IoU {ckpt.get('best_iou', float('nan')):.4f}")
        return model

    print("Loading checkpoints...")
    with mo.status.spinner(title="Loading STLaneNet_Par best checkpoint..."):
        model_par = _load_best(STLaneNet_Par, "STLaneNet_Par", SAVE_DIR, device)
    with mo.status.spinner(title="Loading STLaneNet_Seq best checkpoint..."):
        model_seq = _load_best(STLaneNet_Seq, "STLaneNet_Seq", SAVE_DIR, device)
    print("Both models loaded and in eval() mode.")
    return model_par, model_seq


# ─────────────────────────────────────────────────────────────────────────────
# Section E — Full Inference Loop
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(cv2, device, mo, model_par, model_seq, np, postprocessor, torch, val_dataset, val_loader):
    _H, _W = 368, 640
    _MEAN  = np.array([0.485, 0.456, 0.406], np.float32)
    _STD   = np.array([0.229, 0.224, 0.225], np.float32)

    import time as _time
    _st = {
        m: {v: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'n_gt': 0, 'n_pred': 0,
                'n_gt_lanes': 0, 'fp_lanes': 0, 'fn_lanes': 0}
            for v in ('raw', 'pp')}
        for m in ('par', 'seq')
    }

    fp_maps   = {m: {v: np.zeros((_H, _W), np.float32) for v in ('raw', 'pp')} for m in ('par', 'seq')}
    fn_maps   = {m: {v: np.zeros((_H, _W), np.float32) for v in ('raw', 'pp')} for m in ('par', 'seq')}
    viz_samples  = []
    _n_viz_target = 4
    _n_samples    = 0
    time_par = 0.0
    time_seq = 0.0

    with mo.status.progress_bar(
        total=len(val_loader), title="🔍 Running inference on validation set..."
    ) as _pbar:
        with torch.no_grad():
            for _batch_idx, (_images, _sem_masks, _, _idxs) in enumerate(val_loader):
                _images    = _images.to(device)
                _sem_masks = _sem_masks.to(device)
                _B         = _images.shape[0]
                _n_samples += _B
                _use_amp   = device.type == 'cuda'

                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = _time.time()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=_use_amp):
                    _out_par, _ = model_par(_images)
                if device.type == 'cuda': torch.cuda.synchronize()
                t1 = _time.time()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=_use_amp):
                    _out_seq, _ = model_seq(_images)
                if device.type == 'cuda': torch.cuda.synchronize()
                t2 = _time.time()

                time_par += (t1 - t0)
                time_seq += (t2 - t1)

                _prob_par = torch.sigmoid(_out_par.float()).squeeze(1).cpu().numpy()
                _prob_seq = torch.sigmoid(_out_seq.float()).squeeze(1).cpu().numpy()
                _gt_np    = _sem_masks.cpu().numpy().astype(bool)

                if _batch_idx == 0:
                    for _b in range(min(_n_viz_target, _B)):
                        _img_np = _images[_b].cpu().permute(1,2,0).float().numpy()
                        _img_np = (_img_np * _STD + _MEAN).clip(0.0, 1.0)
                        viz_samples.append({
                            'image':    _img_np,
                            'gt':       _gt_np[_b],
                            'par_prob': _prob_par[_b],
                            'seq_prob': _prob_seq[_b],
                        })

                for _b, _idx in enumerate(_idxs):
                    _gt   = _gt_np[_b]
                    _item = val_dataset.data[_idx.item()]
                    _h_samples = _item['h_samples']
                    _lanes     = _item['lanes']

                    valid_gt_lanes = []
                    for _lane in _lanes:
                        _l_pts = {_y: _x for _x, _y in zip(_lane, _h_samples) if _x != -2}
                        if _l_pts:
                            valid_gt_lanes.append(_l_pts)

                    for _mkey, _prob in (('par', _prob_par[_b]), ('seq', _prob_seq[_b])):
                        _raw_pred = (_prob >= 0.5)
                        _pp_pred  = postprocessor(_prob)

                        for _vkey, _pred in (('raw', _raw_pred), ('pp', _pp_pred)):
                            _s   = _st[_mkey][_vkey]
                            _p_b = _pred.astype(bool)

                            _s['tp'] += int(( _p_b &  _gt).sum())
                            _s['fp'] += int(( _p_b & ~_gt).sum())
                            _s['fn'] += int((~_p_b &  _gt).sum())
                            _s['tn'] += int((~_p_b & ~_gt).sum())

                            for _lane in _lanes:
                                for _x_gt, _y_gt in zip(_lane, _h_samples):
                                    if _x_gt != -2:
                                        _s['n_gt'] += 1
                                        _y_d = min(int(_y_gt * _H / 720), _H - 1)
                                        _x_preds = np.where(_p_b[_y_d])[0] * (1280.0 / _W)
                                        if len(_x_preds) > 0 and np.min(np.abs(_x_preds - _x_gt)) <= 20:
                                            _s['n_pred'] += 1

                            _THRESH_PX   = max(1, int(20 * _W / 1280))
                            _MATCH_RATIO = 0.85
                            _s['n_gt_lanes'] += len(valid_gt_lanes)

                            _kernel      = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (2*_THRESH_PX+1, 2*_THRESH_PX+1)
                            )
                            _gt_corridor = cv2.dilate(_gt.astype(np.uint8), _kernel).astype(bool)

                            for _g_lane in valid_gt_lanes:
                                _total   = len(_g_lane)
                                _matched = sum(
                                    1 for _yg, _xg in _g_lane.items()
                                    if _p_b[min(int(_yg*_H/720),_H-1),
                                           max(0, int(_xg*_W/1280)-_THRESH_PX):
                                           min(_W, int(_xg*_W/1280)+_THRESH_PX+1)].any()
                                )
                                if _matched / _total < _MATCH_RATIO:
                                    _s['fn_lanes'] += 1

                            _fp_px = (_p_b & ~_gt_corridor).astype(np.uint8)
                            _n_blobs, _, _blob_stats, _ = cv2.connectedComponentsWithStats(_fp_px)
                            for _lbl in range(1, _n_blobs):
                                if _blob_stats[_lbl, cv2.CC_STAT_AREA] >= 50:
                                    _s['fp_lanes'] += 1

                            fp_maps[_mkey][_vkey] += (_p_b & ~_gt).astype(np.float32)
                            fn_maps[_mkey][_vkey] += (~_p_b & _gt).astype(np.float32)

                _pbar.update(1)

    for _mk in ('par', 'seq'):
        for _vk in ('raw', 'pp'):
            fp_maps[_mk][_vk] /= max(_n_samples, 1)
            fn_maps[_mk][_vk] /= max(_n_samples, 1)

    fps_par = _n_samples / max(time_par, 1e-5)
    fps_seq = _n_samples / max(time_seq, 1e-5)

    def _metrics(s):
        tp_pix, fp_pix, fn_pix = s['tp'], s['fp'], s['fn']
        iou      = tp_pix / (tp_pix + fp_pix + fn_pix + 1e-8)
        acc      = s['n_pred'] / (s['n_gt'] + 1e-8)
        tp_lane  = s['n_gt_lanes'] - s['fn_lanes']
        fp_lane  = s['fp_lanes']
        fn_lane  = s['fn_lanes']
        prec     = tp_lane / (tp_lane + fp_lane + 1e-8)
        rec      = tp_lane / (tp_lane + fn_lane + 1e-8)
        f1       = 2 * prec * rec / (prec + rec + 1e-8)
        fpr      = fp_lane / (tp_lane + fp_lane + 1e-8)
        fnr      = fn_lane / (s['n_gt_lanes'] + 1e-8)
        return dict(IoU=iou, Precision=prec, Recall=rec, F1=f1,
                    TuSimple_Acc=acc, Detected=tp_lane, FP_lane=fp_lane, FN_lane=fn_lane,
                    TPR=rec, FPR=fpr, FNR=fnr,
                    TP=tp_pix, FP=fp_pix, FN=fn_pix, TN=s['tn'])

    metrics_raw = {m: _metrics(_st[m]['raw']) for m in ('par', 'seq')}
    metrics_pp  = {m: _metrics(_st[m]['pp'])  for m in ('par', 'seq')}

    print(f"✅ Inference complete — {_n_samples} samples processed.")
    return fp_maps, fn_maps, fps_par, fps_seq, metrics_pp, metrics_raw, viz_samples


# ─────────────────────────────────────────────────────────────────────────────
# Section E — Metrics Display Table
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(fps_par, fps_seq, metrics_pp, metrics_raw, mo, pd):
    _rows = []
    for _label, _mdict, _fps in [
        ('Par  — Raw',        metrics_raw['par'], fps_par),
        ('Par  — Post-Proc',  metrics_pp['par'],  fps_par),
        ('Seq  — Raw',        metrics_raw['seq'], fps_seq),
        ('Seq  — Post-Proc',  metrics_pp['seq'],  fps_seq),
    ]:
        _rows.append({
            'Model':         _label,
            'IoU':           f"{_mdict['IoU']:.4f}",
            'TuSimple Acc':  f"{_mdict['TuSimple_Acc']:.4f}",
            'Precision':     f"{_mdict['Precision']:.4f}",
            'Recall':        f"{_mdict['Recall']:.4f}",
            'F1':            f"{_mdict['F1']:.4f}",
            'Detected (TP)': f"{_mdict['Detected']}",
            'FP':            f"{_mdict['FP_lane']}",
            'FN':            f"{_mdict['FN_lane']}",
            'TPR (%)':       f"{_mdict['TPR']:.2%}",
            'FPR (%)':       f"{_mdict['FPR']:.2%}",
            'FNR (%)':       f"{_mdict['FNR']:.2%}",
            'FPS':           f"{_fps:.1f}",
        })

    mo.vstack([
        mo.md("## Section E — Evaluation Metrics\n\n"
              "*Raw* = sigmoid threshold at 0.5  ·  *Post-Proc* = noise removal (min_area=100)"),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
        mo.md("> **Note:** `IoU` is pixel-level (informational only). Lane-level metrics use GT-guided "
              "evaluation: TP if ≥85% of annotated points have a predicted pixel within 20 px. "
              "FP counts predicted blobs (≥50 px²) entirely outside the 20 px GT corridor. "
              "`FPR` = FP/predicted lanes · `FNR` = FN/GT lanes."),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Section F — Qualitative Visualization
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo, plt, postprocessor, viz_samples):
    _N    = len(viz_samples)
    _ROWS = 8
    _fig_f, _axs_f = plt.subplots(_ROWS, _N, figsize=(_N * 4.5, _ROWS * 3.5))
    _fig_f.suptitle('Section F — Qualitative Lane Predictions', fontsize=15, fontweight='bold', y=1.0)

    _titles = ['Input Image', 'Ground Truth',
               'Par — Raw Prob', 'Par — Mask', 'Par — Overlay',
               'Seq — Raw Prob', 'Seq — Mask', 'Seq — Overlay']

    for _i, _s in enumerate(viz_samples):
        _img      = _s['image']
        _gt       = _s['gt']
        _par_pp   = postprocessor(_s['par_prob'])
        _seq_pp   = postprocessor(_s['seq_prob'])

        _par_vis = _img.copy(); _par_vis[_par_pp] = [1.0, 0.15, 0.15]
        _seq_vis = _img.copy(); _seq_vis[_seq_pp] = [0.10, 0.35, 1.0]

        for _r, _data in enumerate([
            (_img,  {},               {}),
            (_gt,   {'cmap':'gray'},  {'vmin':0,'vmax':1}),
            (_s['par_prob'], {'cmap':'viridis'}, {'vmin':0,'vmax':1}),
            (_par_pp, {'cmap':'gray'}, {'vmin':0,'vmax':1}),
            (_par_vis, {}, {}),
            (_s['seq_prob'], {'cmap':'viridis'}, {'vmin':0,'vmax':1}),
            (_seq_pp, {'cmap':'gray'}, {'vmin':0,'vmax':1}),
            (_seq_vis, {}, {}),
        ]):
            _axs_f[_r, _i].imshow(_data[0], **_data[1], **_data[2])
            _axs_f[_r, _i].axis('off')
            _axs_f[_r, _i].set_title(_titles[_r], fontsize=11, pad=6)

    plt.tight_layout()
    _fig_f.savefig('./report_figures/qualitative_viz.png', dpi=100, bbox_inches='tight')
    _html_f = mo.as_html(_fig_f)
    plt.close(_fig_f)

    mo.vstack([
        mo.md("## Section F — Qualitative Visualization\n\n"
              "Each column is a validation sample. Rows: Input · GT · Par prob · Par mask · "
              "Par overlay · Seq prob · Seq mask · Seq overlay."),
        _html_f,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Section G1 — Pixel-Level Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(metrics_pp, metrics_raw, mo, np, plt, sns):

    def _plot_cm(ax, m, title, cmap_name):
        tp, fp, fn, tn = m['TP'], m['FP'], m['FN'], m['TN']
        total  = tp + fp + fn + tn
        pct    = np.array([[tn/total, fp/total], [fn/total, tp/total]]) * 100
        labels = np.array([
            [f"TN\n{tn:,}\n({pct[0,0]:.1f}%)",  f"FP\n{fp:,}\n({pct[0,1]:.2f}%)"],
            [f"FN\n{fn:,}\n({pct[1,0]:.2f}%)",  f"TP\n{tp:,}\n({pct[1,1]:.2f}%)"],
        ])
        sns.heatmap(pct, annot=labels, fmt='', ax=ax, cmap=cmap_name,
                    linewidths=1.5, linecolor='white',
                    xticklabels=['Predicted Neg', 'Predicted Pos'],
                    yticklabels=['Actual Neg',    'Actual Pos'],
                    annot_kws={'size': 9})
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel('Prediction'); ax.set_ylabel('Ground Truth')

    _fig_g1, _axs_g1 = plt.subplots(2, 2, figsize=(13, 10))
    _fig_g1.suptitle('Section G1 — Pixel-Level Confusion Matrices',
                     fontsize=13, fontweight='bold', y=1.02)

    _plot_cm(_axs_g1[0,0], metrics_raw['par'], 'Par  —  Raw',       'Blues')
    _plot_cm(_axs_g1[0,1], metrics_pp['par'],  'Par  —  Post-Proc', 'Blues')
    _plot_cm(_axs_g1[1,0], metrics_raw['seq'], 'Seq  —  Raw',       'Oranges')
    _plot_cm(_axs_g1[1,1], metrics_pp['seq'],  'Seq  —  Post-Proc', 'Oranges')

    plt.tight_layout()
    _fig_g1.savefig('./report_figures/confusion_matrices.png', dpi=150, bbox_inches='tight')
    _html_g1 = mo.as_html(_fig_g1)
    plt.close(_fig_g1)

    mo.vstack([
        mo.md("## Section G1 — Confusion Matrices\n\n"
              "Values shown as **percentage of total pixels**. "
              "TN always dominates (background >> lanes) — focus on TP/FP/FN cells."),
        _html_g1,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Section G2 / G3 — Spatial FP & FN Error Heatmaps
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(fp_maps, fn_maps, mo, plt):
    _panels = [
        (fp_maps['par']['raw'], 'Par FP — Raw  (false positives)',          'hot'),
        (fp_maps['par']['pp'],  'Par FP — Post-Proc',                       'hot'),
        (fn_maps['par']['raw'], 'Par FN — Raw  (missed ground-truth lanes)', 'Blues'),
        (fn_maps['par']['pp'],  'Par FN — Post-Proc',                       'Blues'),
        (fp_maps['seq']['raw'], 'Seq FP — Raw  (false positives)',          'hot'),
        (fp_maps['seq']['pp'],  'Seq FP — Post-Proc',                       'hot'),
        (fn_maps['seq']['raw'], 'Seq FN — Raw  (missed lanes)',              'Blues'),
        (fn_maps['seq']['pp'],  'Seq FN — Post-Proc',                       'Blues'),
    ]

    _fig_g2, _axs_g2 = plt.subplots(4, 2, figsize=(16, 24))
    _fig_g2.suptitle(
        'Section G2 / G3 — Spatial FP & FN Error Heatmaps\n'
        'Each pixel shows average error rate across the validation set.',
        fontsize=13, fontweight='bold', y=1.01,
    )

    for _r, (_data, _title, _cm) in enumerate(_panels):
        _ri, _ci = _r // 2, _r % 2
        _im = _axs_g2[_ri, _ci].imshow(_data, cmap=_cm, vmin=0.0)
        _axs_g2[_ri, _ci].set_title(_title, fontsize=10, pad=5)
        _axs_g2[_ri, _ci].axis('off')
        _fig_g2.colorbar(_im, ax=_axs_g2[_ri, _ci], fraction=0.028, pad=0.02)

    plt.tight_layout()
    _fig_g2.savefig('./report_figures/spatial_heatmaps.png', dpi=100, bbox_inches='tight')
    _html_g2 = mo.as_html(_fig_g2)
    plt.close(_fig_g2)

    mo.vstack([
        mo.md("## Section G2 / G3 — Spatial Error Heatmaps\n\n"
              "**FP map** (`hot`): where the model predicts lanes that don't exist.  \n"
              "**FN map** (`Blues`): where the model misses ground-truth lane pixels.  \n"
              "Bright regions show systematic geometric failure patterns."),
        _html_g2,
    ])


if __name__ == "__main__":
    app.run()

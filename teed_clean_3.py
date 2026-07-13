from __future__ import print_function

"""
TEED: Tiny and Efficient Edge Detector
======================================
Single-file consolidated implementation with:
  - Full TED model architecture (Smish activation + DoubleFusion)
  - bdcn_loss2 + cats_loss combo loss
  - Auto dataset loading from data/raw/ + data/mask/
  - Comprehensive data augmentation pipeline
  - Complete training scheduler (Step/Cosine/Plateau/OneCycle + Warmup)
  - Inference on single images or folders
  - ONNX export support
  - All settings in TEEDConfig dataclass

Usage:
  python teed_clean.py --mode train
  python teed_clean.py --mode train --epochs 20 --lr 1e-3 --img_size 480
  python teed_clean.py --mode train --resume_from ./checkpoints/epoch_5.pth
  python teed_clean.py --mode infer --image_path ./test.jpg --checkpoint ./checkpoints/best_model.pth
  python teed_clean.py --mode infer --image_dir ./test_images --checkpoint ./checkpoints/best_model.pth
  python teed_clean.py --mode export_onnx --checkpoint ./checkpoints/best_model.pth
  python teed_clean.py --mode info
"""

# ============================================================================
# IDE Direct Run Settings (used when running without command-line arguments)
# Modify the values below to control what happens when you press Run in IDE.
# ============================================================================
IDE_MODE = "info"               # train | infer | export_onnx | info
IDE_EPOCHS = 10                 # Only used for train mode
IDE_LR = 8e-4                   # Only used for train mode
IDE_IMG_SIZE = 352              # Only used for train mode
IDE_BATCH_SIZE = 8              # Only used for train mode
IDE_CHECKPOINT = "./checkpoints/best_model.pth"  # For infer / export_onnx
IDE_RESULT_DIR = "./results"                     # For infer
IDE_IMAGE_PATH = ""             # Single image for infer mode
IDE_IMAGE_DIR = ""              # Image directory for infer mode


import os
import sys
import time
import random
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class TEEDConfig:
    """All-in-one configuration for TEED training, inference, and ONNX export."""

    # ── Paths ──────────────────────────────────────────────────────────
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"
    log_dir: str = "./logs"

    # ── Training hyperparameters ───────────────────────────────────────
    epochs: int = 10
    batch_size: int = 8
    lr: float = 8e-4
    wd: float = 2e-4
    img_size: int = 352
    num_workers: int = 4
    seed: int = 1021
    fp16: bool = False
    grad_clip: float = 0.0

    # ── Validation split ───────────────────────────────────────────────
    val_split: float = 0.2

    # ── Learning rate scheduler ────────────────────────────────────────
    lr_scheduler: str = "step"
    lr_milestones: tuple = (4,)
    lr_gamma: float = 0.1
    lr_min: float = 1e-6
    lr_warmup_epochs: int = 0
    lr_warmup_start: float = 1e-6
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    onecycle_pct_start: float = 0.3

    # ── Loss weights ───────────────────────────────────────────────────
    loss_bdcn_weights: tuple = (1.1, 0.7, 1.1, 1.3)
    loss_cats_weights: tuple = (0.01, 3.0)

    # ── Data augmentation ──────────────────────────────────────────────
    aug_scale_range: tuple = (0.5, 1.5)
    aug_rotation: float = 15.0
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.2
    aug_hue: float = 0.05
    aug_crop_prob: float = 0.4
    aug_crop_min_size: int = 256
    aug_edge_boost: float = 0.2

    # ── Mean pixel values for normalization ────────────────────────────
    mean_pixels: tuple = (103.939, 116.779, 123.68)

    # ── Training schedule ──────────────────────────────────────────────
    save_interval: int = 1
    val_interval: int = 1
    log_interval: int = 20
    viz_interval: int = 200
    early_stopping_patience: int = 0
    resume_from: str = ""

    # ── ONNX export ────────────────────────────────────────────────────
    onnx_export: bool = False
    onnx_input_size: tuple = (1, 3, 480, 480)
    onnx_opset: int = 14
    onnx_dynamic_axes: bool = True
    onnx_simplify: bool = False

    # ── Inference ──────────────────────────────────────────────────────
    predict_all_outputs: bool = False


# ============================================================================
# 2. ACTIVATION FUNCTION (Smish)
# ============================================================================

@torch.jit.script
def smish(input: torch.Tensor) -> torch.Tensor:
    """Smish: input * tanh(ln(1 + sigmoid(input)))."""
    return input * torch.tanh(torch.log(1.0 + torch.sigmoid(input)))


class Smish(nn.Module):
    """Smish activation as a nn.Module wrapper."""

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return smish(input)


# Alias for direct function call (used by DoubleFusion in original TED)
Fsmish = smish


# ============================================================================
# 3. TEED MODEL ARCHITECTURE (original TED from ted.py)
# ============================================================================

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class DoubleFusion(nn.Module):
    """TED fusion before the final edge map prediction (DWconv + PixelShuffle)."""

    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, kernel_size=3,
                                 stride=1, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24 * 1, kernel_size=3,
                                 stride=1, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, x):
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return Fsmish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True))
        self.add_module('smish1', Smish())
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(Fsmish(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16
        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        if self.use_ac:
            self.smish = Smish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None, stride=1, use_act=True):
        super(DoubleConvBlock, self).__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = Smish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class TED(nn.Module):
    """Tiny and Efficient Edge Detector (original architecture from ted.py)."""

    def __init__(self):
        super(TED, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # skip1 connection, see fig. 2
        self.side_1 = SingleConvBlock(16, 32, 2)

        # skip2 connection, see fig. 2
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        self.block_cat = DoubleFusion(3, 3)

        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        img_h, img_w = slice_shape
        if img_w != t_shape[-1] or img_h != t_shape[2]:
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    def resize_input(self, tensor):
        t_shape = tensor.shape
        if t_shape[2] % 8 != 0 or t_shape[3] % 8 != 0:
            img_w = ((t_shape[3] // 8) + 1) * 8
            img_h = ((t_shape[2] // 8) + 1) * 8
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    @staticmethod
    def crop_bdcn(data1, h, w, crop_h, crop_w):
        _, _, h1, w1 = data1.size()
        assert (h <= h1 and w <= w1)
        data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
        return data

    def forward(self, x, single_test=False):
        assert x.ndim == 4, x.shape

        original_shape = x.shape[2:]
        x = self.resize_input(x)

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        results = [out_1, out_2, out_3]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)

        results.append(block_cat)

        # Crop back to original resolution
        results = [self.slice(r, original_shape) for r in results]
        return results


# ============================================================================
# 4. LOSS FUNCTIONS
# ============================================================================

def bdcn_loss2(inputs, targets, l_weight=1.1):
    """BDCN loss v2: weighted binary cross-entropy + dice loss."""
    mask_bdr = -1.0 * targets * (targets - 2.0)
    mask_bdr = mask_bdr.float()
    mask_bdr_expand = mask_bdr.repeat(1, inputs.shape[1], 1, 1)

    weights_bdr = target_edge_weight(targets, l_weight)

    targets = targets.float()
    bce = F.binary_cross_entropy(inputs, targets, reduction='none')
    bce = bce * weights_bdr * mask_bdr_expand
    loss_bce = bce.sum() / (mask_bdr_expand.sum() + 1e-8)

    a = (inputs * mask_bdr_expand).sum()
    b = (targets * mask_bdr_expand).sum()
    dice = (2.0 * a + 1e-8) / (a + b + 1e-8)
    loss_dice = 1.0 - dice

    return loss_bce + loss_dice


def target_edge_weight(targets, l_weight=1.1):
    """Edge-aware weight map."""
    pos = torch.eq(targets, 1.0).float()
    neg = 1.0 - pos
    weight_map = pos * l_weight + neg * 1.0
    return weight_map


def cross_entropy2d(inputs, targets, reduction='mean'):
    """2D cross-entropy loss."""
    batch_size, _, h, w = inputs.shape
    log_p = F.log_softmax(inputs, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
    mask = (targets >= 0).float()
    targets = (targets * mask).long()
    targets = targets.view(-1)
    loss = F.nll_loss(log_p, targets, reduction='none', ignore_index=-1)
    loss = loss.view(batch_size, h, w)
    mask = mask.view(batch_size, h, w)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss


def cats_loss(prediction, targets, l_weight, device):
    """Cats loss: texture + boundary weighted."""
    tex_factor, bdr_factor = l_weight
    mask_tex = torch.lt(targets, 0.01).float()
    loss_tex = F.binary_cross_entropy(prediction, targets, reduction='none')
    loss_tex = (loss_tex * mask_tex).sum() / (mask_tex.sum() + 1e-8) * tex_factor

    mask_bdr = torch.gt(targets, 0.01).float()
    loss_bdr = F.binary_cross_entropy(prediction, targets, reduction='none')
    loss_bdr = (loss_bdr * mask_bdr).sum() / (mask_bdr.sum() + 1e-8) * bdr_factor

    return loss_tex + loss_bdr


# ============================================================================
# 5. DATASET
# ============================================================================

class EdgeDataset(Dataset):
    """
    Auto-detecting edge detection dataset.
    Expects:
        data/raw/    - source images
        data/mask/   - edge maps
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

    def __init__(self, data_dir: str, config: TEEDConfig, train_mode: bool = True):
        self.cfg = config
        self.train_mode = train_mode

        raw_dir = os.path.join(data_dir, 'raw')
        mask_dir = os.path.join(data_dir, 'mask')

        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"raw directory not found: {raw_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"mask directory not found: {mask_dir}")

        raw_files = [f for f in os.listdir(raw_dir)
                     if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS]
        mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)
                      if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS}

        self.samples = []
        for rf in raw_files:
            stem = os.path.splitext(rf)[0]
            if stem in mask_files:
                self.samples.append((
                    os.path.join(raw_dir, rf),
                    os.path.join(mask_dir, mask_files[stem]),
                    os.path.splitext(rf)[0]
                ))
            else:
                print(f"[WARN] No matching mask for {rf}, skipping.")

        if len(self.samples) == 0:
            raise RuntimeError(f"No image-mask pairs found in {data_dir}")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, stem = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        if mask.max() > 1.0:
            mask /= 255.0

        if self.train_mode:
            img, mask = self._augment(img, mask)

        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        mask = cv2.resize(mask, (self.cfg.img_size, self.cfg.img_size))

        mean_bgr = np.array(self.cfg.mean_pixels, dtype=np.float32).reshape(1, 1, 3)
        img -= mean_bgr

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        mask = torch.from_numpy(mask[np.newaxis, ...].copy()).float()

        return dict(images=img, labels=mask, file_name=stem + '.png')

    def _augment(self, img, mask):
        h, w = mask.shape

        if self.cfg.aug_crop_prob > 0 and random.random() < self.cfg.aug_crop_prob:
            min_sz = min(self.cfg.aug_crop_min_size, h, w)
            crop_h = random.randint(min_sz, h)
            crop_w = random.randint(min_sz, w)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            img = img[top:top + crop_h, left:left + crop_w]
            mask = mask[top:top + crop_h, left:left + crop_w]

        if self.cfg.aug_rotation > 0 and random.random() < 0.5:
            angle = random.uniform(-self.cfg.aug_rotation, self.cfg.aug_rotation)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        if self.cfg.aug_hflip and random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if self.cfg.aug_vflip and random.random() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        if self.cfg.aug_brightness > 0 or self.cfg.aug_contrast > 0:
            img = self._color_jitter(img)

        if self.cfg.aug_edge_boost > 0:
            mask[mask > 0.1] += self.cfg.aug_edge_boost
            mask = np.clip(mask, 0.0, 1.0)

        return img, mask

    def _color_jitter(self, img):
        delta_b = random.uniform(-self.cfg.aug_brightness * 255,
                                 self.cfg.aug_brightness * 255)
        img = img + delta_b

        alpha_c = 1.0 + random.uniform(-self.cfg.aug_contrast, self.cfg.aug_contrast)
        mean_img = img.mean()
        img = (img - mean_img) * alpha_c + mean_img

        if self.cfg.aug_saturation > 0 or self.cfg.aug_hue > 0:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= (1.0 + random.uniform(-self.cfg.aug_saturation, self.cfg.aug_saturation))
            hsv[:, :, 2] *= (1.0 + random.uniform(-self.cfg.aug_saturation, self.cfg.aug_saturation))
            hsv[:, :, 0] += random.uniform(-self.cfg.aug_hue * 179, self.cfg.aug_hue * 179)
            hsv[:, :, 0] %= 180.0
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

        return img


# ============================================================================
# 6. UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    img = np.float32(img)
    mn, mx = np.min(img), np.max(img)
    img = (img - mn) * (img_max - img_min) / (mx - mn + epsilon) + img_min
    return img


def save_edge_map(tensor, output_dir, file_name, original_shape=None):
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.sigmoid().cpu().detach().numpy()
    img = np.squeeze(tensor)
    img = np.uint8(image_normalization(img))
    img = cv2.bitwise_not(img)
    if original_shape is not None:
        img = cv2.resize(img, (original_shape[1], original_shape[0]))
    cv2.imwrite(os.path.join(output_dir, file_name), img)


def save_visualization(epoch, batch_id, total_batches, loss_val, images, labels, preds_list, cfg):
    viz_dir = os.path.join(cfg.checkpoint_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    h, w = cfg.img_size, cfg.img_size
    pad = 5
    n_cols = 2 + len(preds_list)
    canvas = np.zeros((h * 2 + pad, w * (n_cols // 2) + pad * (n_cols // 2 - 1), 3), dtype=np.uint8)

    img_np = images[0].cpu().numpy().transpose(1, 2, 0)
    img_np += np.array(cfg.mean_pixels, dtype=np.float32).reshape(1, 1, 3)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    gt_np = (labels[0, 0].cpu().numpy() * 255).astype(np.uint8)

    items = [img_np, gt_np]
    for p in preds_list:
        pm = p[0, 0].sigmoid().cpu().detach().numpy()
        pm = np.uint8(image_normalization(pm))
        pm = cv2.bitwise_not(pm)
        pm = cv2.cvtColor(pm, cv2.COLOR_GRAY2BGR)
        items.append(pm)

    col = 0
    row = 0
    for idx, item in enumerate(items):
        if idx == n_cols // 2:
            row = 1
            col = 0
        y = row * (h + pad)
        x = col * (w + pad)
        if len(item.shape) == 2:
            item = cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)
        item = cv2.resize(item, (w, h))
        canvas[y:y + h, x:x + w] = item
        col += 1

    filename = f'epoch_{epoch:03d}_batch_{batch_id:05d}_loss_{loss_val:.4f}.png'
    cv2.imwrite(os.path.join(viz_dir, filename), canvas)


# ============================================================================
# 7. TRAINING
# ============================================================================

def create_optimizer(model, cfg):
    return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)


def create_scheduler(optimizer, cfg, steps_per_epoch):
    name = cfg.lr_scheduler.lower()

    if name == "step":
        if len(cfg.lr_milestones) > 0:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[m * steps_per_epoch for m in cfg.lr_milestones],
                gamma=cfg.lr_gamma)
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=steps_per_epoch, gamma=cfg.lr_gamma)

    elif name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs * steps_per_epoch, eta_min=cfg.lr_min)

    elif name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.plateau_factor,
            patience=cfg.plateau_patience, min_lr=cfg.lr_min)

    elif name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr,
            total_steps=cfg.epochs * steps_per_epoch,
            pct_start=cfg.onecycle_pct_start,
            anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0)

    else:
        raise ValueError(f"Unknown scheduler: {name}")

    return scheduler


def train_one_epoch(epoch, dataloader, model, criterions, optimizer, scheduler,
                    device, scaler, cfg):
    model.train()
    criterion1, criterion2 = criterions
    loss_sum = 0.0
    loss_count = 0

    l_weight0 = cfg.loss_bdcn_weights
    l_weight_last = cfg.loss_cats_weights

    for batch_id, sample in enumerate(dataloader):
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            preds_list = model(images)
            loss1 = sum(criterion2(preds, labels, lw)
                        for preds, lw in zip(preds_list, l_weight0))
            loss2 = criterion1(preds_list[-1], labels, l_weight_last, device)
            loss = loss1 + loss2

        optimizer.zero_grad()
        if cfg.fp16:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if cfg.lr_scheduler in ("step", "cosine", "onecycle"):
            scheduler.step()

        loss_sum += loss.item()
        loss_count += 1

        if batch_id % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [Epoch {epoch:3d}] Batch {batch_id:4d}/{len(dataloader):4d}  "
                  f"Loss: {loss.item():.4f}  LR: {lr:.2e}")

        if batch_id % cfg.viz_interval == 0 and batch_id > 0:
            save_visualization(epoch, batch_id, len(dataloader), loss.item(),
                               images[:1], labels[:1], preds_list, cfg)

    return loss_sum / max(loss_count, 1)


@torch.no_grad()
def validate_one_epoch(epoch, dataloader, model, device, cfg):
    model.eval()
    criterion1, criterion2 = create_losses()
    loss_sum = 0.0
    loss_count = 0

    result_dir = os.path.join(cfg.checkpoint_dir, 'val_results', f'epoch_{epoch:03d}')
    os.makedirs(result_dir, exist_ok=True)

    l_weight0 = cfg.loss_bdcn_weights
    l_weight_last = cfg.loss_cats_weights

    for batch_id, sample in enumerate(dataloader):
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)
        file_names = sample['file_name']

        preds_list = model(images)
        loss1 = sum(criterion2(preds, labels, lw)
                    for preds, lw in zip(preds_list, l_weight0))
        loss2 = criterion1(preds_list[-1], labels, l_weight_last, device)
        loss = loss1 + loss2

        loss_sum += loss.item()
        loss_count += 1

        if batch_id < 5:
            save_edge_map(preds_list[-1][0], result_dir, file_names[0])

    return loss_sum / max(loss_count, 1)


def create_losses():
    return cats_loss, bdcn_loss2


def train(cfg: TEEDConfig):
    """Main training function."""
    print("=" * 60)
    print("TEED Training")
    print("=" * 60)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    full_dataset = EdgeDataset(cfg.data_dir, cfg, train_mode=True)
    n_val = max(1, int(len(full_dataset) * cfg.val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val],
                                              generator=torch.Generator().manual_seed(cfg.seed))

    class SubsetWithTransform:
        def __init__(self, full_ds, indices, train_mode):
            self.full_ds = full_ds
            self.indices = indices
            self.train_mode = train_mode

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            old_mode = self.full_ds.train_mode
            self.full_ds.train_mode = self.train_mode
            item = self.full_ds[self.indices[idx]]
            self.full_ds.train_mode = old_mode
            return item

    train_ds = SubsetWithTransform(full_dataset, train_dataset.indices, True)
    val_ds = SubsetWithTransform(full_dataset, val_dataset.indices, False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = TED().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
    criterions = create_losses()

    start_epoch = 0
    best_val_loss = float('inf')
    early_stop_counter = 0

    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        print(f"Loading checkpoint: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device)
        # Detect checkpoint format:
        #   Full checkpoint: dict with 'model_state_dict' key
        #   Bare state_dict: OrderedDict of weight names (e.g. original TEED .pth)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Full training checkpoint
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f"  Full checkpoint. Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            # Bare state_dict (pretrained weights only)
            model.load_state_dict(ckpt)
            print("  Pretrained weights loaded (starting from epoch 0).")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n{'─' * 50}")
        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        print(f"{'─' * 50}")

        train_loss = train_one_epoch(epoch, train_loader, model, criterions,
                                     optimizer, scheduler, device, scaler, cfg)

        if cfg.val_interval > 0 and (epoch + 1) % cfg.val_interval == 0:
            val_loss = validate_one_epoch(epoch, val_loader, model, device, cfg)
            print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

            if cfg.lr_scheduler == "plateau":
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': cfg,
                }, os.path.join(cfg.checkpoint_dir, 'best_model.pth'))
                print(f"  >>> Best model saved (val_loss: {val_loss:.4f})")
            else:
                early_stop_counter += 1
        else:
            print(f"  Train Loss: {train_loss:.4f}")

        if cfg.save_interval > 0 and (epoch + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': cfg,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        if cfg.early_stopping_patience > 0 and early_stop_counter >= cfg.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    final_path = os.path.join(cfg.checkpoint_dir, 'final_model.pth')
    torch.save({
        'epoch': cfg.epochs,
        'model_state_dict': model.state_dict(),
        'config': cfg,
    }, final_path)
    print(f"\nTraining complete. Final model saved: {final_path}")

    if cfg.onnx_export:
        best_path = os.path.join(cfg.checkpoint_dir, 'best_model.pth')
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
        export_onnx(model, cfg)


# ============================================================================
# 8. INFERENCE
# ============================================================================

def load_model_for_inference(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TED().to(device)

    # Detect checkpoint format
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        # Full training checkpoint
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # Bare state_dict (pretrained weights)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if missing_keys:
            print(f"  [WARN] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  [WARN] Unexpected keys: {unexpected_keys}")

    model.eval()
    return model


def infer_single_image(model, image_path, output_dir, device, cfg):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    original_shape = img.shape[:2]
    file_stem = os.path.splitext(os.path.basename(image_path))[0]

    img = img.astype(np.float32)
    mean_bgr = np.array(cfg.mean_pixels, dtype=np.float32).reshape(1, 1, 3)
    img -= mean_bgr
    img_t = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float().unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_t, single_test=True)

    save_edge_map(preds[-1][0], output_dir, f'{file_stem}.png', original_shape)

    if cfg.predict_all_outputs:
        for i, p in enumerate(preds):
            save_edge_map(p[0], os.path.join(output_dir, 'all_edges'),
                          f'{file_stem}_o{i + 1}.png', original_shape)

    print(f"Result saved: {os.path.join(output_dir, file_stem + '.png')}")


def infer_directory(model, image_dir, output_dir, device, cfg):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in exts]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images.")
    for fname in image_files:
        infer_single_image(model, os.path.join(image_dir, fname),
                           output_dir, device, cfg)


# ============================================================================
# 9. ONNX EXPORT
# ============================================================================

def export_onnx(model, cfg: TEEDConfig):
    onnx_path = os.path.join(cfg.checkpoint_dir, 'model.onnx')
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(*cfg.onnx_input_size)

    with torch.no_grad():
        outputs = model(dummy_input)
    output_names = [f'o{i + 1}' for i in range(len(outputs))]

    dynamic_axes = None
    if cfg.onnx_dynamic_axes:
        dynamic_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch', 2: 'height', 3: 'width'}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=cfg.onnx_opset,
        do_constant_folding=True,
    )

    print(f"ONNX model exported: {onnx_path}")

    if cfg.onnx_simplify:
        try:
            import onnx
            from onnxsim import simplify
            onnx_model = onnx.load(onnx_path)
            model_simp, check = simplify(onnx_model)
            if check:
                onnx.save(model_simp, onnx_path)
                print("  ONNX model simplified successfully.")
            else:
                print("  [WARN] ONNX simplification check failed.")
        except ImportError:
            print("  [WARN] onnxsim not installed. Install: pip install onnxsim onnx")


# ============================================================================
# 10. MODEL INFO
# ============================================================================

def print_model_info(cfg):
    print("=" * 60)
    print("TEED Model Information")
    print("=" * 60)

    model = TED()
    params = count_parameters(model)
    print(f"Total trainable parameters: {params:,}")

    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    model.eval()
    with torch.no_grad():
        outputs = model(dummy)

    print(f"\nInput shape:  (1, 3, {cfg.img_size}, {cfg.img_size})")
    for i, o in enumerate(outputs):
        print(f"  Output {i + 1}:  {list(o.shape)}")

    print(f"\nMemory estimate: {params * 4 / (1024 ** 2):.1f} MB (fp32)")
    print(f"Config image size: {cfg.img_size}x{cfg.img_size}")

    print(f"\n{'─' * 40}")
    print("Current Configuration:")
    print(f"{'─' * 40}")
    for field_name in TEEDConfig.__dataclass_fields__:
        value = getattr(cfg, field_name)
        print(f"  {field_name}: {value}")


# ============================================================================
# 11. MAIN ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='TEED: Tiny and Efficient Edge Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python teed_clean.py --mode train
  python teed_clean.py --mode train --epochs 20 --lr 1e-3 --img_size 480
  python teed_clean.py --mode train --resume_from ./checkpoints/epoch_005.pth
  python teed_clean.py --mode infer --image_path ./test.jpg --checkpoint ./checkpoints/best_model.pth
  python teed_clean.py --mode infer --image_dir ./test_images --checkpoint ./checkpoints/best_model.pth
  python teed_clean.py --mode export_onnx --checkpoint ./checkpoints/best_model.pth
  python teed_clean.py --mode info
        """)

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'infer', 'export_onnx', 'info'],
                        help='Operation mode')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1021)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--grad_clip', type=float, default=0.0)

    parser.add_argument('--lr_scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'onecycle'])
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_warmup_epochs', type=int, default=0)

    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--viz_interval', type=int, default=200)
    parser.add_argument('--early_stopping_patience', type=int, default=0)
    parser.add_argument('--resume_from', type=str, default='')

    parser.add_argument('--aug_rotation', type=float, default=15.0)
    parser.add_argument('--aug_hflip', action='store_true', default=True)
    parser.add_argument('--aug_vflip', action='store_true', default=True)
    parser.add_argument('--aug_brightness', type=float, default=0.2)
    parser.add_argument('--aug_contrast', type=float, default=0.2)

    parser.add_argument('--onnx_export', action='store_true', default=True)

    parser.add_argument('--image_path', type=str, default='')
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--predict_all_outputs', action='store_true', default=False)

    args = parser.parse_args()

    cfg = TEEDConfig()
    for key, value in vars(args).items():
        if key in TEEDConfig.__dataclass_fields__:
            setattr(cfg, key, value)

    return args, cfg


def main():
    args, cfg = parse_args()

    if args.mode == 'info':
        print_model_info(cfg)
        return

    if args.mode == 'train':
        train(cfg)

    elif args.mode == 'infer':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_for_inference(args.checkpoint, device)

        if args.image_path:
            infer_single_image(model, args.image_path, cfg.result_dir, device, cfg)
        elif args.image_dir:
            infer_directory(model, args.image_dir, cfg.result_dir, device, cfg)
        else:
            print("Error: --image_path or --image_dir required for infer mode.")
            sys.exit(1)

    elif args.mode == 'export_onnx':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_for_inference(args.checkpoint, device)
        export_onnx(model, cfg)


if __name__ == '__main__':
    # ── IDE mode: no command-line args → use IDE_RUN_SETTINGS ──
    if len(sys.argv) <= 1:
        print("=" * 60)
        print("TEED - Running in IDE mode (no CLI arguments detected)")
        print("  Modify IDE_MODE and related variables at top of this file")
        print(f"  Current IDE_MODE: {IDE_MODE}")
        print("=" * 60)

        sys.argv.append("--mode")
        sys.argv.append(IDE_MODE)

        if IDE_MODE == "train":
            sys.argv.extend(["--epochs", str(IDE_EPOCHS)])
            sys.argv.extend(["--lr", str(IDE_LR)])
            sys.argv.extend(["--img_size", str(IDE_IMG_SIZE)])
            sys.argv.extend(["--batch_size", str(IDE_BATCH_SIZE)])
            sys.argv.extend(["--checkpoint_dir", "./checkpoints"])
            sys.argv.extend(["--result_dir", IDE_RESULT_DIR])

        elif IDE_MODE == "infer":
            sys.argv.extend(["--checkpoint", IDE_CHECKPOINT])
            sys.argv.extend(["--result_dir", IDE_RESULT_DIR])
            sys.argv.append("--onnx_export")
            sys.argv.append("False")
            if IDE_IMAGE_PATH:
                sys.argv.extend(["--image_path", IDE_IMAGE_PATH])
            if IDE_IMAGE_DIR:
                sys.argv.extend(["--image_dir", IDE_IMAGE_DIR])

        elif IDE_MODE == "export_onnx":
            sys.argv.extend(["--checkpoint", IDE_CHECKPOINT])
            sys.argv.extend(["--checkpoint_dir", "./checkpoints"])

        elif IDE_MODE == "info":
            pass

    main()
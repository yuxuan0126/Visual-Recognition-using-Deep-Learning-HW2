"""
hw2.py — DETR Digit Detection
- ResNet-50 backbone: pretrained weights only
- Encoder/Decoder: trained from scratch
- Optimized for NVIDIA RTX 4090 (24GB VRAM)
- category_id: 1~10 (maps to digits 0~9)

Usage:
  train:   python hw2.py
  eval:    RUN_MODE=eval python hw2.py
  predict: python soup.py
"""

import json
import math
import os
import random
import time
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path

import albumentations as A
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

# ==========================================
# 1. Config
# ==========================================
TRAIN_IMG_DIR = os.environ.get("TRAIN_IMG_DIR", "./data/train")
TRAIN_JSON = os.environ.get("TRAIN_JSON", "./data/train.json")
VAL_IMG_DIR = os.environ.get("VAL_IMG_DIR", "./data/valid")
VAL_JSON = os.environ.get("VAL_JSON", "./data/valid.json")
TEST_IMG_DIR = os.environ.get("TEST_IMG_DIR", "./data/test")
TEST_JSON = os.environ.get("TEST_JSON", "./data/test.json")

RUN_MODE = os.environ.get("RUN_MODE", "train").lower()
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./best_model_hw2")
OUTPUT_JSON = os.environ.get("OUTPUT_JSON", "pred.json")

SEED = int(os.environ.get("SEED", "42"))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
DEBUG_NUM_IMGS = int(os.environ.get("DEBUG_NUM_IMAGES", "64"))

NUM_CLASSES = 10  # digits 0-9; category_id = label_index + 1

# ── Training hyperparams ──────────────────────────────────────────────────────
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "70"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))  
ACCUMULATION_STEPS = int(os.environ.get("ACCUMULATION_STEPS", "3"))  # effective=12
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "2"))

LR_TRANSFORMER = float(os.environ.get("LR_TRANSFORMER", "1e-4"))
LR_BACKBONE = float(os.environ.get("LR_BACKBONE", "1e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "5"))
FREEZE_BACKBONE = int(os.environ.get("FREEZE_BACKBONE_EPOCHS", "5"))

# ── Model architecture ────────────────────────────────────────────────────────
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", "30"))  # auto-computed below
ENCODER_LAYERS = int(os.environ.get("ENCODER_LAYERS", "6"))
DECODER_LAYERS = int(os.environ.get("DECODER_LAYERS", "3"))
USE_DILATION = os.environ.get("USE_DILATION", "1") == "1"
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.9997"))

# ── Inference ─────────────────────────────────────────────────────────────────
PRED_THRESHOLD = float(os.environ.get("PRED_THRESHOLD", "0.05"))
PRED_TOP_K = int(os.environ.get("PRED_TOP_K", "30"))

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

if DEBUG_MODE:
    VAL_EVERY = 1


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


seed_everything(SEED)


# ── Auto-infer NUM_QUERIES from annotation ────────────────────────────────────
def infer_num_queries(annot_file, fallback=30):
    if not os.path.exists(annot_file):
        return fallback
    with open(annot_file, encoding="utf-8") as f:
        coco = json.load(f)
    counts = {}
    for ann in coco["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    if not counts:
        return fallback
    vals = sorted(counts.values())
    p99 = vals[min(len(vals) - 1, int(0.99 * (len(vals) - 1)))]
    return max(10, min(60, p99 + 5))


NUM_QUERIES = infer_num_queries(TRAIN_JSON, NUM_QUERIES)
print(
    f"[Config] NUM_QUERIES={NUM_QUERIES} | enc={ENCODER_LAYERS} dec={DECODER_LAYERS} "
    f"| dilation={USE_DILATION} | batch={BATCH_SIZE}×acc{ACCUMULATION_STEPS}"
)


# ==========================================
# 2. Image processor & transforms
# ==========================================
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50",
    size={"shortest_edge": 600, "longest_edge": 1000},
)

train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.0),  # digits are not symmetric – skip
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
            rotate=(-5, 5),
            shear=(-3, 3),
            p=0.4,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(2, 8),
            hole_width_range=(2, 8),
            fill=0,
            p=0.25,
        ),
        A.RandomShadow(p=0.1),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_visibility=0.15,
        min_area=1.0,
        clip=True,
    ),
)

ACTIVE_TRANSFORM = None if DEBUG_MODE else train_transform


# ==========================================
# 3. Dataset
# ==========================================
class DigitDataset(Dataset):
    def __init__(self, img_dir, annot_file, proc, transforms=None, max_images=None):
        self.img_dir = img_dir
        self.proc = proc
        self.transforms = transforms

        with open(annot_file, encoding="utf-8") as f:
            coco = json.load(f)

        self.images = sorted(coco["images"], key=lambda x: x["id"])
        if max_images:
            self.images = self.images[:max_images]

        self.image_map = {img["id"]: img for img in self.images}
        valid_ids = set(self.image_map)

        # category_id (1-based) → 0-based label
        cat_ids = sorted(c["id"] for c in coco["categories"])
        self.cat2label = {cid: i for i, cid in enumerate(cat_ids)}

        self.annotations = {}
        for ann in coco["annotations"]:
            if ann["image_id"] not in valid_ids:
                continue
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = [img["id"] for img in self.images]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        info = self.image_map[iid]
        image = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        img_np = np.array(image)

        anns = self.annotations.get(iid, [])
        bboxes = [a["bbox"] for a in anns]
        cat_ids = [self.cat2label[a["category_id"]] for a in anns]

        if self.transforms and bboxes:
            t = self.transforms(image=img_np, bboxes=bboxes, category_ids=cat_ids)
            img_np = t["image"]
            bboxes = list(t["bboxes"])
            cat_ids = list(t["category_ids"])

        target = {
            "image_id": iid,
            "annotations": [
                {
                    "bbox": list(b),
                    "category_id": int(c),
                    "area": float(b[2] * b[3]),
                    "iscrowd": 0,
                }
                for b, c in zip(bboxes, cat_ids)
            ],
        }

        enc = self.proc(
            images=Image.fromarray(img_np), annotations=target, return_tensors="pt"
        )
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": enc["labels"][0],
        }


class TestDataset(Dataset):
    def __init__(self, img_dir, proc, annot_file=None, max_images=None):
        self.img_dir = img_dir
        self.proc = proc

        if annot_file and os.path.exists(annot_file):
            with open(annot_file, encoding="utf-8") as f:
                coco = json.load(f)
            self.images = sorted(coco["images"], key=lambda x: x["id"])
        else:
            self.images = []
            for fn in sorted(os.listdir(img_dir)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                stem = Path(fn).stem
                try:
                    iid = int(stem)
                except ValueError:
                    iid = len(self.images)
                self.images.append({"id": iid, "file_name": fn})

        if max_images:
            self.images = self.images[:max_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        enc = self.proc(images=image, return_tensors="pt")
        return {
            "image_id": info["id"],
            "pixel_values": enc["pixel_values"].squeeze(0),
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.long),
        }


# ── Collate ───────────────────────────────────────────────────────────────────
def _pad_batch(pixel_values_list):
    max_h = max(pv.shape[1] for pv in pixel_values_list)
    max_w = max(pv.shape[2] for pv in pixel_values_list)
    pv_out, pm_out = [], []
    for pv in pixel_values_list:
        c, h, w = pv.shape
        p = torch.zeros((c, max_h, max_w), dtype=pv.dtype)
        p[:, :h, :w] = pv
        m = torch.zeros((max_h, max_w), dtype=torch.long)
        m[:h, :w] = 1
        pv_out.append(p)
        pm_out.append(m)
    return torch.stack(pv_out), torch.stack(pm_out)


def collate_fn(batch):
    pvs, pms = _pad_batch([b["pixel_values"] for b in batch])
    return {
        "pixel_values": pvs,
        "pixel_mask": pms,
        "labels": [b["labels"] for b in batch],
    }


def test_collate_fn(batch):
    pvs, pms = _pad_batch([b["pixel_values"] for b in batch])
    return {
        "pixel_values": pvs,
        "pixel_mask": pms,
        "image_ids": [b["image_id"] for b in batch],
        "orig_sizes": torch.stack([b["orig_size"] for b in batch]),
    }


# ── Box utils ─────────────────────────────────────────────────────────────────
def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1
    )


def denormalize_boxes(boxes, orig_size):
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    H, W = float(orig_size[0]), float(orig_size[1])
    scale = boxes.new_tensor([W, H, W, H])
    return box_cxcywh_to_xyxy(boxes) * scale


# ==========================================
# 4. Model — backbone-only pretrained
# ==========================================
id2label = {i: str(i) for i in range(NUM_CLASSES)}
label2id = {v: k for k, v in id2label.items()}

config = DetrConfig(
    backbone="resnet50",
    use_pretrained_backbone=True,  # backbone weights from torchvision
    dilation=USE_DILATION,
    num_labels=NUM_CLASSES,
    num_queries=NUM_QUERIES,
    encoder_layers=ENCODER_LAYERS,
    decoder_layers=DECODER_LAYERS,
    auxiliary_loss=True,
    # Loss weights — tuned for small digit detection
    eos_coefficient=0.02,
    bbox_loss_coefficient=5,
    giou_loss_coefficient=2,
    class_cost=1,
    bbox_cost=5,
    giou_cost=2,
    id2label=id2label,
    label2id=label2id,
)


# Build from scratch, then copy ONLY backbone weights from pretrained
def build_model_backbone_only():
    """
    Creates a DETR model with random encoder/decoder weights but
    loads pretrained ResNet-50 backbone weights from the HuggingFace checkpoint.
    This satisfies the rule: backbone pretrained, encoder-decoder from scratch.
    """
    # Step 1: build fresh model (all random)
    mdl = DetrForObjectDetection(config)

    # Step 2: load full pretrained DETR just to copy backbone state
    pretrained = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        ignore_mismatched_sizes=True,
    )

    # Step 3: copy only backbone parameters
    own_sd = mdl.state_dict()
    pre_sd = pretrained.state_dict()
    copied = 0
    for k, v in pre_sd.items():
        # backbone keys: model.backbone.*
        if (
            k.startswith("model.backbone.")
            and k in own_sd
            and own_sd[k].shape == v.shape
        ):
            own_sd[k] = v.clone()
            copied += 1

    mdl.load_state_dict(own_sd)
    del pretrained
    print(
        f"[Model] Copied {copied} backbone tensors from pretrained DETR. "
        f"Encoder/decoder initialised from scratch."
    )
    return mdl


# 使用你原本 Config 區域定義的變數名稱
model = build_model_backbone_only().to(DEVICE)
ema_model = build_model_backbone_only().to(DEVICE)

ema_model.load_state_dict(model.state_dict())
ema_model.eval()

for p in ema_model.parameters():
    p.requires_grad = False


def set_backbone_trainable(trainable: bool):
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = trainable


@torch.no_grad()
def update_ema():
    for ep, mp in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(EMA_DECAY).add_(mp.data, alpha=1.0 - EMA_DECAY)
    for eb, mb in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(mb)


# ── Optimizer ─────────────────────────────────────────────────────────────────
def make_optimizer():
    backbone_params = [
        p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
    ]
    non_backbone_params = [
        p
        for n, p in model.named_parameters()
        if "backbone" not in n and p.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {"params": non_backbone_params, "lr": LR_TRANSFORMER},
            {"params": backbone_params, "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


optimizer = make_optimizer()
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)


# ── LR scheduler: linear warmup + cosine decay ───────────────────────────────
def build_scheduler(opt, steps_per_epoch):
    total_steps = max(steps_per_epoch * NUM_EPOCHS, 1)
    warmup_steps = max(steps_per_epoch * WARMUP_EPOCHS, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ==========================================
# 5. DataLoaders
# ==========================================
_max = DEBUG_NUM_IMGS if DEBUG_MODE else None

train_dataset = DigitDataset(
    TRAIN_IMG_DIR, TRAIN_JSON, processor, transforms=ACTIVE_TRANSFORM, max_images=_max
)
val_dataset = DigitDataset(
    VAL_IMG_DIR, VAL_JSON, processor, transforms=None, max_images=_max
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=max(1, NUM_WORKERS // 2),
    pin_memory=PIN_MEMORY,
    persistent_workers=True,
    prefetch_factor=2,
)

print(
    f"[Data] train={len(train_dataset)} val={len(val_dataset)} | "
    f"device={DEVICE} | AMP={USE_AMP}"
)


# ==========================================
# 6. Validation
# ==========================================
def run_validation(eval_model, loader):
    eval_model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    total_preds = 0
    total_imgs = 0
    max_score = 0.0

    with torch.no_grad():
        for batch in loader:
            pvs = batch["pixel_values"].to(DEVICE, non_blocking=PIN_MEMORY)
            pms = batch["pixel_mask"].to(DEVICE, non_blocking=PIN_MEMORY)
            outs = eval_model(pixel_values=pvs, pixel_mask=pms)

            target_sizes = torch.stack(
                [lbl["orig_size"] for lbl in batch["labels"]]
            ).to(DEVICE)
            preds = processor.post_process_object_detection(
                outs, target_sizes=target_sizes, threshold=0.0
            )

            targets = [
                {
                    "boxes": denormalize_boxes(lbl["boxes"], lbl["orig_size"]).to(
                        DEVICE
                    ),
                    "labels": lbl["class_labels"].to(DEVICE),
                }
                for lbl in batch["labels"]
            ]

            total_preds += sum(len(p["scores"]) for p in preds)
            total_imgs += len(preds)
            for p in preds:
                if len(p["scores"]):
                    max_score = max(max_score, float(p["scores"].max()))

            metric.update(preds, targets)

    m = metric.compute()
    m["avg_preds"] = total_preds / max(total_imgs, 1)
    m["max_score"] = max_score
    return m


# ==========================================
# 7. Training curve helper
# ==========================================
def plot_curves(history: dict, out_dir: str):
    """
    history keys: epoch, train_loss, val_map, val_map50, lr
    Saves training_curve.png into out_dir.
    """
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_map = history["val_map"]
    val_map50 = history["val_map50"]
    val_epochs = history["val_epoch"]
    lrs = history["lr"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, color="#2196F3", linewidth=1.8, label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── mAP ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(
        val_epochs,
        val_map,
        color="#4CAF50",
        linewidth=1.8,
        marker="o",
        ms=4,
        label="mAP",
    )
    ax.plot(
        val_epochs,
        val_map50,
        color="#FF9800",
        linewidth=1.8,
        marker="s",
        ms=4,
        label="mAP@50",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── LR ────────────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, lrs, color="#9C27B0", linewidth=1.8, label="LR (transformer)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.tight_layout()
    save_path = os.path.join(out_dir, "training_curve.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Training curve saved: {save_path}")


# ==========================================
# 8. Train
# ==========================================
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    steps_per_epoch = max(math.ceil(len(train_loader) / ACCUMULATION_STEPS), 1)
    scheduler = build_scheduler(optimizer, steps_per_epoch)
    best_map = -1.0

    # ── history for curves ────────────────────────────────────────────────────
    history = {
        "epoch": [],
        "train_loss": [],
        "lr": [],
        "val_epoch": [],
        "val_map": [],
        "val_map50": [],
    }

    print(
        f"[Train] Starting | epochs={NUM_EPOCHS} | freeze_backbone={FREEZE_BACKBONE} epochs"
    )

    for epoch in range(NUM_EPOCHS):
        bb_trainable = epoch >= FREEZE_BACKBONE
        set_backbone_trainable(bb_trainable)

        if epoch == 0 or epoch == FREEZE_BACKBONE:
            print(
                f"[{time.strftime('%H:%M:%S')}] Backbone: {'TRAINABLE' if bb_trainable else 'FROZEN'}"
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        t0 = time.time()

        # 1. 把 pbar 換成一般的 enumerate，不要用 tqdm
        # pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step, batch in enumerate(train_loader):  # 改回一般的 enumerate
            pvs = batch["pixel_values"].to(DEVICE, non_blocking=PIN_MEMORY)
            pms = batch["pixel_mask"].to(DEVICE, non_blocking=PIN_MEMORY)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(pixel_values=pvs, pixel_mask=pms, labels=labels)
                loss = out.loss / ACCUMULATION_STEPS

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            should_step = ((step + 1) % ACCUMULATION_STEPS == 0) or (
                (step + 1) == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                update_ema()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            current_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss

            # 2. 這裡就會精準每 500 步才噴一行，不會有進度條在旁邊亂跳
            if step > 0 and step % 500 == 0:
                print(
                    f"  [{time.strftime('%H:%M:%S')}] ep={epoch+1} step={step}/{len(train_loader)} "
                    f"loss={current_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        # 每一輪結束印一次
        avg_loss = total_loss / max(len(train_loader), 1)
        mins = (time.time() - t0) / 60.0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"loss={avg_loss:.4f} | lr={cur_lr:.2e} | {mins:.1f}min",
            flush=True,
        )

        # record per-epoch stats
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["lr"].append(cur_lr)

        if (epoch + 1) % VAL_EVERY != 0 and (epoch + 1) != NUM_EPOCHS:
            plot_curves(history, CHECKPOINT_DIR)  # update curve even on skipped val
            continue

        m = run_validation(ema_model, val_loader)
        cur_map = float(m["map"].item())
        cur_map50 = float(m["map_50"].item())
        print(
            f"[{time.strftime('%H:%M:%S')}] Val mAP={cur_map:.4f} | mAP50={cur_map50:.4f} | "
            f"avg_preds={m['avg_preds']:.1f} | max_score={m['max_score']:.4f}"
        )

        history["val_epoch"].append(epoch + 1)
        history["val_map"].append(cur_map)
        history["val_map50"].append(cur_map50)

        if (epoch + 1) % 5 == 0:
            periodic_path = f"./checkpoints_hw2/epoch_{epoch+1}"
            os.makedirs(periodic_path, exist_ok=True)
            ema_model.save_pretrained(periodic_path)
            processor.save_pretrained(periodic_path)
            print(f"  → Periodic checkpoint saved to {periodic_path}")

        if cur_map > best_map:
            best_map = cur_map
            ema_model.save_pretrained(CHECKPOINT_DIR)
            processor.save_pretrained(CHECKPOINT_DIR)
            with open(
                os.path.join(CHECKPOINT_DIR, "meta.json"), "w", encoding="utf-8"
            ) as f:
                json.dump({"best_map": best_map, "epoch": epoch + 1}, f)
            print(f"  → New best! Saved to {CHECKPOINT_DIR}")

        plot_curves(history, CHECKPOINT_DIR)

    print(f"Training done. Best mAP = {best_map:.4f}")


# ==========================================
# 9. Predict
# ==========================================
def generate_predictions():
    infer_proc = DetrImageProcessor.from_pretrained(CHECKPOINT_DIR)
    infer_model = DetrForObjectDetection.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
    infer_model.eval()

    test_ds = TestDataset(
        TEST_IMG_DIR,
        infer_proc,
        annot_file=TEST_JSON if os.path.exists(TEST_JSON) else None,
        max_images=DEBUG_NUM_IMGS if DEBUG_MODE else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=max(1, NUM_WORKERS // 2),
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
    )

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            pvs = batch["pixel_values"].to(DEVICE, non_blocking=PIN_MEMORY)
            pms = batch["pixel_mask"].to(DEVICE, non_blocking=PIN_MEMORY)
            sizes = batch["orig_sizes"].to(DEVICE)

            outs = infer_model(pixel_values=pvs, pixel_mask=pms)
            results = infer_proc.post_process_object_detection(
                outs, target_sizes=sizes, threshold=PRED_THRESHOLD
            )

            for image_id, res in zip(batch["image_ids"], results):
                scores = res["scores"].cpu()
                labels = res["labels"].cpu()
                boxes = res["boxes"].cpu()

                if len(scores) > PRED_TOP_K:
                    keep = torch.argsort(scores, descending=True)[:PRED_TOP_K]
                    scores = scores[keep]
                    labels = labels[keep]
                    boxes = boxes[keep]

                for score, label, box in zip(scores, labels, boxes):
                    x1, y1, x2, y2 = box.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "bbox": [
                                round(float(x1), 3),
                                round(float(y1), 3),
                                round(float(w), 3),
                                round(float(h), 3),
                            ],
                            "score": round(float(score), 6),
                            # label is 0-indexed; category_id must be 1-indexed (1~10)
                            "category_id": int(label.item()) + 1,
                        }
                    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions → {OUTPUT_JSON}")


# ==========================================
# 10. Eval (checkpoint on val set)
# ==========================================
def evaluate_checkpoint():
    eval_model = DetrForObjectDetection.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
    m = run_validation(eval_model, val_loader)
    print(
        f"Eval | mAP={float(m['map'].item()):.4f} | mAP50={float(m['map_50'].item()):.4f} | "
        f"avg_preds={float(m['avg_preds']):.1f} | max_score={float(m['max_score']):.4f}"
    )


# ==========================================
# 11. Entry
# ==========================================
if __name__ == "__main__":
    if RUN_MODE == "train":
        train()
    elif RUN_MODE == "predict":
        generate_predictions()
    elif RUN_MODE == "eval":
        evaluate_checkpoint()
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}  (train/predict/eval)")

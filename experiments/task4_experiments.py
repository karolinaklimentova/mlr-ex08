"""
Task 4 - a controlled comparison of improvements on top of the Task 3 pipeline.

Everything here starts from the Task 3 code of ex08_student.ipynb: the LeafDataset
class, the LeafNetBN network, the geometric augmentation and the plain training
loop. Each variant changes exactly ONE thing against the fixed baseline, so that a
difference can be attributed to that change.

Usage:
    .venv/bin/python task4_experiments.py --only baseline            # one run, for a smoke test
    .venv/bin/python task4_experiments.py                            # the whole grid
    .venv/bin/python task4_experiments.py --seeds 0 1 2              # multi-seed rows
"""

import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# The fixed parts, copied from the Task 2 / Task 3 notebook cells.
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
CLASSES = ["apple", "cherry", "chestnut", "maple"]   # the sorted subfolder names

BATCH = 32
IMAGE_SIZE = 96

# The three evaluation sets. The test split is never used for tuning.
PRIMARY_TRAIN = os.path.join(ROOT, "data/leaves/train")
PRIMARY_VAL = os.path.join(ROOT, "data/leaves/val")
HIST_TRAIN = os.path.join(ROOT, "leaves/train")        # last year's photographs
HIST_VAL = os.path.join(ROOT, "leaves/val")

CAMERA_VAL_CACHE = os.path.join(ROOT, "task4_camera_val.pt")
RESULTS_PATH = os.path.join(ROOT, "task4_results.json")

# The plain validation transform (Task 2 cell 40).
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# The geometric augmentation of Task 3 cell 60 - this is the BASELINE augmentation.
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# One change: the same geometric augmentation plus a colour jitter.
aug_color_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ---------------------------------------------------------------------------
# The dataset of Task 3 cell 54, with the class order fixed to CLASSES so that the
# primary and the historical folder share one label mapping when they are combined.
# ---------------------------------------------------------------------------

class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        self.classes = CLASSES
        class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_folder = os.path.join(folder, class_name)
            for filename in sorted(os.listdir(class_folder)):
                path = os.path.join(class_folder, filename)
                if os.path.isfile(path) and datasets.folder.has_file_allowed_extension(
                    filename, datasets.folder.IMG_EXTENSIONS
                ):
                    self.samples.append((path, class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


# ---------------------------------------------------------------------------
# The historical dataset, cleaned once. See clean_historical_train() for the rules.
# ---------------------------------------------------------------------------

def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.sha1(handle.read()).hexdigest()


def scan(folder):
    """Every image of a train/val folder as a list of (path, class, hash)."""
    found = []
    for class_name in CLASSES:
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in sorted(os.listdir(class_folder)):
            path = os.path.join(class_folder, filename)
            if os.path.isfile(path) and filename.lower().endswith(
                datasets.folder.IMG_EXTENSIONS
            ):
                found.append((path, class_name, file_hash(path)))
    return found


def clean_historical_train():
    """
    The historical train folder is not clean, and the leakage matters because its
    val folder is our out-of-distribution benchmark. Three rules, applied in order:

      1. drop every image whose bytes also appear in the historical VAL folder -
         otherwise the benchmark contains training images;
      2. drop every image whose bytes appear under two DIFFERENT class folders -
         the label of such an image is unknowable;
      3. keep only the first copy of every remaining image, so that the species
         with duplicated photographs are not simply counted twice.

    Returns (kept_paths_by_class, report dict).
    """
    train_items = scan(HIST_TRAIN)
    val_items = scan(HIST_VAL)

    val_hashes = {h for _, _, h in val_items}
    hashes_to_classes = {}
    for _, class_name, h in train_items:
        hashes_to_classes.setdefault(h, set()).add(class_name)
    conflicting = {h for h, c in hashes_to_classes.items() if len(c) > 1}

    dropped_val_overlap, dropped_conflict, dropped_duplicate = [], [], []
    kept, seen = [], set()
    for path, class_name, h in train_items:
        if h in val_hashes:
            dropped_val_overlap.append(path)
            continue
        if h in conflicting:
            dropped_conflict.append(path)
            continue
        if h in seen:
            dropped_duplicate.append(path)
            continue
        seen.add(h)
        kept.append((path, class_name))

    # Write the kept files into a folder of symlinks, so that LeafDataset can read
    # them unchanged and the cleaning is inspectable on disk.
    out_root = os.path.join(ROOT, "task4_historical_clean")
    for class_name in CLASSES:
        os.makedirs(os.path.join(out_root, class_name), exist_ok=True)
        for existing in os.listdir(os.path.join(out_root, class_name)):
            os.remove(os.path.join(out_root, class_name, existing))
    for path, class_name in kept:
        link = os.path.join(out_root, class_name, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(os.path.relpath(path, os.path.dirname(link)), link)

    report = {
        "historical_train_files_found": len(train_items),
        "historical_train_unique_hashes": len(hashes_to_classes),
        "dropped_because_in_historical_val": dropped_val_overlap,
        "dropped_because_label_conflict": dropped_conflict,
        "dropped_because_exact_duplicate": dropped_duplicate,
        "kept": len(kept),
        "kept_per_class": {
            c: sum(1 for _, cn in kept if cn == c) for c in CLASSES
        },
        "historical_val_files": len(val_items),
    }
    return out_root, report


class FolderDataset(Dataset):
    """The same reading rules as LeafDataset, but over a flat prepared folder."""

    def __init__(self, folder, transform):
        self.transform = transform
        self.samples = []
        for index, class_name in enumerate(CLASSES):
            class_folder = os.path.join(folder, class_name)
            for filename in sorted(os.listdir(class_folder)):
                path = os.path.join(class_folder, filename)
                if os.path.isfile(path) or os.path.islink(path):
                    self.samples.append((path, index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


# ---------------------------------------------------------------------------
# The camera-like validation set: the validation images, randomly cropped, rotated
# and recoloured ONCE with a fixed seed, then frozen to disk and reused unchanged
# by every variant, so all variants face literally the same pixels.
# ---------------------------------------------------------------------------

camera_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.45, 0.9)),
    transforms.RandomRotation(35),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.12),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def materialize(folder):
    """Read a val folder once into frozen tensors, so that every variant is scored
    on literally the same pixels."""
    items = scan(folder)
    images = torch.stack([val_transform(Image.open(p).convert("RGB")) for p, _, _ in items])
    labels = torch.tensor([CLASSES.index(c) for _, c, _ in items])
    return images, labels, [p for p, _, _ in items]


def build_camera_val(views_per_image=3, seed=1234, force=False):
    """
    Every validation image photographed three times through the camera-like
    transform, drawn from one fixed seed. The untouched image is deliberately NOT
    included - that is what the primary validation set already measures, and mixing
    it in would make this benchmark look easier than a phone camera really is.
    """
    if os.path.exists(CAMERA_VAL_CACHE) and not force:
        blob = torch.load(CAMERA_VAL_CACHE)
        return blob["images"], blob["labels"], blob["sources"]

    ds = LeafDataset(PRIMARY_VAL, val_transform)
    images, labels, sources = [], [], []
    torch.manual_seed(seed)                     # the one seed that freezes this set
    for path, label in ds.samples:
        with Image.open(path) as raw:
            raw = raw.convert("RGB")
        for view in range(views_per_image):
            images.append(camera_transform(raw).unsqueeze(0))
            labels.append(label)
            sources.append(f"{path}#view{view}")
    images = torch.cat(images)
    labels = torch.tensor(labels)
    torch.save({"images": images, "labels": labels, "sources": sources}, CAMERA_VAL_CACHE)
    return images, labels, sources


# ---------------------------------------------------------------------------
# The networks. LeafNetBN is the Task 3 cell 60 network, unchanged.
# ---------------------------------------------------------------------------

class LeafNetBN(nn.Module):
    # LeafNet with a batch normalization after every convolution, before the ReLU.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class LeafNetBNGAP(nn.Module):
    """One change: global average pooling instead of flattening, plus a dropout
    before the classifier. Everything else is the network above, layer for layer."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.gap(x).flatten(1)
        x = torch.relu(self.dropout(x))
        return self.fc(x)


def build_mobilenet(freeze_backbone):
    """The Task 2 cell 42 network: a pretrained MobileNetV3 small with a new head."""
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not name.startswith("classifier.")
    return model


# ---------------------------------------------------------------------------
# The variants. Each entry differs from "baseline" in the commented way only.
# ---------------------------------------------------------------------------

def variant_configs():
    base = {
        "net": "leafnetbn",
        "data": "primary",
        "transform": "geo",
        "epochs": 15,
        "optimizer": "adam",
        "lr": 1e-3,
        "scheduler": None,
        "label_smoothing": 0.0,
        "keep_best": False,
    }

    def variant(**changes):
        cfg = dict(base)
        cfg.update(changes)
        return cfg

    return {
        # The fixed starting point: Task 3 cell 60 + cell 61 exactly.
        "baseline":              variant(),

        # One change each.
        "+color":                variant(transform="color"),
        "+transfer-frozen":      variant(net="mobilenet_frozen", epochs=15),
        "+transfer-full":        variant(net="mobilenet_full", epochs=15),
        "+regularized":          variant(net="leafnetbn_gap", optimizer="adamw",
                                         label_smoothing=0.1, keep_best=True),
        "+historical-data":      variant(data="primary+historical"),

        # Pretrained + augmentation + cosine schedule + more epochs: the recipe the
        # assignment suggests trying together, kept as its own row.
        "+transfer-aug-cosine":  variant(net="mobilenet_full", epochs=40,
                                         scheduler="cosine"),

        # The best combinations, filled in from the single-change results.
        "+transfer-cosine+hist": variant(net="mobilenet_full", data="primary+historical",
                                         epochs=40, scheduler="cosine"),
        "+transfer-cosine+hist+color": variant(
            net="mobilenet_full", data="primary+historical", transform="color",
            epochs=40, scheduler="cosine"),
        "+transfer-cosine+hist+color+adamw": variant(
            net="mobilenet_full", data="primary+historical", transform="color",
            epochs=40, scheduler="cosine", optimizer="adamw",
            label_smoothing=0.1, keep_best=True),
    }


# ---------------------------------------------------------------------------
# Training one variant once.
# ---------------------------------------------------------------------------

def build_loaders(cfg, seed):
    generator = torch.Generator().manual_seed(seed)
    transform = {"geo": aug_transform, "color": aug_color_transform}[cfg["transform"]]

    if cfg["data"] == "primary":
        train_ds = LeafDataset(PRIMARY_TRAIN, transform)
    else:
        primary = LeafDataset(PRIMARY_TRAIN, transform)
        historical = FolderDataset(os.path.join(ROOT, "task4_historical_clean"), transform)
        train_ds = ConcatDataset([primary, historical])

    val_ds = LeafDataset(PRIMARY_VAL, val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    return train_ds, val_ds, train_loader, val_loader


def build_net(cfg, seed):
    torch.manual_seed(seed)
    if cfg["net"] == "leafnetbn":
        model = LeafNetBN()
    elif cfg["net"] == "leafnetbn_gap":
        model = LeafNetBNGAP()
    elif cfg["net"] == "mobilenet_frozen":
        model = build_mobilenet(freeze_backbone=True)
    elif cfg["net"] == "mobilenet_full":
        model = build_mobilenet(freeze_backbone=False)
    else:
        raise ValueError(cfg["net"])
    return model.to(DEVICE)


@torch.no_grad()
def evaluate_on_tensor(model, images, labels, batch=128):
    model.eval()
    predictions = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch].to(DEVICE)
        predictions.append(model(chunk).argmax(dim=1).cpu())
    predictions = torch.cat(predictions)
    correct = int((predictions == labels).sum())
    return correct, len(labels), predictions.numpy()


@torch.no_grad()
def evaluate_on_loader(model, loader):
    model.eval()
    all_predictions, all_labels = [], []
    for images, labels in loader:
        all_predictions.append(model(images.to(DEVICE)).argmax(dim=1).cpu())
        all_labels.append(labels)
    return (torch.cat(all_predictions).numpy(), torch.cat(all_labels).numpy())


def run_variant(name, cfg, seed, primary_val, camera_val, hist_val, log=print,
                save_weights=False):
    started = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg, seed)
    model = build_net(cfg, seed)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(trainable, lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scheduler = None
    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_accuracy, best_state = -1.0, None
    curve = []
    for epoch in range(cfg["epochs"]):
        model.train()
        loss_sum, samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            samples += labels.size(0)
        if scheduler is not None:
            scheduler.step()

        # Validation is also the model-selection signal when keep_best is on.
        predictions, targets = evaluate_on_loader(model, val_loader)
        accuracy = float((predictions == targets).mean())
        curve.append({"epoch": epoch + 1, "loss": loss_sum / samples, "val_accuracy": accuracy})
        if cfg["keep_best"] and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
        log(f"    [{name} seed={seed}] epoch {epoch + 1:2d}/{cfg['epochs']} "
            f"loss {loss_sum / samples:.4f} val acc {accuracy:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # The three benchmarks, all of them validation data - never a test split.
    # Each set is a frozen tensor, so every variant is scored on identical pixels.
    hist_images, hist_labels, _ = hist_val
    cam_images, cam_labels, _ = camera_val
    primary_images, primary_labels, _ = primary_val
    prim_correct, prim_total, prim_pred = evaluate_on_tensor(
        model, primary_images, primary_labels)
    cam_correct, cam_total, cam_pred = evaluate_on_tensor(model, cam_images, cam_labels)
    hist_correct, hist_total, hist_pred = evaluate_on_tensor(model, hist_images, hist_labels)

    # Per-class recall on all three val sets, for the report's confusion discussion.
    primary_targets = primary_labels.numpy()
    cam_targets = cam_labels.numpy()
    hist_targets = hist_labels.numpy()
    primary_matrix = confusion_matrix(primary_targets, prim_pred, labels=list(range(len(CLASSES))))
    cam_matrix = confusion_matrix(cam_targets, cam_pred, labels=list(range(len(CLASSES))))
    hist_matrix = confusion_matrix(hist_targets, hist_pred, labels=list(range(len(CLASSES))))
    per_class_recall = {}
    for index, class_name in enumerate(CLASSES):
        primary_support = int(primary_matrix[index].sum())
        cam_support = int(cam_matrix[index].sum())
        hist_support = int(hist_matrix[index].sum())
        per_class_recall[class_name] = {
            "primary_recall": float(primary_matrix[index][index] / primary_support)
                              if primary_support else None,
            "primary_support": primary_support,
            "camera_recall": float(cam_matrix[index][index] / cam_support)
                             if cam_support else None,
            "camera_support": cam_support,
            "historical_recall": float(hist_matrix[index][index] / hist_support)
                                 if hist_support else None,
            "historical_support": hist_support,
        }
    # Keeping the exact weights that produced these numbers means the ONNX export
    # later ships this model, not a retraining that might land somewhere else.
    weights_path = None
    if save_weights:
        weights_path = os.path.join(ROOT, f"task4_weights_{name}_seed{seed}.pt")
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": cfg, "variant": name, "seed": seed,
                    "classes": CLASSES, "mean": mean, "std": std}, weights_path)

    return {
        "variant": name,
        "seed": seed,
        "weights_file": os.path.basename(weights_path) if weights_path else None,
        "config": {k: v for k, v in cfg.items()},
        "train_samples": len(train_ds),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "primary_val_accuracy": prim_correct / prim_total,
        "primary_val_balanced_accuracy": float(balanced_accuracy_score(primary_targets, prim_pred)),
        "camera_val_accuracy": cam_correct / cam_total,
        "camera_val_balanced_accuracy": float(balanced_accuracy_score(cam_targets, cam_pred)),
        "historical_val_accuracy": hist_correct / hist_total,
        "historical_val_balanced_accuracy": float(balanced_accuracy_score(hist_targets, hist_pred)),
        "historical_per_class": per_class_recall,
        "primary_confusion": primary_matrix.tolist(),
        "camera_confusion": cam_matrix.tolist(),
        "historical_confusion": hist_matrix.tolist(),
        "curve": curve,
        "seconds": round(time.time() - started, 1),
    }


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="variant names to run")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--rebuild-camera-val", action="store_true")
    parser.add_argument("--save-weights", action="store_true",
                        help="keep the trained weights of every run for the ONNX export")
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args()

    print(f"device {DEVICE} | torch {torch.__version__}")
    _, cleaning_report = clean_historical_train()
    print("historical cleaning:")
    print(f"  found {cleaning_report['historical_train_files_found']} files, "
          f"{cleaning_report['historical_train_unique_hashes']} unique images")
    print(f"  dropped {len(cleaning_report['dropped_because_in_historical_val'])} in historical val")
    print(f"  dropped {len(cleaning_report['dropped_because_label_conflict'])} with a conflicting label")
    print(f"  dropped {len(cleaning_report['dropped_because_exact_duplicate'])} exact duplicates")
    print(f"  kept {cleaning_report['kept']}: {cleaning_report['kept_per_class']}")
    with open(os.path.join(ROOT, "task4_cleaning_report.json"), "w") as handle:
        json.dump(cleaning_report, handle, indent=1)

    camera_val = build_camera_val(force=args.rebuild_camera_val)
    print(f"camera-like val frozen: {camera_val[0].shape[0]} views "
          f"of {len({s.split('#')[0] for s in camera_val[2]})} validation images")
    primary_val = materialize(PRIMARY_VAL)
    hist_val = materialize(HIST_VAL)
    print(f"primary val: {tuple(primary_val[0].shape)} | "
          f"historical val (out of distribution): {tuple(hist_val[0].shape)}")

    configs = variant_configs()
    names = args.only if args.only else list(configs)
    results = []
    if os.path.exists(args.out):
        with open(args.out) as handle:
            results = json.load(handle)
    done = {(r["variant"], r["seed"]): r for r in results}

    for name in names:
        for seed in args.seeds:
            previous = done.get((name, seed))
            # A run already in the file is skipped, unless we now also want its weights
            # saved and it was run without them.
            if previous is not None and not (
                args.save_weights and not previous.get("weights_file")
            ):
                print(f"  [{name} seed={seed}] already in {os.path.basename(args.out)}, skipping")
                continue
            print(f"\n=== {name} | seed {seed} | {configs[name]}")
            record = run_variant(name, configs[name], seed, primary_val, camera_val,
                                 hist_val, save_weights=args.save_weights)
            if previous is not None:
                results.remove(previous)
            results.append(record)
            print(f"  -> primary {record['primary_val_accuracy']:.3f} "
                  f"| camera {record['camera_val_accuracy']:.3f} "
                  f"| historical {record['historical_val_accuracy']:.3f} "
                  f"| {record['seconds']} s")
            with open(args.out, "w") as handle:
                json.dump(results, handle, indent=1)

    print(f"\nwrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()

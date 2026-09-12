"""MobileNetV3-Small CNN tamper classifier with Grad-CAM explainability."""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from loguru import logger

from ..config import TamperModelConfig
from ..forensics.ela_utils import compute_ela_map


def _resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FocalLoss(nn.Module):
    """Focal Loss — down-weights well-classified examples (Lin et al., ICCV 2017)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        alpha_t = self.alpha * targets_one_hot[:, 1] + (1 - self.alpha) * targets_one_hot[:, 0]
        alpha_t = alpha_t.sum() / max(targets.size(0), 1)  # scalar
        focal_weight = (1 - pt) ** self.gamma
        loss = -alpha_t * focal_weight * torch.log(pt + 1e-8)
        return loss.mean()


class TamperClassifier(nn.Module):
    """4-channel MobileNetV3-Small for tamper classification.

    Input: 3 RGB channels + 1 raw ELA grayscale channel. Pre-trained RGB
    weights are reused; the ELA channel is initialised as their mean.
    """

    def __init__(self, config: TamperModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or TamperModelConfig()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
            if self.config.pretrained
            else None
        )

        orig = backbone.features[0][0]
        new_conv = nn.Conv2d(
            4, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=orig.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = orig.weight
            new_conv.weight[:, 3:4] = orig.weight.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, self.config.num_classes)
        self.classifier = backbone.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """Single-sample prediction → (class_idx, confidence)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            cls = int(torch.argmax(probs, dim=1).item())
            return cls, float(probs[0, cls].item())




class GradCAM:
    """Grad-CAM for TamperClassifier (Selvaraju et al., ICCV 2017)."""

    def __init__(self, model: TamperClassifier) -> None:
        self.model = model
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Hook into the last conv block of MobileNetV3 features
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _input: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module: nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
        self._gradients = grad_out[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a single input tensor."""
        self.model.eval()
        x = input_tensor.unsqueeze(0).requires_grad_(True)

        logits = self.model(x)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Up-sample to input spatial size
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[1]))
        return cam.astype(np.float32)

    @staticmethod
    def overlay(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay a [0,1] heatmap on a BGR image using JET colourmap."""
        coloured = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if coloured.shape[:2] != image_bgr.shape[:2]:
            coloured = cv2.resize(coloured, (image_bgr.shape[1], image_bgr.shape[0]))
        return cv2.addWeighted(coloured, alpha, image_bgr, 1 - alpha, 0)




class TamperDataset(Dataset):
    """
    Loads authentic / tampered images and computes ELA on-the-fly.

    Directory layout::

        data_dir/
            authentic/  ← label 0
            tampered/   ← label 1
    """

    def __init__(
        self,
        data_dir: str,
        input_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augment = augment
        self.samples: List[Tuple[str, int]] = []

        for label, subdir in [(0, "authentic"), (1, "tampered")]:
            d = self.data_dir / subdir
            if d.exists():
                for p in d.glob("*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        self.samples.append((str(p), label))

        n_auth = sum(1 for _, l in self.samples if l == 0)
        n_tamp = sum(1 for _, l in self.samples if l == 1)
        logger.info(f"TamperDataset: {len(self.samples)} samples ({n_auth} auth, {n_tamp} tamp)")

        self.rgb_base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.rgb_augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # ELA channel uses no colour augmentation — just resize + normalise
        self.ela_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            # Single-channel normalisation (ImageNet luminance stats)
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros(4, *self.input_size), label

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Shared ELA computation — raw grayscale, no colourmap
        ela_map = compute_ela_map(image)  # uint8 grayscale

        tfm = self.rgb_augment_transform if self.augment else self.rgb_base_transform
        img_t = tfm(image_rgb)              # (3, H, W)
        ela_t = self.ela_transform(ela_map)  # (1, H, W)
        return torch.cat([img_t, ela_t], dim=0), label  # (4, H, W)


class TamperTrainer:
    """Training engine with Focal Loss, cosine LR schedule, and AMP."""

    def __init__(
        self,
        model: TamperClassifier,
        config: TamperModelConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or TamperModelConfig()
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_precision": [], "val_recall": [], "val_f1": [],
        }


    def train(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        val_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        full_ds = TamperDataset(train_dir, self.config.input_size, augment=True)

        if val_dir:
            val_ds: Dataset = TamperDataset(val_dir, self.config.input_size, augment=False)
            train_ds: Dataset = full_ds
        else:
            total = len(full_ds)
            val_n = int(total * val_split)
            train_ds, val_ds = torch.utils.data.random_split(full_ds, [total - val_n, val_n])

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2,
        )

        # AMP only benefits CUDA
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_val_f1 = 0.0
        patience_ctr = 0
        patience_limit = 7

        logger.info(
            f"Training on {self.device} for {self.config.epochs} epochs | "
            f"AMP: {use_amp} | Focal Loss (α=0.25, γ=2.0) | CosineAnnealingWarmRestarts"
        )

        for epoch in range(self.config.epochs):

            self.model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device, enabled=use_amp):
                    out = self.model(inputs)
                    loss = criterion(out, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                t_loss += loss.item() * inputs.size(0)
                t_correct += (out.argmax(1) == labels).sum().item()
                t_total += labels.size(0)

            scheduler.step()

            train_loss = t_loss / max(t_total, 1)
            train_acc = t_correct / max(t_total, 1)

            val_loss, val_acc, prec, rec, f1 = self._evaluate(val_loader, criterion)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_precision"].append(prec)
            self.history["val_recall"].append(rec)
            self.history["val_f1"].append(f1)

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} — "
                f"TrL: {train_loss:.4f} TrA: {train_acc:.4f} | "
                f"VL: {val_loss:.4f} VA: {val_acc:.4f} "
                f"P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f} | "
                f"LR: {current_lr:.2e}"
            )

            # Save best model by F1 (better than val_loss under class imbalance)
            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_ctr = 0
                self.save_model()
            else:
                patience_ctr += 1
                if patience_ctr >= patience_limit:
                    logger.info(f"Early stopping at epoch {epoch+1} (best F1: {best_val_f1:.3f})")
                    break

        return self.history

    def save_model(self, path: Optional[str] = None) -> None:
        p = path or self.config.model_save_path
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), p)
        logger.info(f"Model saved → {p}")

    def load_model(self, path: Optional[str] = None) -> None:
        p = path or self.config.model_save_path
        self.model.load_state_dict(torch.load(p, map_location=self.device))
        self.model.eval()


    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, float, float, float]:
        """Full evaluation pass → (loss, accuracy, precision, recall, F1)."""
        self.model.eval()
        total_loss, total = 0.0, 0
        tp = fp = fn = tn = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = self.model(inputs)
            total_loss += criterion(out, labels).item() * inputs.size(0)
            preds = out.argmax(1)
            total += labels.size(0)

            tp += int(((preds == 1) & (labels == 1)).sum().item())
            fp += int(((preds == 1) & (labels == 0)).sum().item())
            fn += int(((preds == 0) & (labels == 1)).sum().item())
            tn += int(((preds == 0) & (labels == 0)).sum().item())

        loss = total_loss / max(total, 1)
        acc = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return loss, acc, precision, recall, f1


class TamperClassifierInference:
    """High-level inference wrapper with optional Grad-CAM explanation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: TamperModelConfig | None = None,
    ) -> None:
        self.config = config or TamperModelConfig()
        self.device = _resolve_device()
        self.model = TamperClassifier(self.config).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Tamper classifier loaded from {model_path}")
        else:
            logger.warning("No model weights loaded — using random weights")
        self.model.eval()

        self._cam = GradCAM(self.model)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Separate transform for single-channel ELA — no colourmap
        self.ela_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ])


    def classify(self, image: np.ndarray) -> Dict:
        """Return ``{prediction, label, confidence}`` for a BGR image."""
        combined = self._prepare_input(image)
        cls, conf = self.model.predict(combined)
        return {
            "prediction": cls,
            "label": "tampered" if cls == 1 else "authentic",
            "confidence": round(conf, 4),
        }

    def explain(self, image: np.ndarray, target_class: int | None = None) -> np.ndarray:
        """Grad-CAM heatmap for the predicted (or specified) class."""
        combined = self._prepare_input(image)
        return self._cam.generate(combined, target_class=target_class)

    def explain_overlay(self, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Convenience: Grad-CAM heatmap overlaid on the original BGR image."""
        heatmap = self.explain(image)
        return GradCAM.overlay(image, heatmap, alpha=alpha)

    def benchmark_fps(self, image: np.ndarray, n: int = 50) -> float:
        """Run *n* forward passes and return throughput in FPS."""
        combined = self._prepare_input(image).unsqueeze(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for _ in range(5):
                self.model(combined)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                self.model(combined)
        elapsed = time.perf_counter() - t0
        fps = n / elapsed
        logger.info(f"Benchmark: {fps:.1f} FPS ({elapsed / n * 1000:.1f} ms/frame)")
        return fps

    def _prepare_input(self, image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ela_map = compute_ela_map(image)  # uint8 grayscale

        img_t = self.transform(rgb)
        ela_t = self.ela_transform(ela_map)
        return torch.cat([img_t, ela_t], dim=0).to(self.device)

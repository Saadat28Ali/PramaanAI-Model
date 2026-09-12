"""Synthetic degradation engine for generating augmented document images."""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from loguru import logger

try:
    import albumentations as A

    _HAS_ALBUMENTATIONS = True
except ImportError:  # pragma: no cover
    _HAS_ALBUMENTATIONS = False

from ..config import SyntheticConfig


@dataclass
class AugmentationRecord:
    """Record of what augmentations were applied to a specific image."""
    source_image: str
    output_image: str
    augmentations_applied: List[str]
    parameters: Dict




def add_glare(
    image: np.ndarray,
    intensity: float = 0.8,
    num_spots: int = 2,
    min_radius: int = 30,
    max_radius: int = 120,
) -> np.ndarray:
    """Add Gaussian-blob specular glare to simulate laminated card reflections."""
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]

    for _ in range(num_spots):
        # Random position and size
        cx = np.random.randint(w // 6, 5 * w // 6)
        cy = np.random.randint(h // 6, 5 * h // 6)
        radius = np.random.randint(min_radius, max_radius)

        y_coords, x_coords = np.ogrid[-cy : h - cy, -cx : w - cx]
        gaussian = np.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * radius ** 2))
        gaussian = gaussian * intensity * 255.0

        # Warm tint: slightly more red than blue
        result[:, :, 0] += gaussian * 0.9
        result[:, :, 1] += gaussian * 0.95
        result[:, :, 2] += gaussian * 1.0

    return np.clip(result, 0, 255).astype(np.uint8)


def add_perspective_warp(
    image: np.ndarray,
    max_shift_ratio: float = 0.15,
) -> np.ndarray:
    """Apply random perspective warping to simulate angled capture."""
    h, w = image.shape[:2]
    max_shift_x = int(w * max_shift_ratio)
    max_shift_y = int(h * max_shift_ratio)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    dst = src.copy()
    corners_to_shift = np.random.choice(4, size=np.random.randint(2, 4), replace=False)

    for corner in corners_to_shift:
        dx = np.random.randint(-max_shift_x, max_shift_x)
        dy = np.random.randint(-max_shift_y, max_shift_y)
        dst[corner][0] += dx
        dst[corner][1] += dy

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return warped


def add_motion_blur(
    image: np.ndarray,
    kernel_size: int = 15,
    angle: float = None,
) -> np.ndarray:
    """Add directional motion blur to simulate camera shake."""
    if angle is None:
        angle = np.random.uniform(0, 360)

    # Directional PSF kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    # Draw a line through the center at the given angle
    cos_val = np.cos(np.radians(angle))
    sin_val = np.sin(np.radians(angle))

    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * cos_val)
        y = int(center + offset * sin_val)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1

    kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel

    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def add_jpeg_artifacts(
    image: np.ndarray,
    quality: int = None,
) -> np.ndarray:
    """Re-encode at low JPEG quality to introduce compression artifacts."""
    if quality is None:
        quality = np.random.randint(15, 40)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0,
    sigma: float = 25,
) -> np.ndarray:
    """Add Gaussian noise to simulate low-light capture."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_partial_occlusion(
    image: np.ndarray,
    occlusion_ratio: float = 0.15,
) -> np.ndarray:
    """Simulate a finger partially covering a corner of the document."""
    result = image.copy()
    h, w = result.shape[:2]

    # Random corner
    corner = np.random.choice(4)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    cx, cy = corners[corner]

    radius_x = int(w * occlusion_ratio)
    radius_y = int(h * occlusion_ratio * 1.5)
    angle = np.random.uniform(-30, 30)

    skin_color = (
        np.random.randint(140, 190),
        np.random.randint(130, 170),
        np.random.randint(160, 210),
    )

    cv2.ellipse(
        result, (cx, cy), (radius_x, radius_y),
        angle, 0, 360, skin_color, -1,
    )

    # Slight blur to soften edges
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (radius_x, radius_y), angle, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    result = (result.astype(np.float32) * mask_3c +
              image.astype(np.float32) * (1.0 - mask_3c))

    return np.clip(result, 0, 255).astype(np.uint8)


def add_moire_pattern(
    image: np.ndarray,
    frequency: float = None,
    intensity: float = 0.3,
) -> np.ndarray:
    """Add a moiré interference pattern to simulate a photo of a screen."""
    if frequency is None:
        frequency = np.random.uniform(0.05, 0.15)

    h, w = image.shape[:2]

    # Two slightly detuned sine grids produce a moiré beat pattern
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    pattern1 = np.sin(2 * np.pi * frequency * xx) * 0.5 + 0.5
    pattern2 = np.sin(2 * np.pi * frequency * 1.03 * yy) * 0.5 + 0.5

    moire = (pattern1 * pattern2 * 255 * intensity).astype(np.float32)
    moire_3c = np.stack([moire, moire, moire], axis=2)

    result = image.astype(np.float32) + moire_3c
    return np.clip(result, 0, 255).astype(np.uint8)


def add_random_rotation(
    image: np.ndarray,
    max_angle: float = 30.0,
) -> np.ndarray:
    """Random rotation to simulate tilted capture."""
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def adjust_brightness(
    image: np.ndarray,
    factor: float = None,
) -> np.ndarray:
    """Random brightness adjustment to simulate varying lighting."""
    if factor is None:
        factor = np.random.uniform(0.5, 1.5)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result




class SyntheticDataGenerator:
    """Generates degraded document images from clean samples for training and benchmarking."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()

        # Augmentation registry: (name, function, probability)
        self.augmentations = [
            ("glare", add_glare, self.config.glare_probability),
            ("perspective", add_perspective_warp, self.config.warp_probability),
            ("motion_blur", add_motion_blur, self.config.blur_probability),
            ("jpeg_artifacts", add_jpeg_artifacts, self.config.jpeg_artifact_probability),
            ("noise", add_gaussian_noise, self.config.noise_probability),
            ("occlusion", add_partial_occlusion, self.config.occlusion_probability),
        ]

        self._album_transform: Optional[object] = None
        if _HAS_ALBUMENTATIONS:
            self._album_transform = A.Compose([
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=7, p=1.0),
                    A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                ], p=0.4),
                A.ImageCompression(quality_lower=30, quality_upper=70, p=0.3),
                A.CoarseDropout(
                    max_holes=3, max_height=32, max_width=32,
                    min_holes=1, min_height=8, min_width=8, p=0.2,
                ),
                A.Affine(
                    scale=(0.95, 1.05), translate_percent=(-0.03, 0.03),
                    rotate=(-5, 5), shear=(-3, 3),
                    border_mode=cv2.BORDER_REPLICATE, p=0.3,
                ),
            ])
            logger.info("Albumentations augmentation pipeline initialised")
        else:
            logger.warning(
                "albumentations not installed — falling back to pure-OpenCV augmentations"
            )

    def generate(
        self,
        input_dir: str,
        output_dir: str = None,
        num_per_image: int = None,
    ) -> List[AugmentationRecord]:
        """Generate augmented variants from a directory of clean images."""
        output_dir = output_dir or self.config.output_dir
        num_per_image = num_per_image or self.config.num_augmented_per_image

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Ensure output directories exist
        categories = [
            "glare", "perspective", "motion_blur", "jpeg_artifacts",
            "noise", "occlusion", "moire", "combined", "rotation",
            "brightness",
        ]
        for cat in categories:
            (output_path / cat).mkdir(parents=True, exist_ok=True)

        # Collect input images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        input_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not input_files:
            logger.error(f"No images found in {input_dir}")
            return []

        logger.info(
            f"Generating {num_per_image} augmented images per source "
            f"from {len(input_files)} clean images"
        )

        all_records = []

        for img_file in input_files:
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Could not read: {img_file}")
                continue

            records = self._augment_single_image(
                image, img_file.stem, output_path, num_per_image
            )
            all_records.extend(records)

        # Save manifest
        manifest = [r.__dict__ for r in all_records]
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Generated {len(all_records)} augmented images. "
            f"Manifest saved to {manifest_path}"
        )

        return all_records

    def _augment_single_image(
        self,
        image: np.ndarray,
        base_name: str,
        output_path: Path,
        num_variants: int,
    ) -> List[AugmentationRecord]:
        """Generate augmented variants for a single clean image."""
        records = []
        count = 0

        # Generate a few examples of each individual augmentation type
        single_aug_count = max(1, num_variants // (len(self.augmentations) + 3))

        for aug_name, aug_fn, _ in self.augmentations:
            for i in range(single_aug_count):
                augmented = aug_fn(image)
                filename = f"{base_name}_{aug_name}_{i:04d}.jpg"
                filepath = output_path / aug_name / filename
                cv2.imwrite(str(filepath), augmented)

                records.append(AugmentationRecord(
                    source_image=base_name,
                    output_image=str(filepath),
                    augmentations_applied=[aug_name],
                    parameters={},
                ))
                count += 1

        # Moiré (special case)
        for i in range(single_aug_count):
            augmented = add_moire_pattern(image)
            filename = f"{base_name}_moire_{i:04d}.jpg"
            filepath = output_path / "moire" / filename
            cv2.imwrite(str(filepath), augmented)
            records.append(AugmentationRecord(
                source_image=base_name,
                output_image=str(filepath),
                augmentations_applied=["moire"],
                parameters={},
            ))
            count += 1

        # Rotation
        for i in range(single_aug_count):
            augmented = add_random_rotation(image, self.config.max_rotation_degrees)
            filename = f"{base_name}_rotation_{i:04d}.jpg"
            filepath = output_path / "rotation" / filename
            cv2.imwrite(str(filepath), augmented)
            records.append(AugmentationRecord(
                source_image=base_name,
                output_image=str(filepath),
                augmentations_applied=["rotation"],
                parameters={},
            ))
            count += 1

        # Brightness
        for i in range(single_aug_count):
            augmented = adjust_brightness(image)
            filename = f"{base_name}_brightness_{i:04d}.jpg"
            filepath = output_path / "brightness" / filename
            cv2.imwrite(str(filepath), augmented)
            records.append(AugmentationRecord(
                source_image=base_name,
                output_image=str(filepath),
                augmentations_applied=["brightness"],
                parameters={},
            ))
            count += 1

        remaining = num_variants - count
        for i in range(max(0, remaining)):
            augmented = image.copy()
            applied = []

            # Use Albumentations pipeline ~40% of the time for combined variants
            if self._album_transform is not None and np.random.random() < 0.4:
                augmented = self._album_transform(image=augmented)["image"]
                applied.append("albumentations")

            # Randomly apply multiple custom augmentations
            for aug_name, aug_fn, prob in self.augmentations:
                if np.random.random() < prob:
                    augmented = aug_fn(augmented)
                    applied.append(aug_name)

            # Also possibly add rotation and brightness
            if np.random.random() < 0.3:
                augmented = add_random_rotation(augmented, 15)
                applied.append("rotation")

            if np.random.random() < 0.3:
                augmented = adjust_brightness(augmented)
                applied.append("brightness")

            if not applied:
                # If nothing was applied, force at least one
                aug_name, aug_fn, _ = self.augmentations[
                    np.random.randint(len(self.augmentations))
                ]
                augmented = aug_fn(image)
                applied = [aug_name]

            filename = f"{base_name}_combined_{i:04d}.jpg"
            filepath = output_path / "combined" / filename
            cv2.imwrite(str(filepath), augmented)

            records.append(AugmentationRecord(
                source_image=base_name,
                output_image=str(filepath),
                augmentations_applied=applied,
                parameters={},
            ))

        return records

    def generate_tamper_pair(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a (clean, tampered) pair for tamper classifier training."""
        tampered = image.copy()
        h, w = tampered.shape[:2]

        # Select a random region
        rx = np.random.randint(0, w // 2)
        ry = np.random.randint(0, h // 2)
        rw = np.random.randint(w // 8, w // 3)
        rh = np.random.randint(h // 8, h // 4)

        # Source region for clone
        sx = np.random.randint(0, max(1, w - rw))
        sy = np.random.randint(0, max(1, h - rh))

        # Ensure coordinates are within bounds
        rw = min(rw, w - rx, w - sx)
        rh = min(rh, h - ry, h - sy)

        if rw > 0 and rh > 0:
            source_patch = image[sy : sy + rh, sx : sx + rw]

            # Use Poisson blending ~50% of the time
            use_poisson = np.random.random() < 0.5
            if use_poisson and rw > 4 and rh > 4:
                cx_dst = rx + rw // 2
                cy_dst = ry + rh // 2
                mask = 255 * np.ones(source_patch.shape[:2], dtype=np.uint8)
                try:
                    tampered = cv2.seamlessClone(
                        source_patch, tampered, mask,
                        (cx_dst, cy_dst), cv2.NORMAL_CLONE,
                    )
                except cv2.error:
                    tampered[ry : ry + rh, rx : rx + rw] = source_patch
            else:
                tampered[ry : ry + rh, rx : rx + rw] = source_patch

        if self._album_transform is not None and np.random.random() < 0.4:
            tampered = self._album_transform(image=tampered)["image"]

        # Re-compress to introduce detectable artifacts
        quality = np.random.randint(70, 95)
        _, encoded = cv2.imencode(".jpg", tampered, [cv2.IMWRITE_JPEG_QUALITY, quality])
        tampered = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return image, tampered

"""Central, environment-aware configuration with YAML and env-var support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

@dataclass(frozen=True, slots=True)
class QualityGateConfig:

    blur_threshold: float = 100.0
    tenengrad_threshold: float = 300.0
    min_brightness: float = 30.0
    max_brightness: float = 240.0
    min_resolution: int = 100
    max_glare_ratio: float = 0.50
    glare_intensity_threshold: int = 252

@dataclass(frozen=True, slots=True)
class RectificationConfig:

    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 5
    min_area_ratio: float = 0.10
    approx_epsilon_ratio: float = 0.02
    morph_kernel_size: int = 5
    target_width: int = 600
    target_height: int = 400
    # Homography validation
    max_reprojection_error: float = 5.0
    min_aspect_ratio: float = 0.9
    max_aspect_ratio: float = 2.5


@dataclass(frozen=True, slots=True)
class EnhancementConfig:

    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)
    denoise_h: int = 10
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    # Document-specific OCR enhancement
    sauvola_window_size: int = 25
    sauvola_k: float = 0.08
    tenengrad_threshold: float = 300.0


@dataclass(frozen=True, slots=True)
class GlareConfig:

    hsv_v_threshold: int = 245
    min_glare_area: int = 500
    inpaint_radius: int = 5
    dilate_kernel_size: int = 15


@dataclass(frozen=True, slots=True)
class ELAConfig:

    jpeg_quality: int = 95
    anomaly_threshold: float = 0.5
    scale_factor: int = 15
    block_size: int = 8
    # DCT analysis weights
    dct_block_size: int = 8
    cv_weight: float = 0.3
    hot_ratio_weight: float = 0.4
    dct_weight: float = 0.3


@dataclass(frozen=True, slots=True)
class OCRConfig:

    engine: str = "easyocr"
    fallback_engine: str = "easyocr"
    confidence_threshold: float = 0.6
    languages: list = field(default_factory=lambda: ["en"])
    use_gpu: bool = False
    paddle_det_model: str = "en_PP-OCRv4_det"
    paddle_rec_model: str = "en_PP-OCRv4_rec"


@dataclass(frozen=True, slots=True)
class TamperModelConfig:

    model_name: str = "tamper_classifier"
    input_size: tuple = (224, 224)
    num_classes: int = 2
    backbone: str = "mobilenetv3_small"
    pretrained: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    model_save_path: str = "models/tamper_classifier.pth"


@dataclass(frozen=True, slots=True)
class SyntheticConfig:

    num_augmented_per_image: int = 100
    glare_probability: float = 0.5
    warp_probability: float = 0.5
    blur_probability: float = 0.3
    jpeg_artifact_probability: float = 0.3
    noise_probability: float = 0.3
    occlusion_probability: float = 0.2
    max_rotation_degrees: float = 30.0
    output_dir: str = "data/synthetic"


@dataclass(frozen=True, slots=True)
class NoiseAnalysisConfig:

    block_size: int = 32
    inconsistency_threshold: float = 0.35


@dataclass(frozen=True, slots=True)
class JPEGGhostConfig:

    block_size: int = 16
    q_low: int = 50
    q_high: int = 98
    q_step: int = 2


@dataclass(frozen=True, slots=True)
class CopyMoveConfig:

    n_features: int = 5000
    ratio_thresh: float = 0.75
    min_distance: int = 50
    ransac_reproj: float = 5.0
    min_inliers: int = 15


@dataclass
class DocuNetConfig:
    """Master configuration for the DocuNet pipeline."""

    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    rectification: RectificationConfig = field(default_factory=RectificationConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    glare: GlareConfig = field(default_factory=GlareConfig)
    ela: ELAConfig = field(default_factory=ELAConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    tamper_model: TamperModelConfig = field(default_factory=TamperModelConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    noise_analysis: NoiseAnalysisConfig = field(default_factory=NoiseAnalysisConfig)
    jpeg_ghost: JPEGGhostConfig = field(default_factory=JPEGGhostConfig)
    copy_move: CopyMoveConfig = field(default_factory=CopyMoveConfig)

    # Global settings
    debug: bool = False
    log_level: str = "INFO"
    output_dir: str = "output"
    db_url: str = "sqlite:///docunet.db"

    @classmethod
    def default(cls) -> DocuNetConfig:
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> DocuNetConfig:
        """Load configuration from a YAML file, falling back to defaults."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        import yaml
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocuNetConfig:
        """Build a config from a flat or nested dictionary."""
        sub_map: dict[str, type] = {
            "quality_gate": QualityGateConfig,
            "rectification": RectificationConfig,
            "enhancement": EnhancementConfig,
            "glare": GlareConfig,
            "ela": ELAConfig,
            "ocr": OCRConfig,
            "tamper_model": TamperModelConfig,
            "synthetic": SyntheticConfig,
            "noise_analysis": NoiseAnalysisConfig,
            "jpeg_ghost": JPEGGhostConfig,
            "copy_move": CopyMoveConfig,
        }

        kwargs: dict[str, Any] = {}
        for key, sub_cls in sub_map.items():
            if key in data and isinstance(data[key], dict):
                kwargs[key] = sub_cls(**data[key])
        # Top-level scalars
        for scalar in ("debug", "log_level", "output_dir", "db_url"):
            if scalar in data:
                kwargs[scalar] = data[scalar]

        return cls(**kwargs)

    def with_env_overrides(self, prefix: str = "DOCUNET") -> DocuNetConfig:
        """Return a new config with values overridden by env variables.

        Naming: {PREFIX}_{SECTION}__{KEY} (double-underscore separator).
        """
        data = self._to_nested_dict()

        for env_key, env_val in os.environ.items():
            if not env_key.startswith(f"{prefix}_"):
                continue
            parts = env_key[len(prefix) + 1 :].lower().split("__")
            if len(parts) == 2:
                section, key = parts
                if section in data and isinstance(data[section], dict):
                    data[section][key] = _coerce(env_val)
            elif len(parts) == 1:
                data[parts[0]] = _coerce(env_val)

        return DocuNetConfig.from_dict(data)

    def _to_nested_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if hasattr(val, "__dataclass_fields__"):
                result[f.name] = {
                    sf.name: getattr(val, sf.name) for sf in fields(val)
                }
            else:
                result[f.name] = val
        return result


def _coerce(value: str) -> int | float | bool | str:
    """Best-effort type coercion for env variable strings."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

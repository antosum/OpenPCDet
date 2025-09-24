"""Inference engine wrapper for OpenPCDet models.

This module exposes :class:`PCDetEngine`, a lightweight helper designed for
single-frame LiDAR inference pipelines. It focuses on PointPillars and
CenterPoint-style models but keeps the setup generic enough to extend further.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file


class PCDetEngine:
    """Inference helper to run OpenPCDet models on individual LiDAR frames."""

    def __init__(self, cfg_file: str, ckpt_path: str, device: str = "cuda") -> None:
        self.cfg_file = Path(cfg_file).expanduser().resolve()
        self.ckpt_path = Path(ckpt_path).expanduser()
        self.device = torch.device(device)

        self.cfg: Optional[EasyDict] = None
        self.dataset = None
        self.model = None
        self.class_names = None

        self._load_cfg()

    def _load_cfg(self) -> None:
        """Load the YAML config and strip training-only augmentations."""

        if not self.cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.cfg_file}")

        cfg_struct = cfg_from_yaml_file(str(self.cfg_file), EasyDict())
        cfg_struct = copy.deepcopy(cfg_struct)

        data_cfg = cfg_struct.get("DATA_CONFIG")
        if data_cfg is None:
            raise KeyError("DATA_CONFIG missing from configuration")

        self._sanitize_data_config(data_cfg)

        self.cfg = cfg_struct
        self.data_config = data_cfg
        self.class_names = list(cfg_struct.get("CLASS_NAMES", []))

    def _sanitize_data_config(self, data_cfg: EasyDict) -> None:
        """Remove train-time augmentations and shuffling for inference."""

        data_cfg.pop("DATA_AUGMENTOR", None)

        processors = data_cfg.get("DATA_PROCESSOR", [])
        for proc in processors:
            shuffle_cfg = proc.get("SHUFFLE_ENABLED")
            if shuffle_cfg is None:
                continue

            if isinstance(shuffle_cfg, dict):
                for split in shuffle_cfg:
                    shuffle_cfg[split] = False
            else:
                proc["SHUFFLE_ENABLED"] = False

    def predict(self, points: torch.Tensor) -> Dict[str, Any]:
        """Run a forward pass on a single LiDAR frame.

        Phase 3 fills in the body; currently only here to anchor the API.
        """
        raise NotImplementedError("Phase 3 will implement prediction")

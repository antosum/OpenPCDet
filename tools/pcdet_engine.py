"""Lean LiDAR inference engine for OpenPCDet models.

The :class:`PCDetEngine` helper targets point-cloud only, single-frame pipelines
for PointPillars and CenterPoint style detectors. Camera/BEV image branches are
intentionally omitted to minimise overhead for live deployments.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import build_network


class PCDetEngine:
    """Inference helper to run OpenPCDet models on individual LiDAR frames."""

    def __init__(
        self,
        cfg_file: str,
        ckpt_path: str,
        device: str = "cuda",
        *,
        enable_tf32: bool = True,
        matmul_precision: str = "high",
        cudnn_benchmark: bool = True,
        deterministic: bool = False,
        use_autocast: bool = False,
        autocast_dtype: str = "bf16",
        configure_allocator: bool = True,
    ) -> None:
        self.logger = logging.getLogger("pcdet_engine")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)5s %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.logger.info("Initializing PCDetEngine with cfg=%s, ckpt=%s", cfg_file, ckpt_path)

        self.cfg_file = Path(cfg_file).expanduser().resolve()
        self.ckpt_path = Path(ckpt_path).expanduser()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False")

        # Backend/runtime preferences (Phase 1: safe toggles)
        self.enable_tf32 = enable_tf32
        self.matmul_precision = matmul_precision
        self.cudnn_benchmark = cudnn_benchmark
        self.deterministic = deterministic
        self.use_autocast = bool(use_autocast)
        self.autocast_dtype = autocast_dtype
        self.configure_allocator = bool(configure_allocator)

        self.cfg: Optional[EasyDict] = None
        self.model_cfg: Optional[EasyDict] = None
        self.dataset = None
        self.model = None
        self.class_names = None

        self._load_cfg()
        self.logger.info("Configuration loaded successfully")

        # Configure PyTorch backends for inference throughput
        self._configure_backends()

        # Apply runtime environment tweaks (allocator, launch blocking warnings)
        self._configure_runtime_environment()

        self._build_dataset()
        self.logger.info("Dataset instantiated for live inference (mode=%s)", self.dataset.mode)

        self._build_model()
        self.logger.info("Model ready: %s", self.model.__class__.__name__)
        self._frame_idx = 0

    def _configure_backends(self) -> None:
        """Apply safe, opt-in backend settings for inference throughput."""
        if self.device.type == "cuda":
            try:
                # TF32 can significantly speed up matmul/conv on Ampere+ with minimal accuracy loss
                torch.backends.cuda.matmul.allow_tf32 = bool(self.enable_tf32)
                torch.backends.cudnn.allow_tf32 = bool(self.enable_tf32)
            except Exception:
                pass

            try:
                # cuDNN autotuner is useful for fixed input shapes common in BEV backbones
                torch.backends.cudnn.benchmark = bool(self.cudnn_benchmark)
            except Exception:
                pass

        try:
            # Prefer high-performance FP32 matmul kernels
            torch.set_float32_matmul_precision(str(self.matmul_precision))
        except Exception:
            pass

        try:
            # Favor throughput unless strict determinism is required
            torch.use_deterministic_algorithms(bool(self.deterministic))
        except Exception:
            pass

    def _configure_runtime_environment(self) -> None:
        """Set allocator defaults and warn on blocking configs."""
        if self.device.type == "cuda" and self.configure_allocator:
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            if not alloc_conf:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                self.logger.info(
                    "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for allocator stability"
                )
            elif "expandable_segments" not in alloc_conf:
                self.logger.debug(
                    "Allocator config already set (%s); leaving as-is", alloc_conf
                )

        launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING")
        if launch_blocking == "1":
            self.logger.warning(
                "CUDA_LAUNCH_BLOCKING=1 detected; async execution disabled and latency may increase"
            )

    def _load_cfg(self) -> None:
        """Load the YAML config and strip training-only augmentations."""

        if not self.cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.cfg_file}")

        self.logger.info("Loading configuration from %s", self.cfg_file)
        cfg_struct = cfg_from_yaml_file(str(self.cfg_file), EasyDict())
        cfg_struct = copy.deepcopy(cfg_struct)

        data_cfg = cfg_struct.get("DATA_CONFIG")
        if data_cfg is None:
            raise KeyError("DATA_CONFIG missing from configuration")

        self._sanitize_data_config(data_cfg)

        self.cfg = cfg_struct
        self.data_config = data_cfg
        self.class_names = list(cfg_struct.get("CLASS_NAMES", []))
        self.model_cfg = cfg_struct.get("MODEL")
        if self.model_cfg is None:
            raise KeyError("MODEL configuration missing from cfg file")

    def _build_dataset(self) -> None:
        """Instantiate a lightweight dataset for inference-time preprocessing."""

        dataset_cfg = copy.deepcopy(self.data_config)
        root_path = Path(dataset_cfg.get("DATA_PATH", "")).expanduser()
        self.logger.info("Preparing inference dataset with root=%s", root_path)

        class LiveInferenceDataset(DatasetTemplate):
            """Minimal DatasetTemplate wrapper for live frame inference."""

            def __len__(self) -> int:  # pragma: no cover - trivial
                return 1

            def __getitem__(self, index: int):  # pragma: no cover - unused
                raise RuntimeError("LiveInferenceDataset does not support indexing")

        self.dataset = LiveInferenceDataset(
            dataset_cfg=dataset_cfg,
            class_names=self.class_names,
            training=False,
            root_path=root_path,
            logger=self.logger,
        )
        # Mark and report important preprocessing attributes for clarity.
        setattr(self.dataset, "is_live_inference", True)
        used_feats = list(dataset_cfg.get("POINT_FEATURE_ENCODING", {}).get("used_feature_list", []))
        try:
            vox = tuple(self.dataset.data_processor.voxel_size.tolist())
        except Exception:
            vox = tuple(self.dataset.data_processor.voxel_size)
        grid = tuple(int(x) for x in getattr(self.dataset, "grid_size", []))
        pc_range = list(getattr(self.dataset, "point_cloud_range", []))
        self.logger.info(
            "LiveInferenceDataset ready (no infos/ImageSets; training=False).\n"
            "- Features expected: %d -> %s\n"
            "- Voxel size: %s  Grid size: %s\n"
            "- Point cloud range: %s",
            getattr(self.dataset.point_feature_encoder, "num_point_features", len(used_feats)),
            used_feats,
            vox,
            grid,
            pc_range,
        )

    def _build_model(self) -> None:
        """Create the detector network and load checkpoint weights."""

        num_class = len(self.class_names) if self.class_names is not None else 0
        self.logger.info("Building model network")
        self.model = build_network(self.model_cfg, num_class=num_class, dataset=self.dataset)
        self.model.to(self.device)
        self.model.eval()

        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {self.ckpt_path}")

        self.logger.info("Loading checkpoint weights from %s", self.ckpt_path)
        to_cpu = self.device.type == "cpu"
        self.model.load_params_from_file(
            filename=str(self.ckpt_path),
            logger=self.logger,
            to_cpu=to_cpu,
        )
        if not to_cpu:
            self.model.to(self.device)

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

    def predict(
        self,
        points: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """Run inference on a single LiDAR frame.

        Args:
            points: Array or tensor with shape ``(N, C)`` describing LiDAR points in
                the order expected by the configured ``POINT_FEATURE_ENCODING``.
        Returns:
            Dictionary containing ``boxes_lidar``, ``scores``, ``labels`` as NumPy arrays.
        """

        if self.dataset is None or self.model is None:
            raise RuntimeError("PCDetEngine is not fully initialized")

        points_np, pad_info = self._normalize_points(points)
        roi_mask = self._compute_roi_mask(points_np)

        if roi_mask.size != len(points_np):
            raise RuntimeError(
                "ROI mask length mismatch; ensure _compute_roi_mask returns per-point mask"
            )

        if not roi_mask.any():
            empty_result = {
                "boxes_lidar": np.empty((0, 7), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
            }
            return empty_result

        points_np = points_np[roi_mask]

        frame_id = f"live_{self._frame_idx:06d}"
        self._frame_idx += 1

        input_dict = {
            "frame_id": frame_id,
            "points": points_np,
        }

        data_dict = self.dataset.prepare_data(input_dict)
        if data_dict is None:
            raise RuntimeError("Data preparation returned None; check point cloud validity")

        batch_dict = self.dataset.collate_batch([data_dict])
        self._load_to_device(batch_dict)

        # Use inference_mode to avoid autograd overhead during inference
        autocast_ctx = (
            torch.autocast(
                device_type=self.device.type,
                dtype=self._resolve_autocast_dtype(),
            )
            if self._should_autocast()
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            pred_dicts, _ = self.model(batch_dict)

        if not pred_dicts:
            empty_result = {
                "boxes_lidar": np.empty((0, 7), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
            }
            return empty_result

        first = pred_dicts[0]
        boxes = first.get("pred_boxes")
        scores = first.get("pred_scores")
        labels = first.get("pred_labels")

        if boxes is None or scores is None or labels is None:
            raise RuntimeError("Prediction dict is missing required keys")

        boxes = boxes.to(self.device) if boxes.device != self.device else boxes
        scores = scores.to(self.device) if scores.device != self.device else scores
        labels = labels.to(self.device) if labels.device != self.device else labels

        # Ensure NumPy-compatible dtypes even under autocast (e.g., bf16)
        boxes_cpu = boxes.detach().cpu().to(torch.float32)
        scores_cpu = scores.detach().cpu().to(torch.float32)
        labels_cpu = labels.detach().cpu().to(torch.int32)

        result = {
            "boxes_lidar": boxes_cpu.numpy(),
            "scores": scores_cpu.numpy(),
            "labels": labels_cpu.numpy(),
        }

        if pad_info:
            self.logger.debug("Applied feature padding: %s", pad_info)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_points(
        self, points: Union[np.ndarray, torch.Tensor]
    ) -> tuple[np.ndarray, Optional[str]]:
        if isinstance(points, torch.Tensor):
            points_np = points.detach().cpu().numpy()
        else:
            points_np = np.asarray(points)

        if points_np.ndim != 2:
            raise ValueError(f"points must be a 2D array, got shape {points_np.shape}")

        points_np = points_np.astype(np.float32, copy=False)
        points_np = np.ascontiguousarray(points_np)

        if points_np.shape[1] < 3:
            raise ValueError(
                f"points must include at least XYZ columns; received shape {points_np.shape}"
            )

        expected_feats = int(getattr(self.dataset.point_feature_encoder, "num_point_features", points_np.shape[1]))
        pad_info = None

        if points_np.shape[1] < expected_feats:
            pad_width = expected_feats - points_np.shape[1]
            points_np = np.pad(points_np, ((0, 0), (0, pad_width)), constant_values=0.0)
            pad_info = f"padded {pad_width} feature(s) to match expected {expected_feats}"
        elif points_np.shape[1] > expected_feats:
            raise ValueError(
                f"points provide {points_np.shape[1]} features but config expects {expected_feats}"
            )

        return points_np, pad_info

    def _compute_roi_mask(self, points_sensor: np.ndarray) -> np.ndarray:
        dataset = getattr(self, "dataset", None)
        point_cloud_range = None if dataset is None else getattr(dataset, "point_cloud_range", None)
        if point_cloud_range is None or len(point_cloud_range) < 6:
            raise RuntimeError("Detector dataset missing a valid point_cloud_range for ROI filtering")

        min_x, min_y, min_z, max_x, max_y, max_z = point_cloud_range
        mask = (
            (points_sensor[:, 0] >= min_x)
            & (points_sensor[:, 0] <= max_x)
            & (points_sensor[:, 1] >= min_y)
            & (points_sensor[:, 1] <= max_y)
            & (points_sensor[:, 2] >= min_z)
            & (points_sensor[:, 2] <= max_z)
        )

        return mask

    def _load_to_device(self, batch_dict: Dict[str, Any]) -> None:
        for key, val in list(batch_dict.items()):
            if key in {"frame_id", "metadata", "calib"}:
                continue

            if isinstance(val, torch.Tensor):
                batch_dict[key] = val.to(self.device, non_blocking=True)
                continue

            if not isinstance(val, np.ndarray):
                continue

            tensor = torch.from_numpy(val)
            tensor = tensor.float() if tensor.is_floating_point() else tensor.int()
            batch_dict[key] = tensor.to(self.device, non_blocking=True)

    def _should_autocast(self) -> bool:
        return self.use_autocast and self.device.type == "cuda"

    def _resolve_autocast_dtype(self) -> torch.dtype:
        dtype_key = self.autocast_dtype.lower()
        if dtype_key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_key in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(f"Unsupported autocast dtype: {self.autocast_dtype}")

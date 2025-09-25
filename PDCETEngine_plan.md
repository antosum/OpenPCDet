PCDetEngine Implementation Plan
--------------------------------

- **Goal**: Build a lightweight, real-time inference engine for OpenPCDet models (with PointPillars and CenterPoint Pillars as first targets) that processes live LiDAR frames arriving as `np.ndarray` of shape `(N, 4)` containing `[x, y, z, intensity]`. The engine should skip any train-time augmentations and metrics, and return post-NMS outputs `{boxes_lidar, scores, labels}` as NumPy arrays. CUDA is the default device; CPU fallback is only supported when practical.

Phases Overview
---------------

1. Config Load & Sanitization ✅
2. Inference Dataset (no file IO) ✅
3. Predict Pipeline (np.ndarray → detections) ⏳
4. Verbosity, Options, Safety ⏳
5. Optional Polishing ⏳

Phase 1 — Config Load & Sanitization (Completed)
-------------------------------------------------

- **Implementation**
  - Added `tools/pcdet_engine.py` with `PCDetEngine` skeleton, logging, and device setup.
  - Load YAML configs via `cfg_from_yaml_file`, cloning into a mutable `EasyDict` for manipulation.
  - Remove `DATA_CONFIG.DATA_AUGMENTOR` and disable any `DATA_PROCESSOR.shuffle_points.SHUFFLE_ENABLED` entries.
  - Cache key sections: `CLASS_NAMES`, `MODEL`, and the sanitized `DATA_CONFIG`.
  - Normalize paths (`cfg_file.expanduser().resolve()`, `ckpt.expanduser()`).

- **Artifacts**
  - `tools/pcdet_engine.py`: `_load_cfg`, `_sanitize_data_config`, base class structure.

- **Notes**
  - Verified parsing with both configs:
    - `tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint.yaml`
    - `tools/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml`
  - Logs confirm successful config loading and sanitization.

Phase 2 — Inference Dataset (Completed)
---------------------------------------

- **Problem**
  - Standard dataset classes (e.g., `CustomDataset`) perform file IO at init (ImageSets, info files), which is incompatible with live per-frame inference.

- **Approach**
  - Implement `LiveInferenceDataset`, a minimal subclass of `pcdet.datasets.dataset.DatasetTemplate` that:
    - Runs with `training=False` (no augmentor).
    - Utilizes `PointFeatureEncoder`, `DataProcessor`, `point_cloud_range`, `grid_size`, and `voxel_size` supplied by `DatasetTemplate`.
    - Leaves `__getitem__` unimplemented (we call `prepare_data` manually).

- **Implementation**
  - `_build_dataset` now clones the sanitized `DATA_CONFIG`, resolves `DATA_PATH`, and instantiates `LiveInferenceDataset` without touching disk.
  - Logging now reports:
    - “Dataset instantiated for live inference (mode=test)”.
    - Preprocessing details (feature expectations from encoder, voxel size, grid size, point cloud range).
    - `is_live_inference=True` marker on the dataset.

- **Results**
  - Engine init no longer triggers logs about loading dataset infos/ImageSets.
  - Model construction and checkpoint load remain unaffected and reuse the prepared dataset metadata.

Phase 3 — Predict Pipeline (np.ndarray → Detections) (Pending)
--------------------------------------------------------------

- **Objective**
  - Implement `PCDetEngine.predict(points_np)` to handle a single frame:
    - Accept `np.ndarray` `(N, 4)` float32 `[x, y, z, intensity]`.
    - Pad extra features if required by `POINT_FEATURE_ENCODING.used_feature_list` (e.g., zero timestamp).
    - Execute dataset preprocessing via `prepare_data` and `collate_batch` (batch size 1).
    - Move data to device and run the model under `torch.no_grad()`.
    - Return `{boxes_lidar, scores, labels}` as NumPy arrays (post-NMS, identical to model outputs in `test.py`).

- **Steps**
  - Validate input shape/dtype; convert or copy as needed.
  - Feature padding logic:
    - Compare input channel count with encoder expectations.
    - Zero-pad missing features; raise an error if extra features are provided.
  - Build `input_dict = {'frame_id': f'live_{counter}', 'points': points_np_padded}`.
  - Process through `prepare_data`, then `collate_batch([data])`.
  - Implement a lightweight single-frame `load_data_to_gpu` routine (inspired by `pcdet.models.load_data_to_gpu`).
  - Run model inference and extract the first sample’s predictions.

- **Output**
  - Dictionary with:
    - `boxes_lidar`: `(M, 7 or 9)` NumPy array.
    - `scores`: `(M,)` NumPy array.
    - `labels`: `(M,)` NumPy array (1-based class indices).
  - Optional `score_thresh` argument for local filtering.

- **Validation**
  - Smoke tests with synthetic point clouds for both Geminai and NuScenes configs.
  - Confirm latency is in the same ballpark as `tools/test.py --batch_size 1`.

Phase 4 — Verbosity, Options, Safety (Pending)
----------------------------------------------

- **Logging**
  - Report feature padding decisions, preprocessing timings, and inference/device details per prediction.

- **Options**
  - Add `score_thresh` override support in `predict`.
  - Surface device configuration clearly and guard unsupported CPU execution paths (e.g., where spconv is required).

- **Safety / Robustness**
  - Provide clear errors when configs expect features that cannot be generated.
  - Better messaging for missing or incompatible checkpoints.

- **Validation**
  - Run on real frames (from both target configs) and verify output shapes/values.

Phase 5 — Optional Polishing (Pending)
--------------------------------------

- **Documentation**
  - Add docstrings and a short usage example (instantiate engine, call `predict`).

- **Warm-up**
  - Consider a pre-warm step to amortize first-frame latency (optional).

Current Status
--------------

- Completed Phases 1 and 2; engine now loads configs, sanitizes training-only settings, and constructs a live inference dataset that avoids any dataset file IO.
- Logs clearly indicate no infos/ImageSets are loaded and outline preprocessing expectations.
- Model checkpoints load cleanly (e.g., “Done (loaded 179/179)” refers strictly to model weights).

Challenges and Decisions
------------------------

- **Dataset coupling to file-based infos**
  - *Challenge*: Dataset classes expected to read ImageSets/infos.
  - *Decision*: `LiveInferenceDataset` inherits from `DatasetTemplate` with `training=False`, leveraging existing preprocessing without touching disk.

- **Feature compatibility**
  - *Challenge*: Some configs expect more than 4 features (e.g., timestamp).
  - *Decision*: During prediction, zero-pad missing features based on `POINT_FEATURE_ENCODING.used_feature_list`, matching the behavior of the custom dataset loader.

- **CPU fallback**
  - *Challenge*: Certain CUDA ops (spconv) may not run on CPU.
  - *Decision*: Permit `device="cpu"` only when environment supports it; otherwise raise a clear error. Default remains GPU.

- **Throughput considerations**
  - *Challenge*: Ensure performance close to `tools/test.py` with batch size 1.
  - *Decision*: Reuse `prepare_data` and `collate_batch` to avoid reimplementing voxelization; keep per-frame code minimal.

Next Actions
------------

1. Implement Phase 3 (`PCDetEngine.predict`), including feature padding and inference pipeline.
2. Run smoke tests with synthetic points for both Geminai and NuScenes configs.
3. Proceed to Phase 4 for optional logging improvements and robustness checks once predict path is validated.

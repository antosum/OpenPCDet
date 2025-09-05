**Fine‑Tuning NuScenes Dynamic CenterPoint Pillars on a Custom Dataset**

**Goal**
- Fine‑tune a nuScenes‑pretrained Dynamic CenterPoint Pillars model on a custom dataset while preserving geometry, feature, and head compatibility to maximize reuse of pretrained weights.

**Geometry**
- Voxel size: keep `VOXEL_SIZE: [0.2, 0.2, 8.0]`.
- Z span: enforce single pillar layer by matching `POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2] == 8.0`.
- XY alignment: snap `POINT_CLOUD_RANGE` XY extents to multiples of 0.2 m so `(xmax−xmin)/0.2` and `(ymax−ymin)/0.2` are integers.
- Post range: set `POST_CENTER_LIMIT_RANGE` ≈ `POINT_CLOUD_RANGE` ± small margin (e.g., XY ±5 m, Z ±2 m) to avoid edge clipping.

**Compute and Align Ranges**
- Use the helper script to compute ranges and print ready‑to‑paste YAML lines.
- Examples:
  - Compute from ImageSets splits, align to pillars (0.2,0.2,8.0), and emit post range with padding:
    - `python tools/compute_point_cloud_range.py data/custom --splits train,val --voxel-size 0.2,0.2,8.0 --xy-multiple 16 --z-cells 1 --post-xy-margin 5.0 --post-z-min-margin 2.0 --post-z-max-margin 2.0`
  - Output includes:
    - `POINT_CLOUD_RANGE: [...]`
    - `POST_CENTER_LIMIT_RANGE: [...]`

**Feature Compatibility (nuScenes Pretrain)**
- nuScenes pillars typically use 5 features: `['x','y','z','intensity','timestamp']`.
- If your custom points are 4D (no timestamp), pad a zero column so VFE weights load cleanly.
- Minimal change in loader (pad timestamp when needed):

```
# pcdet/datasets/custom/custom_dataset.py:get_lidar
def get_lidar(self, idx):
    lidar_file = self.root_path / 'points' / (f'{idx}.npy')
    assert lidar_file.exists()
    pts = np.load(lidar_file)
    if pts.ndim == 2 and pts.shape[1] == 4:  # x,y,z,intensity
        pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    return pts
```

- Override point features in your model config (if not using base defaults):

```
DATA_CONFIG:
  POINT_FEATURE_ENCODING:
    encoding_type: absolute_coordinates_encoding
    src_feature_list: ['x','y','z','intensity','timestamp']
    used_feature_list: ['x','y','z','intensity','timestamp']
```

Keeping 4 features is possible, but VFE layers won’t load from the nuScenes checkpoint; backbone/head still can.

**Head, Classes, Imbalance**
- Class names: set `CLASS_NAMES` to match your dataset labels exactly.
- Split heads to mitigate imbalance (dominant “other_vehicule”):
  - `CLASS_NAMES_EACH_HEAD: [['other_vehicule'], ['belt_loader','high_loader']]`
- Keep velocity if the pretrained checkpoint had it:
  - `SEPARATE_HEAD_CFG.HEAD_ORDER: ['center','center_z','dim','rot','vel']`
  - `HEAD_DICT` includes `'vel'`; `LOSS_CONFIG.code_weights` length = 10.
  - If your dataset lacks velocity labels, it still trains (head learns near‑zero).

**Dataset/Loader Notes**
- Use `CustomDataset` unless you’ve implemented and registered your own dataset class in `pcdet/datasets/__init__.py`.
- INFO files must contain `annos` with `gt_boxes_lidar` and `name` per sample.
- Place `DATA_PROCESSOR` under `DATA_CONFIG` (not `DATA_AUGMENTOR`). For pillars, `transform_points_to_voxels_placeholder` is sufficient.

**Augmentation**
- Start modest (flip, small rot/scale). Disable DB sampling until you’ve built a `gt_database` and `*_dbinfos_*.pkl`.
- Later, add `gt_sampling` with skewed `SAMPLE_GROUPS` to oversample rare classes.

**Checkpoint Loading**
- Load the nuScenes checkpoint via your training script flag (e.g., `--pretrained_model <ckpt>`). Expect:
  - Full or near‑full load if feature dims match (with timestamp padding).
  - Partial load if class head shapes differ (expected when class lists change).

**Sanity Checks**
- `PointPillarScatter` enforces `nz==1`; ensure Z span == 8.0 with `VOXEL_SIZE[2]=8.0`.
- Grid size is computed via rounding; aligning spans to 0.2 m multiples avoids off‑by‑one and waste.
- CustomDataset evaluation is KITTI‑style; per‑class metrics are mapped via `MAP_CLASS_TO_KITTI` in the base dataset config. For true custom per‑class metrics, add a custom evaluator.

**Quick Config Snippets**
- Pillar voxelization:
  - `VOXEL_SIZE: [0.2, 0.2, 8.0]`
- Example aligned ranges (edit to match your script output):
  - `POINT_CLOUD_RANGE: [32.914, -48.847, 0.0, 119.314, 47.153, 8.0]`
  - `POST_CENTER_LIMIT_RANGE: [27.914, -53.847, -2.0, 124.314, 52.153, 10.0]`  (example with XY ±5 m, Z −2/+2 m)
- Head grouping for imbalance:
  - `CLASS_NAMES_EACH_HEAD: [['other_vehicule'], ['belt_loader','high_loader']]`

**Order of Operations**
- Generate infos for custom dataset.
- Run `compute_point_cloud_range.py` to compute and align PCR and post range.
- (Optional) Add timestamp padding and update feature lists for pretrain compatibility.
- Verify `DATA_PROCESSOR` placement and pillar settings.
- Set class head grouping and code weights; keep velocity if loading a vel‑trained ckpt.
- Start training with the nuScenes pretrained checkpoint.


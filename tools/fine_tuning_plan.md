Got it—roof-mounted LiDAR, **ground already removed**, dataset in **OpenPCDet custom format**, coordinates in the LiDAR frame with **ground set to z=0** (but your raw z values may be negative because the sensor is above the scene). Below is an updated, **data-efficient fine-tune strategy** for **CenterPoint-Pillars (pretrained on nuScenes)** that bakes in those constraints and the top-down domain shift.

---

# Strategy (from lowest effort → strongest), updated for roof-LiDAR + no-ground

## 0) Make z consistent (this matters most)

Pick **one** convention and keep it identical in train & infer:

* **Recommended:** store **z\_rel = z − z\_ground** so **ground ≈ 0** and object tops ≈ 2–5 m (**non-negative**).

  * Then set `POINT_CLOUD_RANGE[2:5]` to something tight like `[-0.5, 4.0]`.
  * Use a **single thick pillar z-bin** spanning that range.

* If your files already have **negative z** (because the sensor is above everything) and you don’t want to re-export: just set
  `POINT_CLOUD_RANGE[2:5]` to e.g. `[-5.0, 1.0]` and make the **pillar z voxel size exactly (z\_max − z\_min)** (see §2).

> The model doesn’t care about the absolute sign—**it only needs internal consistency** across training and inference.

---

## 1) Geometry/ROI tuned to ramp (small, ground-free)

* Tight ROI in meters (example for “between gear and camera”; adapt to yours):

  * `x: [-20, 60]`, `y: [-25, 25]`, `z:`

    * if **z\_rel**: `[-0.5, 4.0]`
    * if **negative z**: `[-5.0, 1.0]`
* Removing ground means **fewer clutter pillars** → you can raise pillar caps modestly so you don’t miss thin gear:

  * `MAX_POINTS_PER_PILLAR: 40–60`
  * `MAX_NUMBER_OF_PILLARS: 12000–16000` (depends on ROI)

---

## 3) Fine-tune schedule (few labels → quick convergence)

**Stage A — head-only warm-start (10–20 epochs):**

* Load **nuScenes CP-Pillars** checkpoint.
* **Freeze** `vfe`, `map_to_bev`, `backbone_2d` (and all BN) and train only the CenterHead.
* Optimizer: AdamW, LR `1e-3` → cosine, wd `1e-2`.

**Stage B — unfreeze last block (10–20 epochs):**

* Unfreeze the **last** backbone\_2d stage (+ keep BN frozen or very small LR).
* LR `1e-4`.

**Stage C — full fine-tune (optional, 10–20 epochs):**

* Unfreeze all; LR `5e-5–1e-4`.

> This 2–3 stage schedule handles the **domain shift** (sensor height / top-down) fast without overfitting when labels are limited.

---

## 4) Augmentations that fit “roof view, no ground”

* **Do:**

  * Yaw ±15–20°, scale 0.95–1.05, xy jitter ±0.2 m.
  * **Object BEV copy-paste** from your cleanest labels (great since ground is gone; clamp pasted `z` to each class’s empirical range).
  * Random point **dropout** (occlusion by people/carts/jetways).
* **Avoid:**

  * Pitch/roll aug (unnecessary for top-down static sensor).
  * Large z-jitter (keep ±0.1 m).
* **Sampler:** keep **20–30% empty frames** (valuable negatives now that ground is absent).

---

## 5) Label noise handling (quick, effective)

* **Auto-filter GT boxes** during dataset load:

  * Min points inside box (e.g., `>= 15–20` after ground removal).
  * Class-wise physical priors (reasonable L/W/H, z band \~ 0–4 m).
  * Remove boxes outside ROI or with absurd aspect ratios.
* **Loss shaping** (robustness):

  * Wider positives (we set `gaussian_overlap: 0.60`, `min_radius: 3`).
  * Slightly **down-weight classification** vs regression:

    ```yaml
    MODEL.LOSS_CONFIG.loss_weights:
      cls_weight: 0.8
      loc_weight: 2.0
      dir_weight: 0.2
      iou_weight: 0.2
    ```
* If one class is notably noisy, put it in its **own task/head** and give it a **lower cls weight** or sample probability.

---

## 6) Teacher–student “denoise” (optional but strong)

1. Train a **teacher** with the cleanest subset + settings above.
2. Generate **pseudo-labels** on the full set; keep high-confidence only (per-class thresholds).
3. Train a **student** on GT (full weight) + pseudo-labels (reduced weight, 0.5–0.8).
4. Use **EMA** of student as the new teacher.

Because your LiDAR is **static**, you can mine **moving objects** (BEV flow or simple DBSCAN+tracking) to auto-label lots of GSE reliably and feed them as pseudo-labels.

---

## 7) Tracking

* Keep CenterPoint’s **velocity head**; it stabilizes IDs with minimal cost.
* Use **AB3DMOT** for association (<1 ms extra on 4070-mobile).

---

## 8) Training commands (example)

```bash
# Stage A: head-only (freeze backbone in your train loop)
python train.py \
  --cfg_file cfgs/custom_models/cp_pillars_gse.yaml \
  --pretrained_model ../checkpoints/centerpoint_pillars_nuscenes.pth \
  --workers 8 --batch_size 8

# Stage B: unfreeze last block, lower LR
python train.py \
  --cfg_file cfgs/custom_models/cp_pillars_gse.yaml \
  --pretrained_model output/your_run/ckpt_epoch_XX.pth \
  --workers 8 --batch_size 8
```

> Freezing modules: in your training script, set `requires_grad=False` for `vfe`, `map_to_bev`, `backbone_2d` (and BNs) during Stage A; re-enable in Stage B.

---

## 9) Inference hygiene

* Use **AMP** first; once stable, export CP-Pillars to **TensorRT FP16** (clean speedup on Ada).
* Keep voxelization & NMS on GPU; disable test-time aug.
* Tune per-class **score thresholds** (background is cleaner now; you can go a bit lower, e.g., 0.15–0.25).

---

### Quick checklist to avoid common pitfalls

* [x] **One z convention** everywhere (prefer z\_rel with ground≈0).
* [x] Pillar z voxel size **equals** `(z_max − z_min)` (single thick bin).
* [x] Tight ROI.
* [ ] Start **head-only**, then unfreeze last block.
* [ ] **Wider heatmap positives** and slightly lower `cls_weight`.
* [ ] Filter weak GT by **min points** + **physical priors**.
* [ ] Use **copy-paste** and keep **empty frames** for regularization.
* [ ] Add **pseudo-labels** from motion mining if you can.

If you paste your exact ROI bounds and whether you’ll keep negative z or convert to z\_rel, I can drop in a ready-to-run `.yaml` and a tiny dataset filter snippet for your `custom_dataset.py`.

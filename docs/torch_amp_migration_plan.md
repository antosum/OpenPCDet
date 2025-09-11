# A) Implementation strategy (what we’re changing & why)

1. **Move to the unified AMP API**

* Replace any `torch.cuda.amp.autocast` / `torch.cuda.amp.GradScaler` with the modern API:

  * Use **`torch.autocast(device_type="cuda", dtype=...)`** for casting, and
  * Use **`torch.amp.GradScaler("cuda", ...)`** **only for FP16**.
    Rationale: PyTorch’s AMP docs say float16 training is ordinarily **autocast + GradScaler**, while BF16 generally uses **autocast only** (no scaler). CUDA-prefixed AMP APIs are deprecated. ([PyTorch][1])

2. **Expose AMP dtype in config**

* Add a small config block (defaults shown):

  ```yaml
  OPTIMIZATION:
    AMP:
      ENABLED: true
      DTYPE: "fp16"        # or "bf16"
      INIT_SCALE: 65536.0  # optional scaler knobs (fp16 only)
      GROWTH_FACTOR: 2.0
      BACKOFF_FACTOR: 0.5
      GROWTH_INTERVAL: 2000
  ```
* Behavior:

  * **FP16** → autocast(fp16) **+ GradScaler**.
  * **BF16** → autocast(bf16), **no GradScaler**. ([PyTorch][1])

3. **Keep gradient clipping correct under AMP**

* If scaling is active, **unscale before clipping**:

  * `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step()` → `scaler.update()`.
    (This is the official AMP recipe.) ([PyTorch][2])

4. **Add optional `torch.compile`**

* Compile the model **right after construction and before DDP wrapping**:

  * `mode="default"` as baseline; try `mode="reduce-overhead"` for tiny batches; try `mode="max-autotune"` for best speed on steady shapes.
  * Leave `fullgraph=False` (default) so unsupported bits just run eagerly; use `dynamic=True` if shapes vary a lot to avoid recompiles. ([PyTorch Documentation][3])

5. **Touch points in OpenPCDet**

* **`tools/train.py`** (args → cfg, checkpoint save/load) and **training loop helper** (commonly in `pcdet/utils/train_utils.py`). That’s where autocast/backward/clip/step live. ([GitHub][4])

---

# B) Step-by-step plan (how to implement)

1. **Config plumbing**

* Add `OPTIMIZATION.AMP` (above) to configs; default to `ENABLED: true`, `DTYPE: "fp16"`.
* Parse into `use_amp: bool` and `amp_dtype: torch.float16|torch.bfloat16`.

2. **Imports (unified API)**

* Prefer `torch.autocast` and **`torch.amp.GradScaler("cuda", ...)`** (device-explicit, future-proof). ([PyTorch][1])

3. **Create scaler conditionally**

* If `use_amp and DTYPE=="fp16"` → create `GradScaler("cuda", init_scale=..., growth_factor=..., backoff_factor=..., growth_interval=...)`.
* Else `scaler = None`.

4. **Wrap forward + loss with autocast**

* `with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp): loss = ...`

5. **Backward/step paths**

* **If scaler exists (fp16)**:
  `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → (optional) `clip_grad_norm_` → `scaler.step(optimizer)` → `scaler.update()` → `optimizer.zero_grad(set_to_none=True)`. ([PyTorch][2])
* **Else (bf16 or AMP disabled)**:
  `loss.backward()` → (optional) `clip_grad_norm_` → `optimizer.step()` → `optimizer.zero_grad(set_to_none=True)`.

6. **Gradient accumulation (unchanged logic)**

* Only call `scaler.step()/update()` (or `optimizer.step()`) **at the end** of each accumulation window; if you clip, do the `unscale_` + clip right before the step. ([PyTorch][2])

7. **DDP integration**

* Leave `no_sync()` blocks as they are; steps happen only on synced iters (as today).

8. **Validation / test**

* Use `with torch.inference_mode(), torch.autocast(..., enabled=use_amp): ...`
* **No GradScaler** in eval.

9. **Checkpointing**

* On save: if `scaler` exists, include `ckpt["amp_scaler"] = scaler.state_dict()`.
* On resume: if present and scaler exists, `scaler.load_state_dict(...)`. (Absent is fine for older ckpts.)

10. **Integrate `torch.compile` (optional but recommended)**

* After building the model and **before** DDP:

  * Baseline: `torch.compile(model, mode="default")`
  * For small batches: `mode="reduce-overhead"` (CUDA graphs)
  * For steady shapes, performance hunting: `mode="max-autotune"`
  * For varying shapes: add `dynamic=True`.
    (All modes & knobs per docs.) ([PyTorch Documentation][3])

11. **Docs & flags**

* Update README/training help: explain **FP16 uses scaler**, **BF16 doesn’t**, and how to toggle `torch.compile` modes.

12. **CI / smoke tests**

* 1–2-minute GPU smoke runs: AMP off vs fp16 AMP vs bf16 AMP (if GPU supports bf16), with and without `torch.compile`.

---

# C) Checklists & “definition of done”

### Code-change checklist

* [ ] Add `OPTIMIZATION.AMP` config with `ENABLED`, `DTYPE`, and optional scaler knobs.
* [ ] Switch all autocast regions to **`torch.autocast(device_type="cuda", dtype=...)`**. ([PyTorch][1])
* [ ] Instantiate **`torch.amp.GradScaler("cuda", …)`** **only if** `DTYPE=="fp16"` and AMP enabled. ([PyTorch][1])
* [ ] In the train step, if scaler exists: **scale → backward → unscale → (clip) → step → update**; else: standard backward/step. **Unscale before clipping.** ([PyTorch][2])
* [ ] Keep grad-accum logic: step/update only at the end of accumulation. ([PyTorch][2])
* [ ] Eval path uses `inference_mode()` + autocast; **no scaler**.
* [ ] Save/load scaler state with checkpoints when fp16+AMP is active.
* [ ] Optional: compile the model with `torch.compile(...)` **before DDP**; provide a CLI/config to choose `mode` and `dynamic`. ([PyTorch Documentation][3])
* [ ] Touch points updated in **`tools/train.py`** and the training loop helper (`pcdet/utils/train_utils.py`). ([GitHub][4])
* [ ] Update docs/help strings to match.

### Testing checklist

* [ ] **Numerical sanity:** 200 iters on a small slice: FP32 baseline vs fp16 AMP show similar loss trend (no NaNs).
* [ ] **BF16 path:** if hardware supports it, runs stable **without** scaler. ([PyTorch][1])
* [ ] **Clipping correctness:** verify no warnings; spot-check that clipping is after `unscale_`. ([PyTorch][2])
* [ ] **Checkpoint resume:** fp16+AMP run resumes with restored scaler state (scale value is non-default and changes over time).
* [ ] **DDP + accumulation:** 2×GPU smoke run with accumulation>1, confirm steps happen only on sync iters and no scale/NaN issues.
* [ ] **`torch.compile`:** A/B one short run with and without compile; confirm identical metrics and expected speedup (or at least no regression). With `dynamic=True`, confirm minimal recompiles when input shapes vary. ([PyTorch Documentation][3])

### Definition of done

* Training, validation, and resume work with: **AMP off**, **AMP fp16 (with scaler)**, **AMP bf16 (no scaler)**.
* No deprecated `torch.cuda.amp.*` references remain. ([PyTorch][1])
* Optional `torch.compile` path is selectable and safe (falls back gracefully on graph breaks). ([PyTorch Documentation][3])
* Docs updated to reflect new flags/behavior.

---

### Minimal snippets you can allow Codex to emit (if needed)

* **Autocast usage (device-agnostic):**
  `with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp): ...` ([PyTorch][1])

* **FP16 clipping order under AMP:**
  `scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(...); scaler.step(...); scaler.update()` ([PyTorch][2])

* **Compile hook (optional):**
  `model = torch.compile(model, mode="default")  # or 'reduce-overhead' / 'max-autotune', + dynamic=True` ([PyTorch Documentation][3])
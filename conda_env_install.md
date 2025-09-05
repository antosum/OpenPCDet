# Updated OpenPCDet install (editable mode)

```bash
# 0) Create & activate env (example)
conda create -n pcdet118 python=3.10 -y
conda activate pcdet118

# 1) Core deps (match your GPU/driver)
# PyTorch 2.5.1 + CUDA 11.8 (example); adjust if needed
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118

# spconv for CUDA 11.8
pip install spconv-cu118==2.3.8

# utils
pip install ninja numba llvmlite scipy scikit-image tensorboardX tqdm easydict pyyaml SharedArray

# (Wayland-friendly) Open3D
pip install "open3d==0.17.*"

# 2) Clone OpenPCDet
cd ~/projects
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet

# 3) Install PCDet in EDITABLE mode (your working recipe)
pip uninstall -y pcdet || true
rm -rf build *.egg-info pcdet.egg-info
pip install -e . --no-build-isolation --config-settings editable_mode=compat

# 4) Build/verify CUDA ops
python - <<'PY'
import torch, importlib
print("Torch:", torch.__version__, "CUDA available?", torch.cuda.is_available())
mods = [
 "pcdet.ops.iou3d_nms.iou3d_nms_cuda",
 "pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils",
 "pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils",
 "pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils",
]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, "✓")
    except Exception as e:
        print(m, "FAILED ->", e)
PY
```

# Running the demo (Wayland notes)

```bash
# KITTI example
cd tools
python demo.py \
  --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
  --ckpt ../ckpt/pv_rcnn_8369_kitti.pth \
  --data_path ../data/kitti/training/velodyne/000008.bin \
  --ext .bin
```

* On Wayland, **Open3D 0.17** worked for you. If a window fails to open, remove `--save_to_file` (as you did) or set a virtual display (e.g., xvfb) for headless saves.
* If Open3D still complains, try:

  * `export QT_QPA_PLATFORM=xcb` (sometimes helps on Wayland)
  * or run under an x11 session if convenient.

# Optional dataset extras

* If you **don’t** use Argo2, you can skip installing `av2`. If you do need it, pin a version that doesn’t use `np.bool` (or patch to `bool`/`np.bool_`).

# Dev workflow tips

* You’re in editable mode now—Python file changes take effect immediately.
* If you modify any **CUDA/C++** ops under `pcdet/ops/*/src`, clear caches then reimport:

  ```bash
  rm -rf ~/.cache/torch_extensions
  ```
* Keep `spconv` ≥ 2.x (you have 2.3.8) and PyTorch/CUDA aligned.

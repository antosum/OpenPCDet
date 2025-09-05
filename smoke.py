import torch, importlib, pcdet, os, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "CUDA available?", torch.cuda.is_available())
print("PCDet:", getattr(pcdet, "__version__", "unknown"))

# Trigger JIT builds of CUDA ops
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
        
print("All core CUDA ops import ✓")

from pcdet_engine import PCDetEngine
from rich import print
from rich.pretty import Pretty
import numpy as np
import time
from pathlib import Path

try:  # pragma: no cover - optional dependency for visualization
    import rerun as rr
except Exception:  # pragma: no cover - optional dependency
    rr = None

pth = Path("/home/antonin-sumner/data/openpcdet_data/data/geminai_V0/points")

eng = PCDetEngine(
    cfg_file="~/projects/OpenPCDet/tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint.yaml",
    ckpt_path="~/projects/OpenPCDet/output/geminai_models/geminai_cbgs_dyn_pp_centerpoint/vital-cosmos-46/ckpt/best_model.pth",
    device="cuda",
)

print("Engine initialized")

if rr is not None:
    rr.init("pcdet_test_engine", spawn=True)
else:
    print("[yellow]Rerun not available; continuing without live viewer[/yellow]")

# Warmup + profiling of predict
num_warmup = 5
times_ms = []


def _yaw_to_quaternion(yaw: np.ndarray) -> np.ndarray:
    """Convert yaw angles (radians around +Z) to xyzw quaternions."""

    half = 0.5 * yaw
    zeros = np.zeros_like(half)
    return np.stack((zeros, zeros, np.sin(half), np.cos(half)), axis=-1)


def _format_label(class_id: int, score: float) -> str:
    """Return a human-readable label string for Rerun."""

    cls_idx = class_id - 1
    if eng.class_names and 0 <= cls_idx < len(eng.class_names):
        cls_name = eng.class_names[cls_idx]
    else:
        cls_name = str(class_id)
    return f"{cls_name} {score:.2f}"


def _log_frame_to_rerun(frame_idx: int, points: np.ndarray, prediction: dict) -> None:
    if rr is None:
        return

    rr.set_time_sequence("frame", frame_idx)

    if points.size:
        rr.log("world/x", rr.Points3D(points[:, :3]))
    else:
        rr.log("world/x", rr.Points3D(np.empty((0, 3), dtype=np.float32)))

    boxes = prediction.get("boxes_lidar")
    scores = prediction.get("scores")
    labels = prediction.get("labels")

    if boxes is None or scores is None or labels is None or boxes.size == 0:
        rr.log(
            "world/y",
            rr.Boxes3D(
                centers=np.empty((0, 3), dtype=np.float32),
                half_sizes=np.empty((0, 3), dtype=np.float32),
                labels=[],
            ),
        )
        return

    centers = boxes[:, :3]
    half_sizes = boxes[:, 3:6] * 0.5
    quaternions = _yaw_to_quaternion(boxes[:, 6])
    text_labels = [_format_label(int(cls_id), float(score)) for cls_id, score in zip(labels, scores)]

    # Class color coding (consistent per class id)
    # Build a color palette deterministically from class index.
    def _class_color(cidx: int) -> np.ndarray:
        # Simple hash to RGB
        r = (37 * cidx + 77) % 256
        g = (17 * cidx + 151) % 256
        b = (97 * cidx + 13) % 256
        return np.array([r, g, b, 200], dtype=np.uint8)

    # Convert model labels to zero-based class indices
    cls_indices = np.asarray(labels, dtype=np.int32) - 1
    colors = np.stack([_class_color(int(max(idx, 0))) for idx in cls_indices], axis=0) if len(cls_indices) else np.empty((0,4), dtype=np.uint8)

    rr.log(
        "world/y",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            quaternions=rr.Quaternion(xyzw=quaternions),
            labels=text_labels,
            colors=colors,
        ),
    )

# Timed runs
for i, fn in enumerate(pth.glob("*.npy")):
    if i >= 100:
        break
    print(fn)
    x = np.load(fn)
    if i == 0:
        # Warmup runs (not timed)
        for _ in range(num_warmup):
            _ = eng.predict(x)

    t0 = time.perf_counter()
    y = eng.predict(x)
    t1 = time.perf_counter()
    times_ms.append((t1 - t0) * 1000.0)

    _log_frame_to_rerun(i, x, y)

if times_ms:
    arr = np.array(times_ms)
    mean_ms = float(arr.mean())
    min_ms = float(arr.min())
    max_ms = float(arr.max())
    p95_ms = float(np.percentile(arr, 95))
    print(
        f"Predict latency (ms) over {len(arr)} runs: "
        f"mean={mean_ms:.2f}, min={min_ms:.2f}, max={max_ms:.2f}, p95={p95_ms:.2f}"
    )

# print("CFG\n",Pretty(eng.cfg))
# print("Data_CONFIG\n",Pretty(eng.data_config))
# print("Class names \n",Pretty(eng.class_names))

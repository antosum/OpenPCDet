import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    from visual_utils import open3d_vis_utils as V
    OPEN3D = True
except Exception:
    try:
        from visual_utils import visualize_utils as V
        import mayavi.mlab as mlab  # noqa: F401
        OPEN3D = False
    except Exception as e:
        raise RuntimeError(
            "No visualization backend available. Install open3d or mayavi."
        ) from e

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predictions from result.pkl on point clouds")
    parser.add_argument("--cfg_file", type=str, required=True, help="Path to model config YAML used for eval")
    parser.add_argument(
        "--result_pkl", type=str, required=True, help="Path to result.pkl produced by tools/test.py"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index within the result list to visualize"
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of consecutive samples to visualize (use -1 for all)"
    )
    parser.add_argument(
        "--with_gt", action="store_true", help="Overlay ground-truth boxes if available"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    logger = common_utils.create_logger()
    logger.info("Loading dataset in test mode (to fetch raw point clouds)...")

    # Build the test dataloader to access the dataset instance
    # We only need the dataset object, not actual loading/iteration.
    dataset, _dataloader, _sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )

    result_path = Path(args.result_pkl)
    if not result_path.exists():
        raise FileNotFoundError(f"result.pkl not found at {result_path}")

    with open(result_path, "rb") as f:
        det_annos = pickle.load(f)

    total = len(det_annos)
    if total == 0:
        logger.warning("result.pkl contains no predictions.")
        return

    start = max(0, args.start)
    end = total if args.count == -1 else min(total, start + max(0, args.count))
    logger.info(f"Visualizing predictions {start}:{end} of {total} from {result_path}")

    # Helper to fetch GT boxes if requested
    def _get_gt_boxes(frame_id: str):
        try:
            if hasattr(dataset, "get_label"):
                gt_boxes, _gt_names = dataset.get_label(frame_id)
                # Dataset returns [x y z dx dy dz yaw]; visualization expects same ordering
                return gt_boxes.astype(np.float32)
        except Exception:
            return None
        return None

    for i in range(start, end):
        anno = det_annos[i]
        frame_id = str(anno.get("frame_id"))

        # Fetch raw point cloud from dataset using frame_id
        try:
            points = dataset.get_lidar(frame_id)
        except Exception as e:
            logger.error(f"Failed to load point cloud for frame_id={frame_id}: {e}")
            continue

        # Predicted boxes/scores/labels
        pred_boxes = anno.get("boxes_lidar")
        pred_scores = anno.get("score")
        pred_labels = anno.get("pred_labels")

        gt_boxes = _get_gt_boxes(frame_id) if args.with_gt else None

        logger.info(f"Frame {i} id={frame_id}: pred_boxes={0 if pred_boxes is None else len(pred_boxes)}")

        display_texts = []
        if pred_boxes is not None and len(pred_boxes) > 0:
            logger.info("Displayed detections (class | score):")
            num_preds = len(pred_boxes)

            def _ensure_list(values, fill_value=None):
                if values is None:
                    return [fill_value] * num_preds
                if isinstance(values, (list, tuple)):
                    return list(values)
                if hasattr(values, "tolist"):
                    return list(values.tolist())
                return [values] * num_preds

            labels_list = _ensure_list(pred_labels, None)
            scores_list = _ensure_list(pred_scores, None)

            for det_idx, (label, score) in enumerate(zip(labels_list, scores_list)):
                class_name = None
                if label is not None:
                    try:
                        class_idx = int(label) - 1
                        if 0 <= class_idx < len(cfg.CLASS_NAMES):
                            class_name = cfg.CLASS_NAMES[class_idx]
                    except Exception:
                        class_name = None
                if class_name is None:
                    class_name = f"label_{label}"
                formatted_score = f"{float(score):.3f}" if score is not None else "n/a"
                logger.info(f"  #{det_idx:02d}: {class_name} | {formatted_score}")
                display_texts.append(
                    class_name if formatted_score == "n/a" else f"{class_name} {formatted_score}"
                )
        else:
            logger.info("No detections to display.")

        V.draw_scenes(
            points=points,
            gt_boxes=gt_boxes,
            ref_boxes=pred_boxes,
            ref_labels=pred_labels,
            ref_scores=pred_scores,
            ref_names=display_texts if display_texts else None,
        )

        if not OPEN3D:
            # For mayavi backend, ensure blocking window
            mlab.show(stop=True)

    logger.info("Visualization finished.")


if __name__ == "__main__":
    main()

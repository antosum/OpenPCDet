from __future__ import annotations

from typing import Any, Dict


def set_quick_dashboard_keys(wandb_run) -> None:
    """Populate a minimal list of keys to plot/compare in the WANDB UI.

    This writes into run.config under 'dashboard/quick_keys' so users can
    build panels referencing this list without duplicating keys across repos.
    """
    if wandb_run is None:
        return
    quick_keys = [
        'train/loss',
        'train/hm_loss_head_0', 'train/hm_loss_head_1',
        'train/loc_loss_head_0', 'train/loc_loss_head_1',
        'meta_data/learning_rate',
        'valid/simple_bev/mAP@0.5',
        'valid/simple_bev/Belt_loader/AP@0.5',
        'valid/simple_bev/High_loader/AP@0.5',
        'valid/simple_bev/Other_vehicle/AP@0.5',
        'valid/recall/rcnn_0.5',
    ]
    try:
        wandb_run.config.update({'dashboard/quick_keys': quick_keys}, allow_val_change=True)
    except Exception:
        pass


def log_run_basics(wandb_run, cfg, train_set, per_gpu_batch_size: int, total_gpus: int) -> None:
    """Log common run metadata and dataset stats to WANDB config.

    - Effective batch size breakdown
    - Dataset name, size, per-class counts, average boxes per sample
    - Point cloud range for reproducibility
    """
    if wandb_run is None:
        return
    try:
        effective_bs = per_gpu_batch_size * total_gpus
        wandb_run.config.update({
            'train/per_gpu_BS': per_gpu_batch_size,
            'train/num_gpus': total_gpus,
            'train/total_BS': effective_bs,
        }, allow_val_change=True)
    except Exception:
        pass

    # Dataset stats (best-effort; depends on dataset format)
    try:
        class_counts: Dict[str, int] = {}
        total_boxes = 0
        infos = getattr(train_set, 'custom_infos', [])
        for info in infos:
            annos = info.get('annos') if isinstance(info, dict) else None
            if not annos or 'name' not in annos:
                continue
            names = annos['name']
            total_boxes += len(names)
            for n in names:
                class_counts[n] = class_counts.get(n, 0) + 1

        ds_stats: Dict[str, Any] = {
            'dataset/name': str(cfg.DATA_CONFIG.get('DATA_PATH', '')),
            'dataset/train_samples': len(getattr(train_set, 'custom_infos', [])) or len(train_set),
            'dataset/avg_boxes_per_sample': (total_boxes / max(len(infos), 1)) if infos else None,
            'dataset/point_cloud_range': cfg.DATA_CONFIG.get('POINT_CLOUD_RANGE'),
        }
        for k, v in class_counts.items():
            ds_stats[f'dataset/class_counts/{k}'] = v
        wandb_run.config.update(ds_stats, allow_val_change=True)
    except Exception:
        pass


def record_quick_summary(wandb_run, cfg, epochs: int, extra_tag: str) -> None:
    """Write concise summary keys for quick cross-run comparisons."""
    if wandb_run is None:
        return
    try:
        wandb_run.summary['quick/best_key'] = cfg.get('WANDB', {}).get('BEST_KEY')
        wandb_run.summary['quick/epochs'] = epochs
        wandb_run.summary['quick/tag'] = extra_tag
    except Exception:
        pass


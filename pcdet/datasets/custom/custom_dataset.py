import copy
import pickle
import os
from pathlib import Path

import numpy as np
import torch

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        self.custom_infos.extend(custom_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))

    def get_label(self, idx):
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        if not lidar_file.exists():
            raise FileNotFoundError(
                f"Missing points file: {lidar_file} (from lidar_idx='{idx}')"
            )
        point_features = np.load(lidar_file)
        # Pad a zero "timestamp" column if points are 4D (N,4)
        if point_features.ndim == 2 and point_features.shape[1] == 4:
            zeros = np.zeros((point_features.shape[0], 1), dtype=point_features.dtype)
            point_features = np.concatenate([point_features, zeros], axis=1)
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def simple_eval(eval_det_annos, eval_gt_annos, eval_class_names, iou_thresh=0.5, use_bev=True, enable_advanced_metrics=True):
            from pcdet.ops.iou3d_nms import iou3d_nms_utils as iou
            import numpy as np
            import torch
            try:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                sklearn_available = True
            except ImportError:
                sklearn_available = False
                print("Warning: scikit-learn not available. Advanced metrics will be limited.")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Helper function: Compute IoU matrix between detection and GT boxes
            def _compute_iou_matrix(det_boxes, gt_boxes, use_bev=True):
                if use_bev:
                    return iou.boxes_iou_bev(
                        torch.from_numpy(det_boxes).to(device),
                        torch.from_numpy(gt_boxes).to(device)
                    ).cpu().numpy()
                else:
                    return iou.boxes_iou3d_gpu(
                        torch.from_numpy(det_boxes).to(device),
                        torch.from_numpy(gt_boxes).to(device)
                    ).cpu().numpy()

            # Helper function: Perform greedy matching for given IoU threshold
            def _match_detections_to_gt(ious, iou_thresh):
                det_used = np.zeros(ious.shape[0], dtype=bool)
                gt_used = np.zeros(ious.shape[1], dtype=bool)
                
                # Iterate in descending IoU order
                pairs = [(i, j, ious[i, j]) for i in range(ious.shape[0]) for j in range(ious.shape[1])]
                pairs.sort(key=lambda x: x[2], reverse=True)
                
                for i, j, v in pairs:
                    if v < iou_thresh:
                        break
                    if det_used[i] or gt_used[j]:
                        continue
                    det_used[i] = True
                    gt_used[j] = True
                
                return det_used, gt_used

            # Helper function: Compute AP from scores and matches
            def _compute_ap_from_scores_matches(scores, matches, total_gt):
                if scores.size == 0:
                    return 0.0
                
                order = scores.argsort()[::-1]
                matches_ord = matches[order]
                tp_cum = np.cumsum(matches_ord)
                fp_cum = np.cumsum(1 - matches_ord)
                recalls = tp_cum / max(total_gt, 1)
                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
                
                # Add boundary points for AP calculation
                mrec = np.concatenate(([0.0], recalls, [1.0]))
                mpre = np.concatenate(([0.0], precisions, [0.0]))
                mpre = np.maximum.accumulate(mpre[::-1])[::-1]
                ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
                
                return ap

            # Helper function: Compute multi-threshold AP (COCO-style)
            def _compute_multi_threshold_ap(scores, det_boxes, gt_boxes, iou_thresholds, use_bev=True):
                ap_values = []
                
                for thresh in iou_thresholds:
                    if det_boxes.size == 0 or gt_boxes.size == 0:
                        ap_values.append(0.0)
                        continue
                    
                    ious = _compute_iou_matrix(det_boxes, gt_boxes, use_bev)
                    det_used, gt_used = _match_detections_to_gt(ious, thresh)
                    
                    total_gt = gt_boxes.shape[0]
                    matches = det_used.astype(float)
                    ap = _compute_ap_from_scores_matches(scores, matches, total_gt)
                    ap_values.append(ap)
                
                return np.array(ap_values)

            # Helper function: Generate PR curve data and compute AUPRC
            def _generate_pr_curve_data(scores, matches, total_gt):
                if scores.size == 0:
                    return np.array([]), np.array([]), np.array([]), 0.0
                
                # Check if we have any positive matches
                has_positives = np.sum(matches) > 0
                
                if not has_positives:
                    # No positive matches - return sensible default values
                    precision = np.array([0.0, 1.0])  # Precision drops from 1 to 0
                    recall = np.array([0.0, 1.0])    # Recall goes from 0 to 1
                    thresholds = np.array([1.0, 0.0])   # Thresholds from high to low
                    auprc = 0.0
                    return precision, recall, thresholds, auprc
                
                if sklearn_available:
                    try:
                        # Use sklearn's precision_recall_curve for accurate PR curve
                        precision, recall, thresholds = precision_recall_curve(matches, scores)
                        auprc = average_precision_score(matches, scores)
                    except Exception as e:
                        # Fallback to custom implementation if sklearn fails
                        print(f"Warning: sklearn PR curve failed ({e}), using fallback")
                        return _generate_pr_curve_data_fallback(scores, matches, total_gt)
                else:
                    # Use fallback implementation
                    return _generate_pr_curve_data_fallback(scores, matches, total_gt)
                
                return precision, recall, thresholds, auprc
            
            # Fast PR curve generation (optimized for performance)
            def _generate_pr_curve_data_fast(scores, matches, total_gt):
                if scores.size == 0:
                    return np.array([]), np.array([]), np.array([]), 0.0
                
                # Check if we have any positive matches
                has_positives = np.sum(matches) > 0
                
                if not has_positives:
                    # No positive matches - return sensible default values
                    precision = np.array([0.0, 1.0])
                    recall = np.array([0.0, 1.0])
                    thresholds = np.array([1.0, 0.0])
                    auprc = 0.0
                    return precision, recall, thresholds, auprc
                
                # Use fast custom implementation (avoid sklearn overhead)
                order = scores.argsort()[::-1]
                matches_ord = matches[order]
                tp_cum = np.cumsum(matches_ord)
                fp_cum = np.cumsum(1 - matches_ord)
                recalls = tp_cum / max(total_gt, 1)
                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
                
                # Add boundary points
                precision = np.concatenate(([1.0], precisions, [0.0]))
                recall = np.concatenate(([0.0], recalls, [1.0]))
                thresholds = np.concatenate(([scores[order[0]] if len(order) > 0 else 1.0], scores[order], [0.0]))
                
                # Compute AUPRC using trapezoidal rule
                auprc = np.trapz(precision, recall) if len(recall) > 1 else 0.0
                
                return precision, recall, thresholds, auprc

            # Fallback implementation for PR curve generation
            def _generate_pr_curve_data_fallback(scores, matches, total_gt):
                order = scores.argsort()[::-1]
                matches_ord = matches[order]
                tp_cum = np.cumsum(matches_ord)
                fp_cum = np.cumsum(1 - matches_ord)
                recalls = tp_cum / max(total_gt, 1)
                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
                
                precision = np.concatenate(([1.0], precisions, [0.0]))
                recall = np.concatenate(([0.0], recalls, [1.0]))
                thresholds = np.concatenate(([scores[order[0]] if len(order) > 0 else 1.0], scores[order], [0.0]))
                
                # Compute AUPRC using trapezoidal rule
                auprc = np.trapz(precision, recall) if len(recall) > 1 else 0.0
                
                return precision, recall, thresholds, auprc

            # Helper function: Find optimal F1 threshold
            def _find_optimal_f1_threshold(scores, matches, total_gt, threshold_range=np.linspace(0.1, 0.7, 13)):
                if scores.size == 0:
                    return 0.5, 0.0
                
                # Check if we have any positive matches
                has_positives = np.sum(matches) > 0
                if not has_positives:
                    # No positive matches - return default values
                    return 0.5, 0.0
                
                best_f1 = 0.0
                best_threshold = 0.5
                
                for thresh in threshold_range:
                    pred_binary = (scores >= thresh).astype(int)
                    tp = np.sum((pred_binary == 1) & (matches == 1))
                    fp = np.sum((pred_binary == 1) & (matches == 0))
                    fn = np.sum((pred_binary == 0) & (matches == 1))
                    
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
                
                return best_threshold, best_f1

            # Helper function: Compute center error (BEV)
            def _compute_center_error(det_boxes, gt_boxes, matches):
                if det_boxes.size == 0 or gt_boxes.size == 0:
                    return np.array([])
                
                # Extract BEV centers (x, y coordinates)
                det_centers = det_boxes[:, :2]  # [x, y]
                gt_centers = gt_boxes[:, :2]    # [x, y]
                
                # Only compute for matched detections
                matched_indices = np.where(matches > 0)[0]
                if len(matched_indices) == 0:
                    return np.array([])
                
                # For each matched detection, find the corresponding GT box
                # This is a simplified approach - assumes 1:1 matching
                center_errors = []
                gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
                
                for det_idx in matched_indices:
                    det_center = det_centers[det_idx]
                    
                    # Find closest unmatched GT box
                    distances = np.linalg.norm(gt_centers - det_center, axis=1)
                    # Set already matched GT boxes to infinity
                    distances[gt_used] = np.inf
                    
                    if np.min(distances) == np.inf:
                        continue  # No unmatched GT boxes
                    
                    gt_idx = np.argmin(distances)
                    gt_used[gt_idx] = True
                    
                    # Compute Euclidean distance in BEV
                    error = np.linalg.norm(det_center - gt_centers[gt_idx])
                    center_errors.append(error)
                
                return np.array(center_errors)

            # Helper function: Compute size error
            def _compute_size_error(det_boxes, gt_boxes, matches):
                if det_boxes.size == 0 or gt_boxes.size == 0:
                    return np.array([])
                
                # Extract dimensions (l, w, h) - assuming format [x, y, z, l, w, h, ...]
                det_dims = det_boxes[:, 3:6]  # [l, w, h]
                gt_dims = gt_boxes[:, 3:6]    # [l, w, h]
                
                # Only compute for matched detections
                matched_indices = np.where(matches > 0)[0]
                if len(matched_indices) == 0:
                    return np.array([])
                
                size_errors = []
                gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
                
                for det_idx in matched_indices:
                    det_dim = det_dims[det_idx]
                    
                    # Find closest unmatched GT box by center distance
                    det_center = det_boxes[det_idx, :2]
                    gt_centers = gt_boxes[:, :2]
                    distances = np.linalg.norm(gt_centers - det_center, axis=1)
                    distances[gt_used] = np.inf
                    
                    if np.min(distances) == np.inf:
                        continue
                    
                    gt_idx = np.argmin(distances)
                    gt_used[gt_idx] = True
                    
                    # Compute relative size error for each dimension
                    gt_dim = gt_dims[gt_idx]
                    # Avoid division by zero
                    rel_errors = np.abs(det_dim - gt_dim) / np.maximum(gt_dim, 0.01)
                    # Use mean relative error across dimensions
                    size_error = np.mean(rel_errors)
                    size_errors.append(size_error)
                
                return np.array(size_errors)

            # Helper function: Compute yaw error
            def _compute_yaw_error(det_boxes, gt_boxes, matches):
                if det_boxes.size == 0 or gt_boxes.size == 0:
                    return np.array([])
                
                # Extract yaw angles - assuming format [x, y, z, l, w, h, yaw, ...]
                det_yaws = det_boxes[:, 6]  # yaw
                gt_yaws = gt_boxes[:, 6]    # yaw
                
                # Only compute for matched detections
                matched_indices = np.where(matches > 0)[0]
                if len(matched_indices) == 0:
                    return np.array([])
                
                yaw_errors = []
                gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
                
                for det_idx in matched_indices:
                    det_yaw = det_yaws[det_idx]
                    
                    # Find closest unmatched GT box by center distance
                    det_center = det_boxes[det_idx, :2]
                    gt_centers = gt_boxes[:, :2]
                    distances = np.linalg.norm(gt_centers - det_center, axis=1)
                    distances[gt_used] = np.inf
                    
                    if np.min(distances) == np.inf:
                        continue
                    
                    gt_idx = np.argmin(distances)
                    gt_used[gt_idx] = True
                    
                    # Compute angular difference (handle circular nature of angles)
                    gt_yaw = gt_yaws[gt_idx]
                    yaw_diff = np.abs(det_yaw - gt_yaw)
                    # Take the minimum of the difference and 2π - difference
                    yaw_error = np.minimum(yaw_diff, 2 * np.pi - yaw_diff)
                    # Convert to degrees
                    yaw_error_deg = np.degrees(yaw_error)
                    yaw_errors.append(yaw_error_deg)
                
                return np.array(yaw_errors)

            # Collect all detection data for advanced metrics
            all_detections = {
                c: {'det_boxes': [], 'gt_boxes': [], 'scores': [], 'matches': []}
                for c in eval_class_names
            }

            # Basic metrics collection (original logic)
            metrics = {
                c: {'TP': 0, 'FP': 0, 'FN': 0, 'scores': [], 'matches': []}
                for c in eval_class_names
            }

            for det, gt in zip(eval_det_annos, eval_gt_annos):
                # Per-class matching
                for c in eval_class_names:
                    det_mask = (det['name'] == c) if isinstance(det['name'], np.ndarray) else np.array(det['name']) == c
                    gt_mask = (gt['name'] == c) if isinstance(gt['name'], np.ndarray) else np.array(gt['name']) == c
                    det_boxes = det['boxes_lidar'][det_mask]
                    gt_boxes = gt['gt_boxes_lidar'][gt_mask]

                    if det_boxes.size == 0 and gt_boxes.size == 0:
                        continue
                    if det_boxes.size == 0 and gt_boxes.size > 0:
                        metrics[c]['FN'] += int(gt_boxes.shape[0])
                        all_detections[c]['gt_boxes'].append(gt_boxes)
                        continue
                    if det_boxes.size > 0 and gt_boxes.size == 0:
                        scores = det['score'][det_mask]
                        metrics[c]['scores'].extend(scores.tolist())
                        metrics[c]['matches'].extend([0] * scores.shape[0])
                        metrics[c]['FP'] += int(det_boxes.shape[0])
                        all_detections[c]['det_boxes'].append(det_boxes)
                        all_detections[c]['scores'].extend(scores.tolist())
                        all_detections[c]['matches'].extend([0] * scores.shape[0])
                        continue

                    # Store data for advanced metrics
                    all_detections[c]['det_boxes'].append(det_boxes)
                    all_detections[c]['gt_boxes'].append(gt_boxes)
                    all_detections[c]['scores'].extend(det['score'][det_mask].tolist())

                    # Compute IoU and perform matching
                    ious = _compute_iou_matrix(det_boxes, gt_boxes, use_bev)
                    det_used, gt_used = _match_detections_to_gt(ious, iou_thresh)

                    # Update basic metrics
                    metrics[c]['TP'] += int(det_used.sum())
                    metrics[c]['FP'] += int((~det_used).sum())
                    metrics[c]['FN'] += int((~gt_used).sum())
                    metrics[c]['scores'].extend(det['score'][det_mask].tolist())
                    metrics[c]['matches'].extend(det_used.astype(float).tolist())
                    all_detections[c]['matches'].extend(det_used.astype(float).tolist())

            # Compute metrics per class
            result_lines = []
            flat_result = {}
            ap_list = []
            
            # Define IoU thresholds for COCO-style evaluation - DISABLED for performance
            # coco_iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, ..., 0.95]
            
            for c in eval_class_names:
                TP = metrics[c]['TP']; FP = metrics[c]['FP']; FN = metrics[c]['FN']
                scores = np.array(metrics[c]['scores'])
                matches = np.array(metrics[c]['matches'])
                total_gt = TP + FN
                
                prec = TP / max(TP + FP, 1)
                rec = TP / max(TP + FN, 1)

                # Basic AP@0.5
                ap_05 = _compute_ap_from_scores_matches(scores, matches, total_gt)
                ap_list.append(ap_05)

                prefix = f"simple_{'bev' if use_bev else '3d'}/{c}"
                
                # Store basic metrics
                flat_result[f"{prefix}/AP@0.5"] = float(ap_05)
                flat_result[f"{prefix}/precision"] = float(prec)
                flat_result[f"{prefix}/recall"] = float(rec)
                flat_result[f"{prefix}/TP"] = float(TP)
                flat_result[f"{prefix}/FP"] = float(FP)
                flat_result[f"{prefix}/FN"] = float(FN)

                # Advanced metrics (if enabled)
                if enable_advanced_metrics and scores.size > 0:
                    # Prepare data for advanced metrics
                    det_boxes_all = np.vstack(all_detections[c]['det_boxes']) if all_detections[c]['det_boxes'] else np.array([])
                    gt_boxes_all = np.vstack(all_detections[c]['gt_boxes']) if all_detections[c]['gt_boxes'] else np.array([])
                    scores_all = np.array(all_detections[c]['scores'])
                    matches_all = np.array(all_detections[c]['matches'])

                    # 1) Stricter overlap metrics - AP@0.7 DISABLED for performance
                    # if det_boxes_all.size > 0 and gt_boxes_all.size > 0:
                    #     ious_07 = _compute_iou_matrix(det_boxes_all, gt_boxes_all, use_bev)
                    #     det_used_07, gt_used_07 = _match_detections_to_gt(ious_07, 0.7)
                    #     matches_07 = det_used_07.astype(float)
                    #     ap_07 = _compute_ap_from_scores_matches(scores_all, matches_07, gt_boxes_all.shape[0])
                    # else:
                    #     ap_07 = 0.0
                    # flat_result[f"{prefix}/AP@0.7"] = float(ap_07)
                    ap_07 = 0.0  # Placeholder for disabled metric

                    # COCO-style mAP@[0.5:0.95] - DISABLED for performance
                    # if det_boxes_all.size > 0 and gt_boxes_all.size > 0:
                    #     ap_values = _compute_multi_threshold_ap(scores_all, det_boxes_all, gt_boxes_all, coco_iou_thresholds, use_bev)
                    #     map_coco = np.mean(ap_values)
                    # else:
                    #     ap_values = np.zeros(len(coco_iou_thresholds))
                    #     map_coco = 0.0
                    # flat_result[f"{prefix}/mAP@[0.5:0.95]"] = float(map_coco)
                    map_coco = 0.0  # Placeholder for disabled metric

                    # 2) Operating-point metrics
                    # PR curve and AUPRC - OPTIMIZED for performance
                    precision_curve, recall_curve, thresh_curve, auprc = _generate_pr_curve_data_fast(scores_all, matches_all, total_gt)
                    flat_result[f"{prefix}/AUPRC"] = float(auprc)

                    # Optimal F1 threshold - OPTIMIZED for performance
                    best_f1_thresh, best_f1 = _find_optimal_f1_threshold(scores_all, matches_all, total_gt, threshold_range=np.linspace(0.1, 0.7, 7))  # Reduced from 13 to 7 thresholds
                    flat_result[f"{prefix}/F1@best"] = float(best_f1)
                    flat_result[f"{prefix}/best_threshold"] = float(best_f1_thresh)

                    # 3) Geometric accuracy metrics
                    # Center error (BEV)
                    center_errors = _compute_center_error(det_boxes_all, gt_boxes_all, matches_all)
                    if len(center_errors) > 0:
                        flat_result[f"{prefix}/center_error_mean"] = float(np.mean(center_errors))
                        flat_result[f"{prefix}/center_error_std"] = float(np.std(center_errors))
                        flat_result[f"{prefix}/center_error_median"] = float(np.median(center_errors))
                    else:
                        flat_result[f"{prefix}/center_error_mean"] = 0.0
                        flat_result[f"{prefix}/center_error_std"] = 0.0
                        flat_result[f"{prefix}/center_error_median"] = 0.0

                    # Size error
                    size_errors = _compute_size_error(det_boxes_all, gt_boxes_all, matches_all)
                    if len(size_errors) > 0:
                        flat_result[f"{prefix}/size_error_mean"] = float(np.mean(size_errors))
                        flat_result[f"{prefix}/size_error_std"] = float(np.std(size_errors))
                        flat_result[f"{prefix}/size_error_median"] = float(np.median(size_errors))
                    else:
                        flat_result[f"{prefix}/size_error_mean"] = 0.0
                        flat_result[f"{prefix}/size_error_std"] = 0.0
                        flat_result[f"{prefix}/size_error_median"] = 0.0

                    # Yaw error
                    yaw_errors = _compute_yaw_error(det_boxes_all, gt_boxes_all, matches_all)
                    if len(yaw_errors) > 0:
                        flat_result[f"{prefix}/yaw_error_mean"] = float(np.mean(yaw_errors))
                        flat_result[f"{prefix}/yaw_error_std"] = float(np.std(yaw_errors))
                        flat_result[f"{prefix}/yaw_error_median"] = float(np.median(yaw_errors))
                    else:
                        flat_result[f"{prefix}/yaw_error_mean"] = 0.0
                        flat_result[f"{prefix}/yaw_error_std"] = 0.0
                        flat_result[f"{prefix}/yaw_error_median"] = 0.0

                    # Store PR curve data (for potential visualization)
                    if len(precision_curve) > 0:
                        flat_result[f"{prefix}/pr_curve_precision"] = precision_curve.tolist()
                        flat_result[f"{prefix}/pr_curve_recall"] = recall_curve.tolist()
                        if len(thresh_curve) > 0:
                            flat_result[f"{prefix}/pr_curve_thresholds"] = thresh_curve.tolist()

                elif enable_advanced_metrics:
                    # Handle case with no detections
                    flat_result[f"{prefix}/AP@0.7"] = 0.0
                    flat_result[f"{prefix}/mAP@[0.5:0.95]"] = 0.0
                    flat_result[f"{prefix}/AUPRC"] = 0.0
                    flat_result[f"{prefix}/F1@best"] = 0.0
                    flat_result[f"{prefix}/best_threshold"] = 0.5

                # Build result line for this class
                if enable_advanced_metrics:
                    # Initialize variables for advanced metrics
                    ap_07 = flat_result.get(f"{prefix}/AP@0.7", 0.0)
                    map_coco = flat_result.get(f"{prefix}/mAP@[0.5:0.95]", 0.0)
                    best_f1 = flat_result.get(f"{prefix}/F1@best", 0.0)
                    auprc = flat_result.get(f"{prefix}/AUPRC", 0.0)
                    center_err = flat_result.get(f"{prefix}/center_error_mean", 0.0)
                    size_err = flat_result.get(f"{prefix}/size_error_mean", 0.0)
                    yaw_err = flat_result.get(f"{prefix}/yaw_error_mean", 0.0)
                    
                    result_lines.append(
                        f"{c}: AP@0.5={ap_05:.3f} F1@best={best_f1:.3f} AUPRC={auprc:.3f} "
                        f"Center={center_err:.2f}m Size={size_err:.3f} Yaw={yaw_err:.1f}° (TP={TP} FP={FP} FN={FN})"
                    )
                else:
                    result_lines.append(
                        f"{c}: AP@0.5={ap_05:.3f} P={prec:.3f} R={rec:.3f} (TP={TP} FP={FP} FN={FN})"
                    )

            # Overall metrics
            mAP_05 = float(np.mean(ap_list)) if ap_list else 0.0
            prefix = f"simple_{'bev' if use_bev else '3d'}"
            flat_result[f"{prefix}/mAP@0.5"] = mAP_05

            # Advanced overall metrics
            if enable_advanced_metrics:
                # Overall mAP@0.7 - DISABLED for performance
                # ap_07_list = [flat_result.get(f"{prefix}/{c}/AP@0.7", 0.0) for c in eval_class_names]
                # map_07 = float(np.mean(ap_07_list)) if ap_07_list else 0.0
                # flat_result[f"{prefix}/mAP@0.7"] = map_07
                map_07 = 0.0  # Placeholder for disabled metric

                # Overall COCO mAP@[0.5:0.95] - DISABLED for performance
                # coco_map_list = [flat_result.get(f"{prefix}/{c}/mAP@[0.5:0.95]", 0.0) for c in eval_class_names]
                # overall_coco_map = float(np.mean(coco_map_list)) if coco_map_list else 0.0
                # flat_result[f"{prefix}/mAP@[0.5:0.95]"] = overall_coco_map
                overall_coco_map = 0.0  # Placeholder for disabled metric

                # Overall F1 and AUPRC
                f1_list = [flat_result.get(f"{prefix}/{c}/F1@best", 0.0) for c in eval_class_names]
                auprc_list = [flat_result.get(f"{prefix}/{c}/AUPRC", 0.0) for c in eval_class_names]
                overall_f1 = float(np.mean(f1_list)) if f1_list else 0.0
                overall_auprc = float(np.mean(auprc_list)) if auprc_list else 0.0
                flat_result[f"{prefix}/F1@best"] = overall_f1
                flat_result[f"{prefix}/AUPRC"] = overall_auprc

                # Overall geometric metrics
                center_err_list = [flat_result.get(f"{prefix}/{c}/center_error_mean", 0.0) for c in eval_class_names]
                size_err_list = [flat_result.get(f"{prefix}/{c}/size_error_mean", 0.0) for c in eval_class_names]
                yaw_err_list = [flat_result.get(f"{prefix}/{c}/yaw_error_mean", 0.0) for c in eval_class_names]
                
                overall_center_err = float(np.mean(center_err_list)) if center_err_list else 0.0
                overall_size_err = float(np.mean(size_err_list)) if size_err_list else 0.0
                overall_yaw_err = float(np.mean(yaw_err_list)) if yaw_err_list else 0.0
                
                flat_result[f"{prefix}/center_error_mean"] = overall_center_err
                flat_result[f"{prefix}/size_error_mean"] = overall_size_err
                flat_result[f"{prefix}/yaw_error_mean"] = overall_yaw_err

            # Build result string
            if enable_advanced_metrics:
                header = f"Advanced {'BEV' if use_bev else '3D'} metrics @IoU={iou_thresh:.2f}"
                result_str = header + "\n"
                result_str += f"mAP@0.5={mAP_05:.3f} F1@best={overall_f1:.3f} AUPRC={overall_auprc:.3f}\n"
                result_str += f"Center Error={overall_center_err:.2f}m Size Error={overall_size_err:.3f} Yaw Error={overall_yaw_err:.1f}°\n"
                result_str += "\n" + "\n".join(result_lines)
            else:
                header = f"Simple {'BEV' if use_bev else '3D'} metrics @IoU={iou_thresh:.2f}"
                result_str = header + "\n" + f"mAP@0.5={mAP_05:.3f}\n" + "\n".join(result_lines)

            return result_str, flat_result

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        metric = kwargs.get('eval_metric', 'kitti')
        if metric == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
            return ap_result_str, ap_dict
        elif metric == 'simple_bev':
            return simple_eval(eval_det_annos, eval_gt_annos, class_names, iou_thresh=kwargs.get('iou_thresh', 0.5), use_bev=True)
        elif metric == 'simple_3d':
            return simple_eval(eval_det_annos, eval_gt_annos, class_names, iou_thresh=kwargs.get('iou_thresh', 0.5), use_bev=False)
        else:
            raise NotImplementedError(f"Unknown eval_metric: {metric}")

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom',
        )

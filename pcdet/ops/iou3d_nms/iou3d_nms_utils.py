"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch

from ...utils import common_utils
from . import iou3d_nms_cuda


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    # Cast to float32 for CUDA kernels (bf16/fp16 inputs may segfault)
    a = boxes_a.contiguous().to(dtype=torch.float32)
    b = boxes_b.contiguous().to(dtype=torch.float32)
    ans_iou = torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    iou3d_nms_cuda.boxes_iou_bev_gpu(a, b, ans_iou)
    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    a = boxes_a.contiguous().to(dtype=torch.float32)
    b = boxes_b.contiguous().to(dtype=torch.float32)

    # height overlap
    boxes_a_height_max = (a[:, 2] + a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (a[:, 2] - a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (b[:, 2] + b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (b[:, 2] - b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(a.contiguous(), b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (a[:, 3] * a[:, 4] * a[:, 5]).view(-1, 1)
    vol_b = (b[:, 3] * b[:, 4] * b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d

def boxes_aligned_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N,)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    a = boxes_a.contiguous().to(dtype=torch.float32)
    b = boxes_b.contiguous().to(dtype=torch.float32)

    # height overlap
    boxes_a_height_max = (a[:, 2] + a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (a[:, 2] - a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (b[:, 2] + b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (b[:, 2] - b[:, 5] / 2).view(-1, 1)

    # bev overlap
    overlaps_bev = torch.zeros((a.shape[0], 1), device=a.device, dtype=torch.float32)  # (N, 1)
    iou3d_nms_cuda.boxes_aligned_overlap_bev_gpu(a.contiguous(), b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (a[:, 3] * a[:, 4] * a[:, 5]).view(-1, 1)
    vol_b = (b[:, 3] * b[:, 4] * b[:, 5]).view(-1, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous().to(dtype=torch.float32)
    # Kernel expects a CPU LongTensor buffer for keep indices
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous().to(dtype=torch.float32)

    # Kernel expects a CPU LongTensor buffer for keep indices
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def paired_boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    a = boxes_a.contiguous().to(dtype=torch.float32)
    b = boxes_b.contiguous().to(dtype=torch.float32)

    # height overlap
    boxes_a_height_max = (a[:, 2] + a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (a[:, 2] - a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (b[:, 2] + b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (b[:, 2] - b[:, 5] / 2).view(-1, 1)

    # bev overlap
    overlaps_bev = torch.zeros((a.shape[0], 1), device=a.device, dtype=torch.float32)  # (N, 1)
    iou3d_nms_cuda.paired_boxes_overlap_bev_gpu(a.contiguous(), b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (a[:, 3] * a[:, 4] * a[:, 5]).view(-1, 1)
    vol_b = (b[:, 3] * b[:, 4] * b[:, 5]).view(-1, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d.view(-1)

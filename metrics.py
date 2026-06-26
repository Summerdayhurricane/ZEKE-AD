import cv2
import numpy as np
import torch
from torchmetrics import AUROC

from utils import normalize


def compute_mask_iou(pred_mask, gt_mask):
    """
    Strictly align with the IoU computation in test_gaussian_auroc_fast.py
    calculate_iou_with_mask: IoU = intersection_area / union_area.
    """
    intersection = cv2.bitwise_and(pred_mask, gt_mask)
    union = cv2.bitwise_or(pred_mask, gt_mask)
    intersection_area = np.sum(intersection > 0)
    union_area = np.sum(union > 0)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


class BinaryAUROCIoUMetrics:
    """
    Aggregator for image-level AUROC and segmentation IoU metrics.

    Parameters
    ----------
    device : torch.device or str
        Device on which the AUROC metric lives.
    iou_thresholds : list[float]
        Thresholds for IoU-Acc reporting.
    binary_threshold : float
        Threshold used to binarize the normalized anomaly map.
    """

    def __init__(self, device, iou_thresholds=(0.2, 0.4, 0.6), binary_threshold=0.5):
        self.auroc_metric = AUROC(task="binary").to(device)
        self.iou_thresholds = list(iou_thresholds)
        self.binary_threshold = binary_threshold
        self.iou_results = {}

    def update_auroc(self, image_score, label_gt):
        """Update image-level AUROC with a single sample score and ground-truth label."""
        self.auroc_metric.update(image_score, label_gt)

    def add_iou(self, anomaly_map, gt_mask, cls_name):
        """
        Compute and store the mask IoU for a single anomalous sample.

        Parameters
        ----------
        anomaly_map : np.ndarray
            HxW anomaly score map (on CPU).
        gt_mask : np.ndarray
            HxW ground-truth mask.
        cls_name : str
            Class name used for per-class aggregation.

        Returns
        -------
        iou : float
            Computed IoU for this sample.
        """
        anomaly_map_norm = normalize(anomaly_map)

        # Binarize prediction map
        if anomaly_map_norm.max() <= 1.0:
            binary_pred = (anomaly_map_norm > self.binary_threshold).astype(np.uint8)
        else:
            binary_pred = (anomaly_map_norm > self.binary_threshold * 255).astype(np.uint8)

        # Binarize ground truth
        if gt_mask.max() > 1:
            gt_binary = (gt_mask > 127).astype(np.uint8)
        else:
            gt_binary = (gt_mask > 0.5).astype(np.uint8)

        iou = compute_mask_iou(binary_pred, gt_binary)
        self.iou_results.setdefault(cls_name, []).append(iou)
        return iou

    def compute(self):
        """
        Compute final AUROC and aggregate IoU statistics.

        Returns
        -------
        dict
            Compatible with the original `seed_result` structure:
            {'auroc': float, 'iou': {...}}
        """
        total_auroc = self.auroc_metric.compute().item()

        all_ious = []
        per_class_mean = {}
        for cls_name_val, iou_list in self.iou_results.items():
            if len(iou_list) > 0:
                cls_mean = np.mean(iou_list)
                per_class_mean[cls_name_val] = cls_mean
                all_ious.extend(iou_list)

        if len(all_ious) > 0:
            sample_mean_iou = np.mean(all_ious)
            class_mean_iou = np.mean(list(per_class_mean.values()))
            eval_classes = len(per_class_mean)
            total_anomaly_samples = len(all_ious)
        else:
            sample_mean_iou = 0.0
            class_mean_iou = 0.0
            eval_classes = 0
            total_anomaly_samples = 0

        iou_gt_counts = {th: 0 for th in self.iou_thresholds}
        for iou in all_ious:
            for th in self.iou_thresholds:
                if iou > th:
                    iou_gt_counts[th] += 1

        iou_acc = {
            th: iou_gt_counts[th] / total_anomaly_samples if total_anomaly_samples > 0 else 0.0
            for th in self.iou_thresholds
        }

        return {
            'auroc': total_auroc,
            'iou': {
                'mean_iou': sample_mean_iou,
                'class_mean_iou': class_mean_iou,
                'eval_classes': eval_classes,
                'total_anomaly_samples': total_anomaly_samples,
                'iou_acc': iou_acc
            }
        }

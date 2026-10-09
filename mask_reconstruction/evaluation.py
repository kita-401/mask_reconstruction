"""インスタンスセグメンテーションの評価指標．"""
import numpy as np

METRICS = ("miou", "prec", "rec", "over", "under")


def evaluate_masks(pred_masks, gt_masks, iou_thresh=0.5, overlap_thresh=0.2):
    """
    mIoU      : 各GTに対する最大IoUの平均
    Precision : IoU >= iou_thresh を満たすGT数 / 予測数
    Recall    : IoU >= iou_thresh を満たすGT数 / GT数
    Over-seg  : 面積の overlap_thresh 超を2つ以上の予測に覆われたGTの数
    Under-seg : 面積の overlap_thresh 超が2つ以上のGTにまたがる予測の数
    """
    gts = [g for g in gt_masks if np.sum(g) > 0]
    N, M = len(gts), len(pred_masks)
    if M == 0:
        return dict(miou=0.0, prec=0.0, rec=0.0, over=0, under=0)
    if N == 0:
        return dict(miou=0.0, prec=0.0, rec=0.0, over=0, under=M)

    iou = np.zeros((N, M))
    inter_gt = np.zeros((N, M))
    inter_pred = np.zeros((N, M))
    for i, g in enumerate(gts):
        g_area = np.sum(g)
        for j, p in enumerate(pred_masks):
            p_area = np.sum(p)
            inter = np.sum(g & p)
            union = np.sum(g | p)
            iou[i, j] = inter / union if union > 0 else 0
            inter_gt[i, j] = inter / g_area if g_area > 0 else 0
            inter_pred[i, j] = inter / p_area if p_area > 0 else 0

    best = np.max(iou, axis=1)
    tp = np.sum(best >= iou_thresh)
    return dict(
        miou=float(np.mean(best)),
        prec=float(tp / M),
        rec=float(tp / N),
        over=int(np.sum(np.sum(inter_gt > overlap_thresh, axis=1) >= 2)),
        under=int(np.sum(np.sum(inter_pred > overlap_thresh, axis=0) >= 2)),
    )

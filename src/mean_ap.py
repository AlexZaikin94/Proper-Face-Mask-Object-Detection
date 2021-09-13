import torch
import torchvision


def calculate_map(target_boxes, pred_boxes, pred_scores, threshold=0.5):
    pred_boxes = pred_boxes[pred_scores.argsort().flip(-1)]
    iou_scores = torchvision.ops.boxes.box_iou(target_boxes, pred_boxes)

    iou_scores = iou_scores.where(iou_scores > threshold, torch.tensor(0.))

    mappings = torch.zeros_like(iou_scores)
    if not (iou_scores[:, 0] == 0.).all():
        mappings[iou_scores[:, 0].argsort()[-1], 0] = 1

    for i in range(1, iou_scores.shape[1]):
        temp = torch.logical_not(mappings[:, :i].sum(dim=1)).long() * iou_scores[:, i]
        if (temp == 0).all():
            continue
        mappings[temp.argsort()[-1], i] = 1

    tp = mappings.sum()
    fp = (mappings.sum(dim=0) == 0).sum()
    fn = (mappings.sum(dim=1) == 0).sum()
    return tp / (tp + fp + fn)


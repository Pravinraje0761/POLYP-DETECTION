import torch
import numpy as np


def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    return (true_positive + smooth) / (predicted_positive + smooth)


def recall(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    return (true_positive + smooth) / (actual_positive + smooth)


def f1_score(pred, target, smooth=1e-6):
    prec = precision(pred, target, smooth)
    rec = recall(pred, target, smooth)
    return (2 * prec * rec + smooth) / (prec + rec + smooth)


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

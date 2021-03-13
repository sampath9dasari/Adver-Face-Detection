#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:59:21 2020

@author: susanthdasari
"""

import numpy as np
import tqdm as tqdm
import itertools
import matplotlib.pyplot as plt

def bbox_overlaps(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    
    overlaps = np.zeros((N, K), dtype=np.float)
    box_areas = (query_boxes[:, 3] - query_boxes[:, 1]) * (query_boxes[:, 2] - query_boxes[:, 0])
    for k in range(K):
        box_area = box_areas[k]
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1

            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    ua = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih
                    overlaps[n, k] = iw * ih / ua
    
    return overlaps


def image_eval(pred, gt, iou_thresh=0.5):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    """

    _pred = pred['boxes'].numpy()
    _scores = pred['scores'].numpy()
    
    idx_sorted = np.argsort(_scores)[::-1]
    _pred = _pred[idx_sorted]
    _scores = _scores[idx_sorted]
    
    _gt = gt['boxes'].numpy()
    
    TP = np.zeros(_pred.shape[0])
    FP = np.zeros(_pred.shape[0])
    gt_list = np.zeros(_gt.shape[0])

    overlaps = bbox_overlaps(_pred, _gt)
    
    for h in range(_pred.shape[0]):
        
#         if _scores[h] < score_thresh: 
#             continue
        
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh and gt_list[max_idx] == 0:
            TP[h] = 1
            gt_list[max_idx] = 1
        else: FP[h] = 1
            
            
    return TP, FP, _scores


def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

# @staticmethod
# 11-point interpolated average precision
def ElevenPointInterpolatedAP(rec, prec):
    # def CalculateAveragePrecision2(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]

def evaluation(pred, gt_box, iou_thresh=0.5, interpolation_method = 'ElevenPoint'):
#     pred = get_preds(pred)
#     norm_score(pred)
    TP_info = []
    FP_info = []
    scores_info = []
    # different setting
    count_face = 0
    # [hard, medium, easy]
    pbar = tqdm.tqdm(range(len(pred)))
    for i in pbar:
        pbar.set_description('Processing ')
#             pred_list = pred
#             sub_gt_list = gt_list[i][0]
#             # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
#             gt_bbx_list = facebox_list[i][0]

#             for j in range(len(img_list)):

        pred_info = pred[i]

        gt_boxes = gt_box[i]

        if len(gt_boxes) == 0 or len(pred_info) == 0:
            continue
        
        count_face += len(gt_boxes['boxes'])

        eval_results = image_eval(pred_info, gt_boxes, iou_thresh)
        
        
        TP_info.append(eval_results[0].copy())
        FP_info.append(eval_results[1].copy())
        
        scores_info.append(eval_results[2].copy())
        
    TP_info = np.array(list(itertools.chain(*TP_info)))
    FP_info = np.array(list(itertools.chain(*FP_info)))
    scores_info = np.array(list(itertools.chain(*scores_info)))
    
    scores_sorted_idx = np.argsort(scores_info)[::-1]
    TP_info = TP_info[scores_sorted_idx]
    FP_info = FP_info[scores_sorted_idx]
    scores_info = scores_info[scores_sorted_idx]
    
    acc_FP = np.cumsum(FP_info)
    acc_TP = np.cumsum(TP_info)
    rec = acc_TP / count_face
#     print(count_face)
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    
    if interpolation_method == 'ElevenPoint':
        [ap, mpre, mrec, ii] = ElevenPointInterpolatedAP(rec, prec)
    else: 
        [ap, mpre, mrec, ii] = CalculateAveragePrecision(rec, prec)
    
    r = {
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': count_face,
            'total TP': np.sum(TP_info),
            'total FP': np.sum(FP_info)
        }

    return r

def PlotPrecisionRecallCurve(r, method='EveryPoint'):
    result = r

    precision = result['precision']
    recall = result['recall']
    average_precision = result['AP']
    mpre = result['interpolated precision']
    mrec = result['interpolated recall']

    plt.close()
    plt.figure(figsize=(16,6))
#     if showInterpolatedPrecision:
    if method == 'EveryPoint':
        plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
    elif method == 'ElevenPoint':
        # Uncomment the line below if you want to plot the area
        # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
        # Remove duplicates, getting only the highest precision of each recall value
        nrec = []
        nprec = []
        for idx in range(len(mrec)):
            r = mrec[idx]
            if r not in nrec:
                idxEq = np.argwhere(mrec == r)
                nrec.append(r)
                nprec.append(max([mpre[int(id)] for id in idxEq]))
        plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
    plt.plot(recall, precision, label='Precision')
    plt.xlabel('recall')
    plt.ylabel('precision')
#     if showAP:
    ap_str = "{0:.2f}%".format(average_precision * 100)
        # ap_str = "{0:.4f}%".format(average_precision * 100)
    plt.title('Precision x Recall curve \nAP: %s' % (ap_str))
    plt.legend(shadow=True)
    plt.grid()
    plt.show()


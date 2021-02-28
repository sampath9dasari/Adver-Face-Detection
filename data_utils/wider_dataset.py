#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:30:41 2020

@author: susanthdasari
"""

import cv2
import numpy as np
import torch


DIR_INPUT = 'data'
DIR_TRAIN_IMG = f'{DIR_INPUT}/WIDER_train/images'
DIR_TRAIN_LABELS = f'{DIR_INPUT}/wider_face_split'




class WiderDataset(object):
    """
    Build a wider parser
    Parameters
    ----------
    image_list : path of the label file
    bboxes_list : path of the image files
    transforms : Any pytorch transforms needs to be performed
    Returns
    -------
    a wider parser
    """
    def __init__(self, image_list, bboxes_list, transforms=None):

        self.transforms = transforms
        self.image_list = image_list
        self.bboxes_list = bboxes_list


    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        boxes = self.bboxes_list[idx]

        im = cv2.imread(image_name, cv2.IMREAD_COLOR)
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.float32)
        im /= 255.0
        
#         im = F.to_tensor(im)

        num_objs = len(boxes)

#         imshow(im, boxes)

    
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
#         imshow(im, boxes)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target

    def __len__(self):
        return len(self.image_list)
    
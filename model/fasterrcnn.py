import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.nn as nn
import torch as t

from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from data_utils.wider_dataset import *
from data_utils.wider_eval import *
from data_utils.arraytools import *
from data_utils.data_read import *
from model.model_utils import *
from data_utils.utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_Faster_RCNN(backbone=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2  # 1 class (face) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # change backbone
    if backbone is not None:
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone, True)
        model.backbone = backbone

    return model


def model_eval(model, data_loader):
    prediction_info = []
    target_info = []
    model.eval()

    for images, targets in data_loader:
        # images = list(F.to_tensor(image) for image in images)
        # targets = [{k: v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
        prediction_info.append(predictions)
        target_info.append(targets)
    #     print(len(predictions[0]['scores']))

    prediction_info = list(itertools.chain(*prediction_info))
    target_info = list(itertools.chain(*target_info))

    return prediction_info, target_info


def predict(model, image):
    img = [totensor(image)]

    # print(len(img))
    model.eval()

    with torch.no_grad():
        prediction = model(img)

    return prediction


def train_epoch(model, epoch, train_dataloader, averager, optimizer):
    model.train()
    itr = 1
    step_time = time.time()
    try:
        for images, targets in train_dataloader:
            # images, targets = images.to(device), targets.to(device)

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            #         print(loss_dict)
            loss_value = losses.item()

            averager.send(loss_value)

            if itr % 500 == 0:
                update_time = time.time()
                print(f"Epoch #{epoch} | Training Iteration #{itr} loss: {loss_value} | Time elapsed: {convert(update_time - step_time)}")
                step_time = update_time

            itr += 1

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            
            images = list(image.detach() for image in images)
            targets = [{k: v.detach() for k, v in t.items()} for t in targets]
            images.clear()
            targets.clear()
            torch.cuda.empty_cache()
#             images = list(image.to('cpu') for image in images)
#             targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
#             torch.cuda.empty_cache()
#             model.to(device)

    except Exception() as e:
        print(e)
        e2 = sys.exc_info()[0]
        print(e2)
        return e


def val_epoch(model, val_dataloader, averager):
    # model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            # images, targets = images.to(device), targets.to(device)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            # print(loss_dict)
            loss_value = losses.item()

            averager.send(loss_value)
            
            images = list(image.detach() for image in images)
            targets = [{k: v.detach() for k, v in t.items()} for t in targets]
            images.clear()
            targets.clear()
            torch.cuda.empty_cache()



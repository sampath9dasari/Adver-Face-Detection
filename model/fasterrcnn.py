import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import skimage as skimage

import torch
import torchvision

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torch.nn as nn
import torch as t

from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from data_utils.wider_dataset import *
from data_utils.wider_eval import *
from data_utils.data_read import *
from model.model_utils import *
from lib.utils import *


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
        images = list(F.to_tensor(image) for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
        prediction_info.append(predictions)
        target_info.append(targets)
    #     print(len(predictions[0]['scores']))

    prediction_info = list(itertools.chain(*prediction_info))
    target_info = list(itertools.chain(*target_info))

    return prediction_info, target_info

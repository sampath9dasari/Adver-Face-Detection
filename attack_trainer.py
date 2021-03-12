import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import skimage as skimage
import torch
import torchvision


import torch.nn as nn
import torch as t

from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from data_utils.wider_dataset import *
from data_utils.wider_eval import *
from data_utils.data_read import *
from model.model_utils import *
from model.fasterrcnn import *
from lib.utils import *

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
wider_img_list, wider_bboxes = wider_read(150)

train_dataset = WiderDataset(wider_img_list[:140], wider_bboxes[:140])
test_dataset = WiderDataset(wider_img_list[140:], wider_bboxes[140:])
train_loader = DataLoader(train_dataset,
    batch_size=4,
    shuffle=True,
    # num_workers=6,
    collate_fn=collate_fn
)
test_loader = DataLoader(test_dataset,
    batch_size=4,
    shuffle=True,
    # num_workers=6,
    collate_fn=collate_fn
)

model = load_Faster_RCNN(backbone='resnet18')

model.load_state_dict(torch.load('./saved_models/fasterrcnn_resnet18_fpn3.pth'))

prediction_info, target_info = model_eval(model, test_loader)

r = evaluation(prediction_info, target_info, iou_thresh=0.3, interpolation_method='EveryPoint')
print(r['AP'])

PlotPrecisionRecallCurve(r)
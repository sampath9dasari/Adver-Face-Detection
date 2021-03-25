
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torchvision
import art
import os

from model.fasterrcnn import *
from model.advattack import *
from data_utils.utils import *
from data_utils.arraytools import *

import os
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
COLORS = np.random.uniform(0, 255, size=(2, 3))

im = np.array(Image.open(os.getcwd()+'/demo/demo_image.jpg'))
im = im.astype(np.float32)
im /= 255.0
DEMO_IMAGE = im


@st.cache
def annotate_image(
    raw_image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    image = raw_image.copy()
    # loop over the detections
    # (h, w) = image.shape[:2]
    if detections['boxes'] is not None:
        for i in np.arange(0, len(detections['boxes'])):
            confidence = detections['scores'][i].cpu().numpy()
            print(confidence)

            if confidence > confidence_threshold:
                # print(confidence)
                bbox = detections['boxes'][i].cpu().numpy()
                print(bbox)
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                (startX, startY, endX, endY) = bbox.astype("int")

                # print(image.shape, image.dtype)

                cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)

    return image


@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs={torchvision.models.detection.faster_rcnn.FasterRCNN: lambda _:None})
def predict_lit_org(image):
    st.write('cache miss org')
    model = load_Faster_RCNN(backbone='resnet18')
    model.load_state_dict(torch.load('./saved_models/fasterrcnn_resnet18_2021-03-24.pth')['model'])
    img = [totensor(image)]
    model.eval()

    with torch.no_grad():
        prediction = model(img)

    return prediction


@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs={torchvision.models.detection.faster_rcnn.FasterRCNN: lambda _:None})
def predict_lit_adv(image):
    st.write('cache miss adv')
    model = load_Faster_RCNN(backbone='resnet18')
    model.load_state_dict(torch.load('./saved_models/fasterrcnn_resnet18_2021-03-24.pth')['model'])
    img = [totensor(image)]
    model.eval()

    with torch.no_grad():
        prediction = model(img)

    return prediction


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def adv_attack_lit(image):
    st.write('cache miss adv attack')
    model = load_Faster_RCNN(backbone='resnet18')
    model.load_state_dict(torch.load('./saved_models/fasterrcnn_resnet18_2021-03-24.pth')['model'])
    detector = PyTorchFasterRCNN(model=model, clip_values=(0, 1), preprocessing=None)
    # attack = FastGradientMethod(estimator=detector, eps=0.02, eps_step=0.001)
    attack = ProjectedGradientDescent(detector, eps=0.05, eps_step=0.001, max_iter=40, verbose=True)

    image_in_list = np.array([image])
    image_adv = attack.generate(x=image_in_list, y=None)

    img = [totensor(image_adv[0].copy())]
    model.eval()

    with torch.no_grad():
        prediction = model(img)

    return image_adv, prediction



st.title("Face Detection Privacy")

# model = load_Faster_RCNN(backbone='resnet18')
# model.load_state_dict(torch.load('./saved_models/fasterrcnn_resnet18_fpn3.pth'))
#
#
# adv_model = gen_adv_model(model)
# attack = gen_adv_attack(adv_model)


img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    image = image.astype(np.float32)
    image /= 255.0
    # print(image.shape)
else:
    image = np.array(DEMO_IMAGE)

org_detections = predict_lit_org(image)
ann_image = annotate_image(image, org_detections[0], confidence_threshold)

st.image(
    ann_image, caption=f"Original image", use_column_width=True, clamp=(0, 1)
)


image_adv, adv_detections = adv_attack_lit(image)

print(image_adv)
# adv_detections = predict_lit_adv(image_adv[0])

ann_image_adv = annotate_image(image_adv[0], adv_detections[0], confidence_threshold)

st.image(
    ann_image_adv, caption=f"Adversarial image", use_column_width=True, clamp=(0, 1)
)
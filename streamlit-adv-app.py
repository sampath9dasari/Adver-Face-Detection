
import streamlit as st
import cv2
from PIL import Image
import numpy as np

from model.fasterrcnn import *
from lib.advattack import *
from lib.utils import *
from data_utils.arraytools import *

import os
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
COLORS = np.random.uniform(0, 255, size=(2, 3))

im = np.array(Image.open('demo/demo_image.jpg'))
im = im.astype(np.float32)
im /= 255.0
DEMO_IMAGE = im

@st.cache
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    # (h, w) = image.shape[:2]
    if detections['boxes'] is not None:
        for i in np.arange(0, len(detections['boxes'])):
            confidence = detections['scores'][i]

            if confidence > confidence_threshold:
                bbox = detections['boxes'][i].cpu().numpy()
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                (startX, startY, endX, endY) = bbox.astype("int")

                # print(image.shape, image.dtype)

                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[1], 2)

    return image



st.title("Face Detection Privacy")

model = load_Faster_RCNN(backbone='resnet18')
model.load_state_dict(torch.load('./saved_models/fasterrcnn_resnet18_fpn3.pth'))


adv_model = gen_adv_model(model)
attack = gen_adv_attack(adv_model)

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


org_detections = predict(model, image)
image = annotate_image(image, org_detections[0], confidence_threshold)

st.image(
    image, caption=f"Original image", use_column_width=True, clamp=(0, 1)
)


image2 = np.array([image])
image_adv = attack.generate(x=image2, y=None)
adv_detections = predict(model, image_adv[0])

image = annotate_image(image_adv[0], adv_detections[0], confidence_threshold)

st.image(
    image, caption=f"Adversarial image", use_column_width=True, clamp=(0, 1)
)
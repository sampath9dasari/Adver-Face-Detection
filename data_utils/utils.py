
from matplotlib import pyplot as plt
import torch as t
import time

def imshow(img, bboxes=None):
    if isinstance(img, t.Tensor):
        #         print('yes')
        img = img.cpu().numpy().transpose(1, 2, 0)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    if bboxes is not None:
        for bbox in bboxes:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def imshow_th(img, predictions, threshold=0.7):
    box_filter = predictions[0]['scores'] > threshold
    boxes = predictions[0]['boxes'][box_filter]
    imshow(img, boxes)
    
def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))
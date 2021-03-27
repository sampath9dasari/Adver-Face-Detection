import os

import scipy.io as sio

DIR_INPUT = os.getcwd()+'/data'

DIR_TRAIN_IMG = f'{DIR_INPUT}/WIDER_train/images'
DIR_TRAIN_LABELS = f'{DIR_INPUT}/wider_face_split/wider_face_train.mat'

DIR_VAL_IMG = f'{DIR_INPUT}/WIDER_val/images'
DIR_VAL_LABELS = f'{DIR_INPUT}/wider_face_split/wider_face_val.mat'


def wider_read(limit_images=None, train=True):

    if train:
        wider_raw = sio.loadmat(f'{DIR_TRAIN_LABELS}')
        IMG_DIR = DIR_TRAIN_IMG
    else:
        wider_raw = sio.loadmat(f'{DIR_VAL_LABELS}')
        IMG_DIR = DIR_VAL_IMG

    wider_img_list = []
    wider_bboxes = []
    event_list = wider_raw.get('event_list')
    file_list = wider_raw.get('file_list')
    face_bbx_list = wider_raw.get('face_bbx_list')
    blur_list = wider_raw.get('blur_label_list')
    occl_list = wider_raw.get('occlusion_label_list')
    invalid_list = wider_raw.get('invalid_label_list')
    illum_list = wider_raw.get('illumination_label_list')
    img_count = 1
    for event_idx, event in enumerate(event_list):
        directory = event[0][0]
        for im_idx, im in enumerate(file_list[event_idx][0]):
            im_name = im[0][0]
            face_bbx = face_bbx_list[event_idx][0][im_idx][0]

                
#             print(face_bbx.shape)

            bboxes = []

            for i in range(face_bbx.shape[0]):
                xmin = int(face_bbx[i][0])
                ymin = int(face_bbx[i][1])
                xmax = int(face_bbx[i][2]) + xmin
                ymax = int(face_bbx[i][3]) + ymin
                if xmin != 0 and ymin != 0 and xmax != 0 and ymax != 0 and xmin < xmax and ymin < ymax:
                    if train:
                        bboxes.append((xmin, ymin, xmax, ymax))
                    else:
                        if occl_list[event_idx][0][im_idx][0][i] != 2 and invalid_list[event_idx][0][im_idx][0][i] != 1:
                            bboxes.append((xmin, ymin, xmax, ymax))
                        else:
                            bboxes.append((xmin, ymin, xmax, ymax))

            image_name = os.path.join(IMG_DIR, directory,
                                      im_name + '.jpg')
            
            if len(bboxes)==0 or len(bboxes) > 500: 
#                 print('No boxes')
                continue
            #         print(im_name)
            wider_img_list.append(image_name)
            wider_bboxes.append(bboxes)

            #         imshow(im, bboxes)

            if limit_images is not None:
                if img_count >= limit_images:
                    return wider_img_list, wider_bboxes
            img_count += 1

    return wider_img_list, wider_bboxes


def collate_fn(batch):
    return tuple(zip(*batch))

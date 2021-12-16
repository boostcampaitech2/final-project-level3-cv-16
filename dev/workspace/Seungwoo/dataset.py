import json
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

class PieDataset(Dataset):
    def __init__(
        self,
        json_path,
        image_root,
        image_size=(192,192)
        # transforms
    ):
        self.json_path = json_path
        self.image_root = image_root
        self.image_size = image_size
        with open(json_path, "r") as f:
            json_dict = json.load(f)
        self.df_images = pd.DataFrame(json_dict["images"])
        self.df_anno = pd.DataFrame(json_dict['annotations'])

        self.image_len = len(self.df_images)
        
        self.transforms = A.Compose([
            A.Resize(width=self.image_size[0], height=self.image_size[1]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return self.image_len
    
    def __getitem__(self, idx):
        image_path = self.image_root + self.df_images.iloc[idx]["file_name"]
        image_id = self.df_images.iloc[idx]["id"]
        annotations = self.df_anno[self.df_anno["image_id"] == image_id]

        
        PIL_image = PIL.Image.open(image_path).convert('RGB')
        image = np.array(PIL_image)/255.
        PIL_image.close()
        image, keypoints = self.square_pad(image, annotations)
        keypoints = self.get_xyform_kp(keypoints)

        transformed = self.transforms(image=image, keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']

        heatmap = self._get_heatmaps_from_points(transformed_keypoints)
        return transformed_image, heatmap
    


    def square_pad(self, image, annotations):
        # Pad to make square
        H, W, C = image.shape
        epsilon = 1e-5
        if H != W:
            long, short = max(H, W), min(H, W)
            pad_size, is_odd = divmod(long-short, 2)

            # v_pad
            if long == W:
                up_pad = np.zeros([pad_size, W, C])
                down_pad = np.zeros([pad_size + is_odd, W, C])
                image = np.concatenate(
                    (up_pad, image, down_pad), axis = 0
                )
                shifted_kp = []
                for key_points in annotations["bbox"]:
                    x1,y1, x2,y2, x3,y3 = key_points
                    y1 += pad_size
                    y2 += pad_size
                    y3 += pad_size

                    x1 = min(x1, W-epsilon)
                    x2 = min(x2, W-epsilon)
                    x3 = min(x3, W-epsilon)
                    y1 = min(y1, H-epsilon)
                    y2 = min(y2, H-epsilon)
                    y3 = min(y3, H-epsilon)

                    shifted_kp.append([
                        x1, y1,
                        x2, y2,
                        x3, y3,
                    ])

            # h_pad
            elif long == H:
                left_pad = np.zeros([H, pad_size, C])
                right_pad = np.zeros([H, pad_size + is_odd, C])
                image = np.concatenate(
                    (left_pad, image, right_pad), axis = 1
                )
                shifted_kp = []
                for key_points in annotations["bbox"]:
                    x1,y1, x2,y2, x3,y3 = key_points
                    x1 += pad_size
                    x2 += pad_size
                    x3 += pad_size

                    x1 = min(x1, W-epsilon)
                    x2 = min(x2, W-epsilon)
                    x3 = min(x3, W-epsilon)
                    y1 = min(y1, H-epsilon)
                    y2 = min(y2, H-epsilon)
                    y3 = min(y3, H-epsilon)

                    shifted_kp.append([
                        x1, y1,
                        x2, y2,
                        x3, y3,
                    ])
        else:
            shifted_kp = []
            for key_points in annotations["bbox"]:
                    x1,y1, x2,y2, x3,y3 = key_points

                    x1 = min(x1, W-epsilon)
                    x2 = min(x2, W-epsilon)
                    x3 = min(x3, W-epsilon)
                    y1 = min(y1, H-epsilon)
                    y2 = min(y2, H-epsilon)
                    y3 = min(y3, H-epsilon)
                    
                    shifted_kp.append([
                        x1, y1,
                        x2, y2,
                        x3, y3
                    ])
        return image, shifted_kp

    
    def get_xyform_kp(self, kps):
        xy_form = []
        for kp in kps:
            if len(kp) != 6:
                raise ValueError("invalid kepoint shape")
            xy_form.append((kp[0], kp[1], "value_point"))
            xy_form.append((kp[2], kp[3], "value_point"))
            xy_form.append((kp[4], kp[5], "center_point"))
        return xy_form


    def _get_heatmaps_from_points(self, keypoints):

        value_pts = np.array([(pt[0], pt[1]) for pt in keypoints if pt[2] == 'value_point'])
        center_pts = np.array([(pt[0], pt[1]) for pt in keypoints if pt[2] == 'center_point'])
        heatmap = np.zeros((2, self.image_size[0], self.image_size[1]), dtype=np.float32)
        for i, pt in enumerate(center_pts):
            heatmap[0] = self._draw_labelmap(heatmap[0], pt, sigma=3)
        for i, pt in enumerate(value_pts):
            heatmap[1] = self._draw_labelmap(heatmap[1], pt, sigma=3)
        
        
        return heatmap

    def _draw_labelmap(self, heatmap, pt, sigma):
        # Draw a 2D gaussian
        # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
        H, W = heatmap.shape[:2]

        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
        if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            return heatmap

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1

        '''======================================================='''
        '''======================== TO DO ========================'''
        g = np.exp(-((x-x0) ** 2 + (y - y0) ** 2) / (2 * sigma **2))
        '''======================== TO DO ========================'''
        '''======================================================='''

        # Usable gaussian range
        # 
        g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
        # Image range
        heatmap_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
        heatmap_y = max(0, ul[1]), min(br[1], heatmap.shape[0])
        
        try:
            heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        except:
            heatshape = heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1]].shape
            gshape = g[g_y[0]:g_y[1], g_x[0]:g_x[1]].shape
            if heatshape[0] > gshape[0] and heatshape[1] > gshape[1]:
                heatmap[heatmap_y[0]:heatmap_y[1]-1, heatmap_x[0]:heatmap_x[1]-1] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            elif heatshape[0] > gshape[0]:
                heatmap[heatmap_y[0]:heatmap_y[1]-1, heatmap_x[0]:heatmap_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            elif heatshape[1] > gshape[1]:
                heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1]-1] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            else:
                print("size mismatch")

        return heatmap

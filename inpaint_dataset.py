import os
from random import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


DATA_DIR = '../camera_data/'
DATA_AUGED = 'seg_image_augmented(HR_skelonton)'

if not os.path.exists(DATA_DIR):
    print('DIR not exists, exit')
    quit()
if not os.path.exists(DATA_DIR+DATA_AUGED):
    os.mkdir(DATA_DIR+DATA_AUGED)

x_train_dir = os.path.join(DATA_DIR, 'rgb_image')
y_train_dir = os.path.join(DATA_DIR, 'seg_image')
save_train_dir = os.path.join(DATA_DIR, DATA_AUGED)
print(os.path.exists(x_train_dir))
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()



class Dataset(BaseDataset):
    
    CLASSES = ['dlo','other']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        print(self.ids)
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print(self.masks_fps)
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        print(self.class_values)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        mask = cv2.imread(self.masks_fps[i], 0)
        #print(self.masks_fps[i], )
        if np.max(mask) == np.min(mask):
            return mask
        mask = np.where(mask==np.max(mask),255,0)
        cnt = 0
        # width = 5

        # Using simple way to mask DLOs
        # width = 50
        # while True:
        #     cnt += 1
        #     random_sample = np.random.rand(mask.shape[0],mask.shape[1])
        #     #random_sample = np.where(random_sample>0.99,1,0)
        #     random_sample = np.where(random_sample>0.9999,1,0)
        #     random_mask = mask*random_sample
        #     if np.sum(random_mask) >= 2:
        #         break
        #     if cnt > 50:
        #         return mask
        # random_mask_picked = np.where(random_mask==1)
        # ori_mask = np.copy(mask)
        # for i in range(0,np.sum(random_mask)-1):
        #     #print(np.sum(mask), np.sum(random_mask)-1, random_mask[0][i], random_mask.shape)
        #     mask[random_mask_picked[0][i]-width:random_mask_picked[0][i]+width,random_mask_picked[1][i]-width:random_mask_picked[1][i]+width]=0
        #     #print(i)
        # #print("after",mask)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        print(mask.shape)
        thinned = cv2.ximgproc.thinning(mask.astype('uint8'))
        _, binaryImage = cv2.threshold(thinned, 128, 10, cv2.THRESH_BINARY)
        h = np.array([[1, 1, 1],
                    [1, 10, 1],
                    [1, 1, 1]])
        imgFiltered = cv2.filter2D(binaryImage, -1, h)
        endPointsMask = np.where(imgFiltered == 110, 255, 0)
        endPointsMask = endPointsMask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours[0].shape)
        
        contour_image = np.zeros_like(mask)
        for i in range(contours[0].shape[0]//3):
            print(contours[0][i][0])
            contour_image[contours[0][i][0][1],contours[0][i][0][0]] = 1
        #contour_image = 1
        visualize(images=mask, mask=binaryImage, endpoint=endPointsMask, contours=contour_image)

        return mask
        
    def __len__(self):
        return len(self.ids)

dataset = Dataset(x_train_dir, y_train_dir, classes=['dlo'])

for i in range(0,len(dataset.masks_fps)):
    mask = dataset[i] # get some sample
    #cv2.imwrite(dataset.masks_fps[i].replace('seg_image',DATA_AUGED),mask*255)
    print('save ',dataset.masks_fps[i].replace('seg_image',DATA_AUGED))
#visualize(image=mask)

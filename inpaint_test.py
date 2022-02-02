import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
import matplotlib.pyplot as plt
from util import *
from dataset import *
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATA_DIR = '../camera_data/'
DATA_AUGED = 'seg_image_augmented(HR_largeblank)'
MODEL_DIR = '../inpaint_model/'

if not os.path.exists(DATA_DIR):
    print('DIR not exists, exit')
    quit()

DATASET = 'seg_image_augmented(HR_largeblank)'

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['dlo']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# x_test_dir = os.path.join(DATA_DIR, 'inpaint_predicted_mask/real')
# y_test_dir = os.path.join(DATA_DIR, 'inpaint_predicted_mask/real')
x_test_dir = os.path.join(DATA_DIR, 'seg_image_augmented(HR_largeblank)')
y_test_dir = os.path.join(DATA_DIR, 'seg_image')
best_model = torch.load('../inpaint_model/seg_image_augmented(HR_largeblank)_1_BCELoss_720*720.pth')



model = smp.PAN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)



test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    #augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)

for i in range(len(test_dataset_vis.images_fps)):

    n = i
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    x_tensor_ori = torch.clone(x_tensor)
    for j in range(0,3):
        x_tensor = best_model.predict(x_tensor_ori)
        x_tensor = torch.where(x_tensor>0.99,255,0)
        for k in range(0,3):
            x_tensor_ori[:,0,:,:] = x_tensor[:,0,:,:]
    
    x_tensor = best_model.predict(x_tensor_ori)

    pr_mask = x_tensor
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        #img_name=test_dataset_vis.images_fps[i],
        #save_dir=DATA_DIR+'disp_image/',
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask,
        diff_mask=gt_mask-pr_mask
    )

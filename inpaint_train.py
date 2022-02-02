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
import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATA_DIR = '../camera_data/'
DATA_AUGED = 'seg_image_augmented(HR_largeblank)'
MODEL_DIR = '../inpaint_model/'

# for cropping image
IMAGE_WIDTH, IMAGE_HEIGHT = 700, 700

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['dlo']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

if not os.path.exists(DATA_DIR):
    print('DIR not exists, exit')
    quit()

if not os.path.exists(os.path.join(MODEL_DIR, DATA_AUGED)):
    os.mkdir(os.path.join(MODEL_DIR, DATA_AUGED))

x_train_dir = os.path.join(DATA_DIR, DATA_AUGED)
y_train_dir = os.path.join(DATA_DIR, 'seg_image')
# No validation so far
# x_val_dir = os.path.join(DATA_DIR, 'inpaint/seg_image_augmented')
# y_val_dir = os.path.join(DATA_DIR, 'inpaint/seg_image')
# valid_dataset = Dataset(
#     x_val_dir, 
#     y_val_dir, 
#     augmentation=get_validation_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )
# valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)
# valid_epoch = smp.utils.train.ValidEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     device=DEVICE,
#     verbose=True,
# )



model = smp.PAN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)



train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)


loss = smp.utils.losses.DiceLoss() #BCELoss()#
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=2e-2),
])
optimizer.param_groups[0]['lr'] = 2e-2

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

max_score = 0

for i in range(0, 30):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    #valid_logs = valid_epoch.run(valid_loader)
    
    #do something (save model, change lr, etc.)
    now = datetime.datetime.now()
    print(now.year, now.month, now.day, now.hour, now.minute, now.second)
    torch.save(model, '../inpaint_model/{}_{}_BCELoss_720*720_{}_{}_{}_{}_{}_{}.pth'.format(DATA_AUGED,i,now.year, now.month, now.day, now.hour, now.minute, now.second))
    
    #if max_score < valid_logs['iou_score']:
    #    max_score = valid_logs['iou_score']
    #    torch.save(model, '../seg_model/best_model_colored_rope.pth')
    #    print('Model saved!')
    
    LR = optimizer.param_groups[0]['lr']        
    if i == 10 or i == 20:
        optimizer.param_groups[0]['lr'] = LR*0.1
        print('Decrease decoder learning rate to 1e-5!')


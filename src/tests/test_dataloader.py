import cv2
import os, os.path as osp

from beehive.controls import datamaker
from omegaconf import OmegaConf


cfg = OmegaConf.load('configs/mks/mk000.yaml')
cfg.transform.preprocess = None
cfg.data.num_workers = 0
loader, _ = datamaker.get_train_val_dataloaders(cfg)

for data in loader:
    break

SAVE_IMAGES = 'test-images/'
if not osp.exists(SAVE_IMAGES): os.makedirs(SAVE_IMAGES)


images = data[0].view(-1, 448, 448).numpy()
for ind, im in enumerate(images):
    _ = cv2.imwrite(osp.join(SAVE_IMAGES, f'image_{ind:003d}.png'), 
                    im.astype('uint8'))
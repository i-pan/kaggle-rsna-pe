import pandas as pd
import numpy as np
import glob
import pydicom
import cv2
import os, os.path as osp

from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_dicom_array(f):
    dicom_files = glob.glob(osp.join(f, '*.dcm'))
    dicoms = [pydicom.dcmread(d) for d in dicom_files]
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    # Assume all images are axial
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
    dicoms = np.asarray([d.pixel_array for d in dicoms])
    dicoms = dicoms[np.argsort(z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    return dicoms, np.asarray(dicom_files)[np.argsort(z_pos)]


def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X


def save_array(X, save_dir, file_names):
    for ind, img in enumerate(X):
        savefile = osp.join(save_dir, file_names[ind])
        if not osp.exists(osp.dirname(savefile)): 
            os.makedirs(osp.dirname(savefile))
        _ = cv2.imwrite(osp.join(save_dir, file_names[ind]), img)


def edit_filenames(files):
    dicoms = [f"{ind:04d}_{f.split('/')[-1].replace('dcm','jpg')}" for ind,f in enumerate(files)]
    series = ['/'.join(f.split('/')[-3:-1]) for f in files]
    return [osp.join(s,d) for s,d in zip(series, dicoms)]


class Lungs(Dataset):
    def __init__(self, dicom_folders):
        self.dicom_folders = dicom_folders
    def __len__(self): return len(self.dicom_folders)
    def get(self, i):
        return load_dicom_array(self.dicom_folders[i])
    def __getitem__(self, i):
        try:
            return self.get(i)
        except Exception as e:
            print(e)
            return None


SAVEDIR = '../../data/train-jpegs/'
MAX_LENGTH = 256.

if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)

df = pd.read_csv('../../data/train/train.csv')
dicom_folders = list(('../../data/train/' + df.StudyInstanceUID + '/'+ df.SeriesInstanceUID).unique())[5645:]
dset = Lungs(dicom_folders)
loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

for data in tqdm(loader, total=len(loader)):
    data = data[0]
    if type(data) == type(None): continue
    try:
        image, files = data
        # Windows from https://pubs.rsna.org/doi/pdf/10.1148/rg.245045008
        image_lung = np.expand_dims(window(image, WL=-600, WW=1500), axis=3)
        image_mediastinal = np.expand_dims(window(image, WL=40, WW=400), axis=3)
        image_pe_specific = np.expand_dims(window(image, WL=100, WW=700), axis=3)
        image = np.concatenate([image_mediastinal, image_pe_specific, image_lung], axis=3)
        rat = MAX_LENGTH / np.max(image.shape[1:])
        image = zoom(image, [1.,rat,rat,1.], prefilter=False, order=1)
        files = edit_filenames(files)
        save_array(image, SAVEDIR, files)
    except Exception as e:
        print(e)


# kaggle datasets create -u -r zip -p ../../data/train-jpegs/



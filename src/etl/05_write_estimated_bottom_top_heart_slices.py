"""
Write 1,000 CTs to JPEG so I can use it to manually identify heart slices.
"""
import pandas as pd
import numpy as np
import pydicom
import glob
import cv2
import os, os.path as osp

from tqdm import tqdm


def window(X, WL=50, WW=350):
    lower, upper = WL-WW/2, WL+WW/2
    X = np.clip(X, lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X


def load_dicom(fp):
    dicomfile = pydicom.dcmread(fp)
    m = float(dicomfile.RescaleSlope)
    b = float(dicomfile.RescaleIntercept)
    array = dicomfile.pixel_array*m+b
    array = window(array)
    return array


DATADIR = '../../data/train/'
SAVEDIR = '../../data/heart-slices-2/'
if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)


df = pd.read_csv(osp.join(DATADIR, 'train_kfold.csv'))
# lung0 = pd.read_csv(osp.join(DATADIR, 'pct_lung_slices_0.csv'))
# lung1 = pd.read_csv(osp.join(DATADIR, 'pct_lung_slices_1.csv'))
# lung2 = pd.read_csv(osp.join(DATADIR, 'pct_lung_slices_2.csv'))
# lung3 = pd.read_csv(osp.join(DATADIR, 'pct_lung_slices_3.csv'))
# lung_df = pd.concat([lung0,lung1,lung2,lung3])
# df = df.merge(lung_df, on=['filepath','SeriesInstanceUID'])

for series_id, series_df in tqdm(df.groupby('SeriesInstanceUID'), total=len(df.SeriesInstanceUID.unique())):
    series_df = series_df.sort_values('ImagePositionPatient_2').reset_index(drop=True)
    #series_df = series_df[series_df.pct_lung >= 0.05]
    # Take top 60%
    series_df = series_df.iloc[:int(len(series_df)*0.6)]
    bot_slice = series_df.filepath.iloc[0]
    top_slice = series_df.filepath.iloc[-1]
    bot_array = load_dicom(osp.join(DATADIR, bot_slice))
    top_array = load_dicom(osp.join(DATADIR, top_slice))
    bot_array = cv2.resize(bot_array, (200,200))
    top_array = cv2.resize(top_array, (200,200))
    _ = cv2.imwrite(osp.join(SAVEDIR, f'{series_id}_bot_slice.jpg'), bot_array)
    _ = cv2.imwrite(osp.join(SAVEDIR, f'{series_id}_top_slice.jpg'), top_array)


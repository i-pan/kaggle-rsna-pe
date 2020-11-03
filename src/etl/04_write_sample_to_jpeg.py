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


def save_images(sample, save_dir):
    if not osp.exists(save_dir): os.makedirs(save_dir)
    for series, _df in tqdm(sample.groupby('SeriesInstanceUID'), total=len(sample.SeriesInstanceUID.unique())):
        _df = _df.sort_values('ImagePositionPatient_2')
        dicoms = [pydicom.dcmread(osp.join(DATADIR, _)) for _ in _df.filepath.values[::4]]
        m = float(dicoms[0].RescaleSlope)
        b = float(dicoms[0].RescaleIntercept)
        arrays = [d.pixel_array*m+b for d in dicoms]
        arrays = [window(_) for _ in arrays]
        for ind, a in enumerate(arrays):
            a = cv2.resize(a, (128,128))
            _ = cv2.imwrite(osp.join(save_dir, f'{_df.iloc[0].StudyInstanceUID}_{series}_{ind:04d}_{_df.iloc[ind].SOPInstanceUID}.jpg'), a)


DATADIR = '../../data/train/'
SAVEDIR = '../../data/sample-1500/'
if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)

df = pd.read_csv(osp.join(DATADIR, 'train_kfold.csv'))
# Take 500 from normals
normal = df[df.negative_exam_for_pe == 1][['StudyInstanceUID', 'SeriesInstanceUID']].drop_duplicates().sample(n=500, random_state=0)
normal = normal.merge(df, on=['StudyInstanceUID', 'SeriesInstanceUID'])
# Take 500 from RV/LV >= 1
bigrv = df[df.rv_lv_ratio_gte_1 == 1][['StudyInstanceUID', 'SeriesInstanceUID']].drop_duplicates().sample(n=500, random_state=0)
bigrv = bigrv.merge(df, on=['StudyInstanceUID', 'SeriesInstanceUID'])
# Take 500 from +PE, RV/LV < 1
pe_normalrv = df[(df.rv_lv_ratio_lt_1 == 1) & (df.negative_exam_for_pe == 0)][['StudyInstanceUID', 'SeriesInstanceUID']].drop_duplicates().sample(n=500, random_state=0)
pe_normalrv = pe_normalrv.merge(df, on=['StudyInstanceUID', 'SeriesInstanceUID'])

save_images(normal, osp.join(SAVEDIR, 'normal'))
save_images(bigrv, osp.join(SAVEDIR, 'bigrv'))
save_images(pe_normalrv, osp.join(SAVEDIR, 'pe_normalrv'))

import pandas as pd
import pydicom
import torch
import glob
import os, os.path as osp

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


KEYS = [
    'SOPInstanceUID', 
    'SeriesInstanceUID', 
    'StudyInstanceUID', 
    'InstanceNumber', 
    'ImagePositionPatient',
    'ImageOrientationPatient', 
    'PixelSpacing', 
    'RescaleIntercept', 
    'RescaleSlope', 
    'WindowCenter', 
    'WindowWidth'
]

DATADIR = '../../data/train/'


class Metadata(Dataset):
    def __init__(self,
                 dcmfiles):
        self.dcmfiles = dcmfiles
    def __len__(self): return len(self.dcmfiles)
    def __getitem__(self, i):
        dcm = pydicom.dcmread(self.dcmfiles[i], stop_before_pixels=True)
        metadata = {}
        for k in KEYS:
            try:
                att = getattr(dcm, k)
                if k in ['InstanceNumber', 'RescaleSlope', 'RescaleIntercept']:
                    metadata[k] = float(att)
                elif k in ['PixelSpacing', 'ImagePositionPatient', 'ImageOrientationPatient']:
                    for ind, coord in enumerate(att):
                        metadata[f'{k}_{ind}'] = float(coord)
                else:
                    metadata[k] = str(att)
            except Exception as e:
                print(e)
        metadata['filepath'] = '/'.join(self.dcmfiles[i].split('/')[-3:])
        return pd.DataFrame(metadata, index=[0])


files = glob.glob(osp.join(DATADIR, '*/*/*.dcm'))
dset = Metadata(files)

loader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=4, collate_fn=lambda x: x)

meta = []
for data in tqdm(loader, total=len(loader)): 
    meta += [data[0]]


meta_df = pd.concat(meta, axis=0, ignore_index=True)
unique_series = pd.DataFrame(meta_df.SeriesInstanceUID.value_counts()).reset_index()
unique_series.columns = ['SeriesInstanceUID', 'num_slices']
meta_df = meta_df.merge(unique_series, on='SeriesInstanceUID')
meta_df.to_csv(osp.join(DATADIR, 'metadata.csv'), index=False)

train = pd.read_csv(osp.join(DATADIR, 'train.csv'))
train = train.merge(meta_df, on=['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID'])
train.to_csv(osp.join(DATADIR, 'train_with_meta.csv'), index=False)


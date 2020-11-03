"""
Modified lungmask to run w/o having to use SimpleITK. 
Confirmed that this gives almost identical results.
This lets me use multiprocessing w/ PyTorch Dataloader 
for faster segmentation.
"""
import SimpleITK as sitk
import argparse
import pandas as pd
import pydicom
import numpy as np
import glob
import gc
import os, os.path as osp

from torch.utils.data import Dataset, DataLoader
from lungmask import mask as lungmask_mask
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('save_file', type=str)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


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


def get_lung_mask(f, use_sitk=False): 
    if use_sitk:
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(f)
        sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(f, series_IDs[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(sorted_file_names)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        image = series_reader.Execute()
        segmentation = (lungmask_mask.apply(image) > 0).astype('uint8')
    else:
        image, sorted_file_names = load_dicom_array(f)
        segmentation = (lungmask_mask.apply(image=None, inimg_raw=image) > 0).astype('uint8')
    return segmentation, sorted_file_names


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


def main():
    args = parse_args()
    DATADIR = '../../data/train/'
    df = pd.read_csv(osp.join(DATADIR, 'train_kfold.csv'))
    dicom_folders = list((DATADIR + '/' + df.StudyInstanceUID + '/'+ df.SeriesInstanceUID).unique())[args.start:args.end]
    dset = Lungs(dicom_folders)
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: x)

    pct_lung_in_slice = []
    for data in tqdm(loader, total=len(loader)):
        data = data[0]
        if type(data) == type(None): continue
        try:
            image, file_names = data
            mask = (lungmask_mask.apply(image=None, inimg_raw=image) > 0).astype('uint8')
            lung_slice_indices = np.mean(mask, axis=(1,2))
            pct_lung_df = pd.DataFrame({
                            'filepath': ['/'.join(_.split('/')[-3:]) for _ in file_names],
                            'pct_lung': lung_slice_indices
                    })
            pct_lung_in_slice += [pct_lung_df]
        except Exception as e:
            print(e)

    pct_lung_in_slice = pd.concat(pct_lung_in_slice)
    pct_lung_in_slice['StudyInstanceUID'] = pct_lung_in_slice.filepath.apply(lambda x: x.split('/')[-3])
    pct_lung_in_slice['SeriesInstanceUID'] = pct_lung_in_slice.filepath.apply(lambda x: x.split('/')[-2])
    pct_lung_in_slice['SOPInstanceUID'] = pct_lung_in_slice.filepath.apply(lambda x: x.split('/')[-1].replace('.dcm', ''))
    pct_lung_in_slice.to_csv(osp.join(DATADIR, args.save_file), index=False)


if __name__ == '__main__':
    main()

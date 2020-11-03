import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np
import glob
import gc
import os, os.path as osp

from lungmask import mask as lungmask_mask
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('save_file', type=str)
    return parser.parse_args()


def get_lung_mask(f): 
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(f)
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(f, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(sorted_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image = series_reader.Execute()
    segmentation = (lungmask_mask.apply(image) > 0).astype('uint8')
    return segmentation, sorted_file_names


def main():
    args = parse_args()
    DATADIR = '../../data/train/'
    df = pd.read_csv(osp.join(DATADIR, 'train.csv'))
    dicom_folders = list((DATADIR + '/' + df.StudyInstanceUID + '/'+ df.SeriesInstanceUID).unique())[args.start:args.end]

    pct_lung_in_slice = []
    for dicom_folder in tqdm(dicom_folders, total=len(dicom_folders)):
        try:
            mask, file_names = get_lung_mask(dicom_folder)
            lung_slice_indices = np.mean(mask, axis=(1,2))
            pct_lung_df = pd.DataFrame({
                            'filepath': ['/'.join(_.split('/')[-3:]) for _ in file_names],
                            'pct_lung': lung_slice_indices
                    })
            pct_lung_in_slice += [pct_lung_df]
        except Exception as e:
            print(e)
            continue
    pct_lung_in_slice = pd.concat(pct_lung_in_slice)
    pct_lung_in_slice['StudyInstanceUID'] = pct_lung_in_slice.filepath.apply(lambda x: x.split('/')[-3])
    pct_lung_in_slice['SeriesInstanceUID'] = pct_lung_in_slice.filepath.apply(lambda x: x.split('/')[-2])
    pct_lung_in_slice['SOPInstanceUID'] = pct_lung_in_slice.filepath.apply(lambda x: x.split('/')[-1].replace('.dcm', ''))
    pct_lung_in_slice.to_csv(osp.join(DATADIR, args.save_file), index=False)


if __name__ == '__main__':
    main()

import glob
import os, os.path as osp
import pandas as pd


def enumerate_slices(df):
    df = df.sort_values('ImagePositionPatient_2')
    df['SliceIndex'] = list(range(len(df)))
    return df


PE_PROBA_DIR = '../../data/train-pe-probas/'
HEART_PROBA_DIR = '../../data/train-heart-probas/'


pe_probas = pd.concat([pd.read_csv(_) for _ in glob.glob(osp.join(PE_PROBA_DIR, '*/*.csv'))]).drop_duplicates()
heart_probas = pd.concat([pd.read_csv(_) for _ in glob.glob(osp.join(HEART_PROBA_DIR, '*.csv'))]).reset_index(drop=True)
heart_probas = heart_probas.iloc[heart_probas[['SeriesInstanceUID','SliceIndex']].drop_duplicates().index]

df = pd.read_csv('../../data/train/train_5fold.csv')
df = pd.concat([enumerate_slices(series_df) for series, series_df in df.groupby('SeriesInstanceUID')])

merge_cols = ['StudyInstanceUID', 'SeriesInstanceUID', 'SliceIndex']
df = df.merge(pe_probas, on=merge_cols).merge(heart_probas, on=merge_cols)

df.to_csv('../../data/train/train_5fold_with_probas.csv', index=False)
import glob
import numpy as np
import os, os.path as osp
import pandas as pd


def prepare_df(f):
    probs = glob.glob(osp.join(f,'*','*.csv'))
    probs = pd.concat([pd.read_csv(_) for _ in probs])
    probs = probs.groupby('StudyInstanceUID').first().reset_index()
    return probs


pe_probas   = prepare_df('../../data/train-pe-cls-probas/').sort_values('StudyInstanceUID').reset_index(drop=True)
pe_probas_b = prepare_df('../../data/train-pe-cls-probas-b/').sort_values('StudyInstanceUID').reset_index(drop=True)
pe_probas_c = prepare_df('../../data/train-pe-cls-probas-c/').sort_values('StudyInstanceUID').reset_index(drop=True)
rvlv_probas = prepare_df('../../data/train-rvlv-probas/').sort_values('StudyInstanceUID').reset_index(drop=True)

pe_cols = list(pe_probas.columns)
pe_cols.remove('StudyInstanceUID')
pe_cols.remove('SeriesInstanceUID')
pe_probas[pe_cols] = np.mean(np.asarray([pe_probas[pe_cols].values, pe_probas_b[pe_cols].values, pe_probas_c[pe_cols].values]), axis=0)

probas = pe_probas.merge(rvlv_probas, on=['StudyInstanceUID', 'SeriesInstanceUID'])

for c in probas.columns:
    if c not in ['StudyInstanceUID', 'SeriesInstanceUID']:
        probas[f'prob_{c}'] = probas[c]
        del probas[c]


df = pd.read_csv('../../data/train/train_5fold.csv')

study_targets = [
    'negative_exam_for_pe',
    'indeterminate',
    'chronic_pe',
    'acute_and_chronic_pe',
    'central_pe',
    'leftsided_pe',
    'rightsided_pe',
    'rv_lv_ratio_gte_1',
    'rv_lv_ratio_lt_1'
]

cv_columns = [_ for _ in df.columns if 'inner' in _ or 'outer' in _]
df = df[['StudyInstanceUID', 'SeriesInstanceUID'] + study_targets + cv_columns].drop_duplicates().reset_index(drop=True)
df = df.merge(probas, on=['StudyInstanceUID', 'SeriesInstanceUID'])
df.to_csv('../../data/train/train_5fold_probdataset_v2.csv', index=False)





import pandas as pd
import numpy as np

from beehive.controls.inference import enforce_label_consistency
from sklearn import metrics


def get_pos_frac(df):
    df['pos_frac'] = df.pe_present_on_image.mean()*WEIGHTS[0]
    return df


def wide_to_long(df):
    exam_labels = list(df[ALL_LABELS[1:]].values[0])
    new_df = pd.DataFrame({
        'id': [f'{df.StudyInstanceUID.values[0]}_{label}' for label in ALL_LABELS[1:]] + list(df.SOPInstanceUID),
        'label': exam_labels +list(df.pe_present_on_image),
    })
    if 'pos_frac' in df.columns:
        new_df['wt'] = WEIGHTS + list(df.pos_frac)
    return new_df


def loss(x, y, wt):
    assert len(x)==len(y)==len(wt)
    losslist = []
    for i in range(len(x)):
        L = -wt[i] * (y[i]*np.log(x[i]) + (1-y[i])*np.log(1-x[i]))
        losslist += [L]
    return np.sum(losslist) / np.sum(wt)


WEIGHTS = [
    0.0736196319,
    0.09202453988,
    0.1042944785,
    0.1042944785,
    0.1877300613,
    0.06257668712,
    0.06257668712,
    0.2346625767,
    0.0782208589
]

ALL_LABELS = [
    'pe_present_on_image',
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


########
# WIDE #
########
x = pd.read_csv('submission-wide-norules.csv')
x = pd.concat([enforce_label_consistency(exam) for _, exam in x.groupby('StudyInstanceUID')])

y = pd.read_csv('../data/train/train_5fold.csv')
y = y[y.StudyInstanceUID.isin(x.StudyInstanceUID)]
y = pd.concat([get_pos_frac(exam) for _, exam in y.groupby('StudyInstanceUID')])
x = x.sort_values(['StudyInstanceUID','SOPInstanceUID']).reset_index(drop=True)
y = y.sort_values(['StudyInstanceUID','SOPInstanceUID']).reset_index(drop=True)

assert len(x)==len(y)

wx = pd.concat([wide_to_long(exam) for _, exam in x.groupby('StudyInstanceUID')])
###

########
# LONG #
########
x = pd.read_csv('submission.csv')

y = pd.read_csv('../data/train/train_5fold.csv')
y = y[y.outer == 0]
#y = y[y.StudyInstanceUID.isin(x.StudyInstanceUID)]
y = pd.concat([get_pos_frac(exam) for _, exam in y.groupby('StudyInstanceUID')])
#x = x.sort_values(['StudyInstanceUID','SOPInstanceUID']).reset_index(drop=True)
#y = y.sort_values(['StudyInstanceUID','SOPInstanceUID']).reset_index(drop=True)

wx = x
###
wy = pd.concat([wide_to_long(exam) for _, exam in y.groupby('StudyInstanceUID')])
wy = wy[wy['id'].isin(wx['id'])]

wx = wx.sort_values('id')
wy = wy.sort_values('id')
loss(wx.label.values, wy.label.values, wy.wt.values)

losses = []
for i in ALL_LABELS:
  losses += [metrics.log_loss(y[i],x[i],labels=[0,1], eps=1e-6)]
  print(i)
  print(f'LOSS : {metrics.log_loss(y[i],x[i],labels=[0,1]):.4f}')
  try:
    print(f'AUC  : {metrics.roc_auc_score(y[i],x[i]):.4f}')
  except:
    print('AUC  : UNDEFINED')

np.sum(np.asarray(losses[1:])*np.asarray(WEIGHTS))


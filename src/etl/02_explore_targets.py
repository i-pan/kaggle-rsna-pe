import pandas as pd
import numpy as np


TARGETS = [
    'negative_exam_for_pe',
    'qa_motion',
    'qa_contrast',
    'flow_artifact',
    'rv_lv_ratio_gte_1',
    'rv_lv_ratio_lt_1',
    'leftsided_pe',
    'chronic_pe',
    'true_filling_defect_not_pe',
    'rightsided_pe',
    'acute_and_chronic_pe',
    'central_pe',
    'indeterminate',
]


df = pd.read_csv('../../data/train/train_kfold.csv')

# 1- Percent positive for each target
for t in TARGETS:
    tmp_df = df[['SeriesInstanceUID', t]].drop_duplicates()
    print(f'{t} : {tmp_df[t].mean()*100:.2f}%')

# 2- Distribution of number of slices/series
for p in [5,10,25,50,75,90,95]:
    print(f'{p:02d}th percentile : {int(np.percentile(df.num_slices, p))}')

# 3- For positive studies, what is the distribution of proportion
#    of positive slices?
pos_series = df[['SeriesInstanceUID', 'negative_exam_for_pe']].drop_duplicates()
pos_series = pos_series[pos_series.negative_exam_for_pe == 0] 
pos_series = pos_series.SeriesInstanceUID.unique()
pos_df = df[df.SeriesInstanceUID.isin(pos_series)]
proportion_pos_slices = [_df.pe_present_on_image.mean() for series_id, _df in pos_df.groupby('SeriesInstanceUID')]
for p in [5,10,25,50,75,90,95]:
    print(f'{p:02d}th percentile : {np.percentile(proportion_pos_slices, p)*100:0.2f}%')

# 4- # of bilateral PEs
np.unique((df.rightsided_pe + df.leftsided_pe + df.central_pe == 1).astype('int'), return_counts=True)
df['rightcentral_pe'] = (df.rightsided_pe + df.central_pe == 2).astype('int')
df['leftcentral_pe'] = (df.leftsided_pe + df.central_pe == 2).astype('int')
df['bilateral_pe'] = (df.rightsided_pe + df.leftsided_pe - df.central_pe == 2).astype('int')
df['diffuse_pe'] = (df.rightsided_pe + df.leftsided_pe + df.central_pe == 3).astype('int')

df.rightcentral_pe.value_counts(normalize=True)*100
df.leftcentral_pe.value_counts(normalize=True)*100
df.bilateral_pe.value_counts(normalize=True)*100
df.diffuse_pe.value_counts(normalize=True)*100

# 5- Explore RV/LV ratio more
series_df = df[['SeriesInstanceUID'] + TARGETS].drop_duplicates()
pos_series = series_df[series_df.negative_exam_for_pe == 0]
neg_series = series_df[series_df.negative_exam_for_pe == 1]
# For negative series, RV/LV ratio is ALWAYS 0 for both values ...
# For positive series:
#   RV/LV ratio >= 1 -- 39.7%
#   RV/LV ratio <  1 -- 53.7%
# So there are still some positive series where RV/LV ratio is 0 for both ...






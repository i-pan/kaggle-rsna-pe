import glob
import pandas as pd


all_files = glob.glob('../../data/sample-1500/normal/*.jpg') + glob.glob('../../data/sample-1500/bigrv/*.jpg')
with open('../../data/sample-1500/normal.txt') as f:
    existing_files = [line.strip() for line in f.readlines()]

with open('../../data/sample-1500/bigrv.txt') as f:
    existing_files += [line.strip() for line in f.readlines()]

all_files = [_.split('/')[-1].replace('.jpg','') for _ in all_files]
existing_files = [_.split('/')[-1].replace('.jpg','') for _ in existing_files]

deleted_files = list(set(all_files)-set(existing_files))

study = [_.split('_')[0] for _ in deleted_files]
series = [_.split('_')[1] for _ in deleted_files]
index = [int(_.split('_')[2]) for _ in deleted_files]
sop = [_.split('_')[3] for _ in deleted_files]

del_df = pd.DataFrame({
        'StudyInstanceUID': study,
        'SeriesInstanceUID': series,
        'SOPInstanceUID': sop,
        'slice_index': index
    })


# I screwed up the naming so the SOPInstanceUIDs don't match up...
# So now I need to rely on the fact that I saved every 4th slice
del_df = del_df.sort_values(['StudyInstanceUID', 'slice_index'])
del_df = pd.concat([series_df.iloc[[0,-1]] for series, series_df in del_df.groupby('SeriesInstanceUID')])
del_df = del_df.reset_index(drop=True)
del_df['heart'] = ['start', 'stop'] * (len(del_df) // 2)

df = pd.read_csv('../../data/train/train_5fold.csv')
df = df.sort_values(['StudyInstanceUID', 'ImagePositionPatient_2'])
df = df.merge(del_df, on=['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], how='left')
df = df[df.StudyInstanceUID.isin(del_df.StudyInstanceUID)]
df_list = []
for series, series_df in df.groupby('SeriesInstanceUID'):
    start = int(series_df.slice_index.dropna().min())
    stop  = int(series_df.slice_index.dropna().max())
    heart_df = series_df.iloc[start*4:(stop*4+1)]
    df_list += [heart_df]


heart_df = pd.concat(df_list)
df['heart_present'] = 0
df.loc[df.SOPInstanceUID.isin(heart_df.SOPInstanceUID), 'heart_present'] = 1

df.to_csv('../../data/train/train_heart_slices.csv', index=False)


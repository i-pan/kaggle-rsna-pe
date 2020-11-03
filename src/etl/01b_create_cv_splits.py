import pandas as pd

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


EXAM_LEVEL = [
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
    'SeriesInstanceUID'
]

STRATIFY_BY = EXAM_LEVEL.copy()
STRATIFY_BY.remove('SeriesInstanceUID')


def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else MultilabelStratifiedKFold
    outer_spl = splitter(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_counter = 0
    for outer_train, outer_test in outer_spl.split(df) if stratified is None else outer_spl.split(df, df[stratified]):
        df.loc[outer_test, 'outer'] = outer_counter
        inner_spl = splitter(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_counter = 0
        df['inner{}'.format(outer_counter)] = 888
        inner_df = df[df['outer'] != outer_counter].reset_index(drop=True)
        # Determine which IDs should be assigned to inner train
        for inner_train, inner_valid in inner_spl.split(inner_df) if stratified is None else inner_spl.split(inner_df, inner_df[stratified]):
            inner_train_ids = inner_df.loc[inner_valid, id_col]
            df.loc[df[id_col].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
            inner_counter += 1
        outer_counter += 1
    return df


df = pd.read_csv('../../data/train/train_with_meta.csv')
series_df = df[EXAM_LEVEL].drop_duplicates().reset_index(drop=True)
series_df.sum()
cv_df = create_double_cv(series_df.copy(), 'SeriesInstanceUID', 10, 10, stratified=STRATIFY_BY)
series_df = series_df.merge(cv_df[list(set(cv_df.columns)-set(STRATIFY_BY))], on='SeriesInstanceUID')
series_df = series_df[['SeriesInstanceUID'] + [col for col in series_df.columns if 'inner' in col or 'outer' in col]]
df = df.merge(series_df, on='SeriesInstanceUID')

df.to_csv('../../data/train/train_kfold_10x10.csv', index=False)



import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csvfile) 
    for fold in df['outer'].unique():
        train_df = df[df.outer != fold]
        valid_df = df[df.outer == fold]
        assert len(list(set(train_df.StudyInstanceUID.values) & set(valid_df.StudyInstanceUID.values))) == 0
        assert len(list(set(train_df.SeriesInstanceUID.values) & set(valid_df.SeriesInstanceUID.values))) == 0
    print('NO LEAK FOUND !')


if __name__ == '__main__':
    main()
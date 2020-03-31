import numpy as np
import pandas as pd
from tabulate import tabulate

from src.reader.columns import symptoms_name_dict, symptoms_dict
from src.reader.dir import LEUKEMIA_MASTER_PATH, LEUKEMIA_RAW_PATH


def read_data() -> pd.DataFrame:
    df = pd.read_csv(LEUKEMIA_MASTER_PATH)
    df = make_human_readable(df)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    return df


def make_human_readable(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns and map values based on mappings
    for key, replacer in symptoms_dict.items():
        df = df.replace({key: replacer['values']})
    df.rename(columns=symptoms_name_dict, inplace=True)
    return df


def build_master_data() -> None:
    df = pd.read_excel(LEUKEMIA_RAW_PATH)
    class_frames = split_by_class(df)
    master_frame = pd.concat(class_frames)
    master_frame.to_csv(LEUKEMIA_MASTER_PATH, index=False)


def split_by_class(df: pd.DataFrame) -> []:
    # find indices of rows when next class group
    indices = np.where(df['class'].notnull())[0]

    # find the number of rows in each class
    # (difference between next two indices in list)
    class_frames = []
    class_counts = [b - a for a, b in zip(indices[:-1], indices[1:])]

    # split dfs based on indices and count
    for class_id, class_count in enumerate(class_counts, start=1):
        class_df = format_class_df(df[:class_count], class_id)
        df = df[class_count:]
        class_frames.append(class_df)

    # at this stage, df contains only last class, format it as well
    class_frames.append(format_class_df(df, len(class_counts) + 1))

    return class_frames


def format_class_df(df: pd.DataFrame, class_id: int) -> pd.DataFrame:
    # get rid of every column except feature columns and assign id
    class_df = df.iloc[:, 2:22]
    class_df['class'] = class_id
    return class_df


read_data()

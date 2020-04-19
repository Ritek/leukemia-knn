import pandas as pd
from sklearn.model_selection import train_test_split

from src.reader.columns import symptoms_name_dict
from src.reader.dir import LEUKEMIA_MASTER_PATH


def feature_class_data():
    """
    Get leukemia data split into leukemia features and
    leukemia type
    :return: tuple (X,Y)

    X - leukemia data (features only)
    Y - leukemia data (leukemia type only)
    """
    leukemia_data = pd.read_csv(LEUKEMIA_MASTER_PATH)
    leukemia_data.rename(columns=symptoms_name_dict, inplace=True)
    y = leukemia_data['class'].values
    x = leukemia_data.drop('class', axis=1)
    return x, y


def train_test_data(test_size=0.3, random_state=4, proportional_split=True):
    """
    Get tuples with train/test data
    :param test_size: how much of data should be test data
    :param random_state: random seed
    :param proportional_split: split data proportionally to class occurrences
    :return: (X_train, X_test, y_train, y_test) tuple

    X_train - training data (without target parameter to predict - leukemia type)

    X_test - test data (without target parameter to predict - leukemia type)

    y_train - array of leukemia types corresponding to training data

    y_test - array of leukemia types corresponding to test data
    """
    x, y = feature_class_data()
    stratify = y if proportional_split else None;
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=stratify)

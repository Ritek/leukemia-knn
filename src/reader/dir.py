import os
from os.path import join
from pathlib import Path

READER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = str(Path(READER_DIR).parents[1])
DATA_PATH = '{}/data'.format(ROOT_PATH)
LEUKEMIA_RAW_PATH = join(DATA_PATH, 'leukemia_raw.xlsx')
LEUKEMIA_MASTER_PATH = join(DATA_PATH, 'leukemia_master.csv')

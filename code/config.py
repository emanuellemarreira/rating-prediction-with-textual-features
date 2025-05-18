import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))
DIR_TRAIN = os.path.join(DATASET_DIR, 'train.csv')
DIR_TEST = os.path.join(DATASET_DIR, 'test.csv')
SAVE_PATH = BASE_DIR

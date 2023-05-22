import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
import matplotlib.pyplot as plt
# from utils.data_utils import get_test_loaders
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import yaml
import seaborn as sns
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def set_dataset(self, dataset_name):
    with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
        return yaml.safe_load(stream)[dataset_name]


def main():
    test_data
    test_img_dir = {name: set_dataset(name) for name in test_dataset_names}
    test_number_of_people = 20
    test_celeb_lab = {}

    for dataset_name, img_dir in self.test_img_dir.items():
        label_list = os.listdir(img_dir['root_path'])[:self.test_number_of_people]
        if dataset_name == self.train_dataset_name:
            label_list = os.listdir(img_dir['root_path'])[-self.test_number_of_people:]
        self.test_celeb_lab[dataset_name] = label_list
    self.test_celeb_lab_mapper = {dataset_name: {i: lab for i, lab in enumerate(self.test_celeb_lab[dataset_name])}
                                  for dataset_name in self.test_dataset_names}
    emb_loaders, test_loaders = get_test_loaders(self.config, self.config.test_celeb_lab.keys())


print("finish")



if __name__=='__main__':
    main()

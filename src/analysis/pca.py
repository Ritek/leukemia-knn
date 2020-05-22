import sys

import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import colors as mcolors
import math

sys.path.append('C:/Users/writh/Desktop/leukemia-knn')
from src.reader.dir import LEUKEMIA_MASTER_PATH


leukemia_data = pd.read_csv(LEUKEMIA_MASTER_PATH)
leukemia_data.drop('class', axis=1, inplace=True)
pca = PCA(3)
pca.fit(leukemia_data)
pca_data = pd.DataFrame(pca.transform(leukemia_data))



print(pca_data.head())
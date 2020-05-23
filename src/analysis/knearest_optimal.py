import sys

from numpy.ma import arange
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('C:/Users/writh/Desktop/leukemia-knn')

from src.preprocessing.train_data import feature_class_data

X, Y = feature_class_data()

# Use GridSearch to find optimal value of neighbours - train model multiple times on
# range of specified parameters, to find the optimal parameter values
# Create classifier and a dictionary of all values we want to test for n_neighbors
# in this case, find optimal value N of neighbours in 2-20 range
# Also, cross validate data using K-Fold where K=5 - split dataset into K groups
# and the model will be trained and tested 5 separate times sos each group
# will get a chance to be a test set

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': arange(2, 20)}
# you can also test other parameters, such as knn algorithm
# param_grid = {'n_neighbors': arange(2, 20), 'algorithm': ['kd_tree', 'ball_tree']}
knn_gscv = GridSearchCV(knn, param_grid, cv=5, n_jobs=4)
knn_gscv.fit(X, Y)
print(knn_gscv.best_params_, knn_gscv.best_score_)

from numpy.ma import arange
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from src.preprocessing.train_data import feature_class_data

# KNN works poorly for high-dimensional data - use PCA
# to reduce dimensions
# for example PCA(10) will pick 10 most significant features
X, Y = feature_class_data()

# Define data processing pipeline
# data -> normalized data -> data with reduced dimensions
# Normalize data using Normalizer/ StandardScaler, and apply PCA to normalized data
num_features = 18
pipeline = make_pipeline(StandardScaler(), PCA(n_components=num_features))
X_pca = pipeline.fit_transform(X)

# Find out optimal N neighbours for PCA data
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': arange(2, 20)}
knn_gscv = GridSearchCV(knn, param_grid, cv=10, n_jobs=4)
knn_gscv.fit(X, Y)
knn_gscv.fit(X_pca, Y)
print(knn_gscv.best_params_, knn_gscv.best_score_)

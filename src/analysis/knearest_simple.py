import sys
import os
import pathlib

from sklearn.neighbors import KNeighborsClassifier
sys.path.append('C:/Users/writh/Desktop/leukemia-knn')

from src.preprocessing.train_data import train_test_data

X_train, X_test, y_train, y_test = train_test_data()

# basic example - train data with KNN, where for num of neighbours = 5

scores = []
N = 10
knn = KNeighborsClassifier(n_neighbors=N)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(f'Prediction score for N={N}: {score}')

num_predictions = 5
predictions = knn.predict(X_test)[0:num_predictions]
print(f'First {num_predictions} predictions: {predictions}')
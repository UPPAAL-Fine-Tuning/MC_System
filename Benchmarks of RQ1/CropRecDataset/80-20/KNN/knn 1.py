import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def create_np_array(line):
    values = [float(val) for val in line.strip().split(',')]
    return np.array(values)

with open("X_train.txt", "r") as file:
    train_lines = file.readlines()

TrainDatas = [create_np_array(line) for line in train_lines]

with open("y_train.txt", "r") as label_file:
    Trainlabels = label_file.readlines()

TrainLabel = [label.strip() for label in Trainlabels]

with open("X_test.txt", "r") as file:
    test_lines = file.readlines()

TestDatas = [create_np_array(line) for line in test_lines]

with open("y_test.txt", "r") as label_file:
    Testlabels = label_file.readlines()

TestLabel = [label.strip() for label in Testlabels]

neighbors_to_test = 400

knn = KNeighborsClassifier(n_neighbors=neighbors_to_test)

knn.fit(TrainDatas, TrainLabel)

y_pred = knn.predict(TestDatas)

accuracy = metrics.accuracy_score(TestLabel, y_pred)

print(f"Number of neighbors: {neighbors_to_test} - Accuracy: {accuracy:.6f}")

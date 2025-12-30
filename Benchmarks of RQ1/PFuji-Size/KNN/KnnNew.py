from sklearn.datasets import load_iris
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

def create_np_array(line):
    values = [float(val) for val in line.split()]
    return np.array(values)

with open("training.txt", "r") as file:
    train_lines = file.readlines()

TrainDatas = [create_np_array(line) for line in train_lines]

with open("trainingLabel.txt", "r") as label_file:
    Trainlabels = label_file.readlines()

TrainLabel = np.array([int(label.strip()) for label in Trainlabels])

with open("testing.txt", "r") as file:
    test_lines = file.readlines()

TestDatas = [create_np_array(line) for line in test_lines]

with open("testingLabel.txt", "r") as label_file:
    Testlabels = label_file.readlines()

TestLabel = np.array([int(label.strip()) for label in Testlabels])

TrainData = np.array(TrainDatas)
TestData = np.array(TestDatas)

knn = KNeighborsClassifier(n_neighbors=4)

start_time = time.time()
knn.fit(TrainData, TrainLabel)
end_time = time.time()
training_time = end_time - start_time

start_time = time.time()
y_pred = knn.predict(TestData)
end_time = time.time()
prediction_time = end_time - start_time

accuracy = metrics.accuracy_score(TestLabel, y_pred)
print("Accuracy:", accuracy)
print("Training Time:", training_time, "seconds")
print("Prediction Time:", prediction_time, "seconds")

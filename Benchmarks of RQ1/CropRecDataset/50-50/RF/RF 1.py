import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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

n_estimators = 14

rf = RandomForestClassifier(n_estimators=n_estimators, random_state=None)

rf.fit(TrainDatas, TrainLabel)

y_pred = rf.predict(TestDatas)

accuracy = metrics.accuracy_score(TestLabel, y_pred)

print(f"Number of estimators: {n_estimators} - Accuracy: {accuracy:.6f}")

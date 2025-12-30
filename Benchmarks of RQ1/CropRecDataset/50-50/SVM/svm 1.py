import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

C = 0.081

svm = SVC(kernel='linear', C=C, random_state=None)

svm.fit(TrainDatas, TrainLabel)

y_pred = svm.predict(TestDatas)

accuracy = accuracy_score(TestLabel, y_pred)
print(f"C: {C} - Accuracy: {accuracy:.6f}")

print("\nAccuracy:", accuracy)
print("C:", C)

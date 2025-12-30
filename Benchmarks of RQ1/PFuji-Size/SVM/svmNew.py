import numpy as np
from sklearn import metrics
from sklearn import svm
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

start_time = time.time()

clf = svm.SVC(kernel='poly', C=0.001)
clf.fit(TrainDatas, TrainLabel)
y_pred = clf.predict(TestDatas)

end_time = time.time()
training_time = end_time - start_time

print("Accuracy:", metrics.accuracy_score(TestLabel, y_pred))
print("Precision:", metrics.precision_score(TestLabel, y_pred))
print("Recall:", metrics.recall_score(TestLabel, y_pred))
print("F1 Score:", metrics.f1_score(TestLabel, y_pred))
print("Area Under Curve (AUC):", metrics.roc_auc_score(TestLabel, y_pred))
print("Training Time:", training_time, "seconds")

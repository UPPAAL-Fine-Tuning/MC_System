import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

start_time = time.time()

model = RandomForestClassifier(n_estimators=1, random_state=None)
model.fit(TrainData, TrainLabel)
predictions = model.predict(TestData)

end_time = time.time()
training_time = end_time - start_time

accuracy = accuracy_score(TestLabel, predictions)
precision = precision_score(TestLabel, predictions, average='macro')
recall = recall_score(TestLabel, predictions, average='macro')
f1 = f1_score(TestLabel, predictions, average='macro')

print(f'Accuracy: {accuracy:.6f}')
print(f'Precision: {precision:.6f}')
print(f'Recall: {recall:.6f}')
print(f'F1 Score: {f1:.6f}')
print("Training Time:", training_time, "seconds")

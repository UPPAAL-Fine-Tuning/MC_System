import random
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mealpy.bio_based import SMA
from mealpy.utils.space import FloatVar

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

X_train = np.array(TrainDatas)
y_train = np.array(TrainLabel)
X_test = np.array(TestDatas)
y_test = np.array(TestLabel)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

def fitness_function(solution):
    C = solution[0]
    model = SVC(C=C, gamma='scale', kernel='rbf')
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return 1 - scores.mean()

bounds = FloatVar(lb=[0.001], ub=[1000], name=["C"])

problem = {
    "obj_func": fitness_function,
    "bounds": bounds,
    "minmax": "min",
}

random.seed(42)
np.random.seed(42)
model = SMA.OriginalSMA(epoch=300, pop_size=40)

start_time = time.time()
best = model.solve(problem)
best_C = best.solution[0]

final_model = SVC(C=best_C, gamma='scale', kernel='rbf')
final_model.fit(X_train, y_train)

accuracy = final_model.score(X_test, y_test)

end_time = time.time()
execution_time = end_time - start_time

print(f"\nBest parameter found for C: C = {best_C:.4f}")
print(f"Final test accuracy: {accuracy * 100:.2f}%")
print(f"Total execution time: {execution_time:.2f} seconds")

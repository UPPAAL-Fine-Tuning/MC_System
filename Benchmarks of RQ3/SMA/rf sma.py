import random
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

cv_folds = 5

def fitness_function(solution):
    n_estimators = int(round(solution[0]))
    n_estimators = max(1, min(n_estimators, 1000))
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
    return 1 - scores.mean()

bounds = FloatVar(lb=[1], ub=[1000], name=["n_estimators"])

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
best_n_estimators = int(round(best.solution[0]))
best_n_estimators = max(1, min(best_n_estimators, 1000))

final_model = RandomForestClassifier(
    n_estimators=best_n_estimators,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train, y_train)

accuracy = final_model.score(X_test, y_test)

end_time = time.time()
execution_time = end_time - start_time

print(f"\nBest number of estimators (n_estimators): {best_n_estimators}")
print(f"Final test accuracy: {accuracy * 100:.2f}%")
print(f"Total execution time: {execution_time:.2f} seconds")

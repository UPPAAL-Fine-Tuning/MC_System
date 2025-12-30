import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

def create_np_array(line):
    values = line.strip().split(",")
    try:
        return np.array([float(value.replace(',', '.')) if value.strip() != "" else 0.0 for value in values])
    except ValueError as e:
        print(f"Line conversion error: {line} -> {e}")
        return np.zeros(len(values))

def load_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return np.array([create_np_array(line) for line in lines])

X_train = load_data("X_train.txt")
X_test = load_data("X_test.txt")

with open("y_train.txt", "r") as label_file:
    Trainlabels = label_file.readlines()
with open("y_test.txt", "r") as label_file:
    Testlabels = label_file.readlines()

label_encoder = LabelEncoder()
TrainLabel = label_encoder.fit_transform([label.strip() for label in Trainlabels])
TestLabel = label_encoder.transform([label.strip() for label in Testlabels])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def objective(C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='linear', random_state=42)
    model.fit(X_train, TrainLabel)
    y_pred = model.predict(X_test)
    return accuracy_score(TestLabel, y_pred)

pbounds = {
    "C": (0.001, 1000),
}

optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

print("Starting Bayesian optimization for SVM...")
start_time = time.time()
optimizer.maximize(init_points=40, n_iter=600)
end_time = time.time()

best_params = optimizer.max['params']
print("Best hyperparameters found:", best_params)

final_model = SVC(C=best_params['C'], kernel='linear', random_state=42)
final_model.fit(X_train, TrainLabel)

y_pred_final = final_model.predict(X_test)
accuracy_final = accuracy_score(TestLabel, y_pred_final)

print(f"Optimization runtime: {end_time - start_time:.4f} seconds")
print(f"Final SVM model accuracy: {accuracy_final:.4f}")

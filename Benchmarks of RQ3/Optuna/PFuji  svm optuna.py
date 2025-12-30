import numpy as np
import optuna
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

def create_np_array(line):
    values = line.strip().replace("\t", " ").split()
    try:
        return np.array([float(value.replace(',', '.')) if value.strip() != "" else 0.0 for value in values])
    except ValueError:
        return np.zeros(len(values))

with open("training.txt", "r") as file:
    train_lines = file.readlines()
X_train = np.array([create_np_array(line) for line in train_lines])

with open("trainingLabel.txt", "r") as label_file:
    Trainlabels = label_file.readlines()
label_encoder = LabelEncoder()
TrainLabel = label_encoder.fit_transform([label.strip() for label in Trainlabels])

with open("testing.txt", "r") as file:
    test_lines = file.readlines()
X_test = np.array([create_np_array(line) for line in test_lines])

with open("testingLabel.txt", "r") as label_file:
    Testlabels = label_file.readlines()
TestLabel = label_encoder.transform([label.strip() for label in Testlabels])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def objective(trial):
    C = trial.suggest_float("C", 0.001, 1000)
    model = SVC(kernel="rbf", C=C, random_state=42)
    model.fit(X_train, TrainLabel)
    y_pred = model.predict(X_test)
    return accuracy_score(TestLabel, y_pred)

start_time = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2000)
end_time = time.time()

best_params = study.best_params
print("Best hyperparameter C found:")
print(best_params)

final_model = SVC(kernel="linear", C=best_params["C"], random_state=42)
final_model.fit(X_train, TrainLabel)
y_pred_final = final_model.predict(X_test)
accuracy_final = accuracy_score(TestLabel, y_pred_final)

print(f"Final model accuracy: {accuracy_final:.4f}")
print(f"Optuna optimization runtime: {end_time - start_time:.2f} seconds")

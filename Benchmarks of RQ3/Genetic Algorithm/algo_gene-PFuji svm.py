import numpy as np
from deap import base, creator, tools, algorithms
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import time

def create_np_array(line):
    values = line.strip().replace("\t", " ").split()
    try:
        return np.array([float(value.replace(',', '.')) if value.strip() != "" else 0.0 for value in values])
    except ValueError as e:
        print(f"Line conversion error: {line} -> {e}")
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

def evaluate(individual):
    C = max(individual[0], 0.001)
    model = svm.SVC(C=C, kernel='linear')
    model.fit(X_train, TrainLabel)
    y_pred = model.predict(X_test)
    return accuracy_score(TestLabel, y_pred),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_C", random.uniform, 0.001, 1000)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_C,), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.001], up=[1000], eta=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

pop_size = 40
gen_count = 600
pop = toolbox.population(n=pop_size)

print("Starting genetic algorithm optimization...")
start_time = time.time()
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=gen_count, verbose=True)
end_time = time.time()

best_ind = tools.selBest(pop, k=1)[0]
best_C = max(best_ind[0], 0.001)
print("Best hyperparameters found:")
print({"C": best_C})

final_model = svm.SVC(C=best_C, gamma='scale', kernel='linear')
final_model.fit(X_train, TrainLabel)
y_pred_final = final_model.predict(X_test)
accuracy_final = accuracy_score(TestLabel, y_pred_final)

print(f"Final model accuracy: {accuracy_final:.4f}")
print(f"Genetic optimization runtime: {end_time - start_time:.4f} seconds")

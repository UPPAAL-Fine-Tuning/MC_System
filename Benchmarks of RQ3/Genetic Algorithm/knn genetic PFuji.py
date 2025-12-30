import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
from deap import base, creator, tools, algorithms
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

max_neighbors = X_train.shape[0]

def evaluate(individual):
    n_neighbors = int(max(min(individual[0], max_neighbors), 1))
    p = int(max(min(individual[1], 2), 1))
    model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
    model.fit(X_train, TrainLabel)
    y_pred = model.predict(X_test)
    return accuracy_score(TestLabel, y_pred),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_n_neighbors", random.randint, 1, max_neighbors)
toolbox.register("attr_p", random.randint, 1, 2)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attr_n_neighbors, toolbox.attr_p),
    n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=[1, 1],
    up=[max_neighbors, 2],
    eta=1.0,
    indpb=0.2
)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

pop_size = 40
gen_count = 600
pop = toolbox.population(n=pop_size)

print("Genetic optimization started...")
start_time = time.time()
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=gen_count, verbose=True)
end_time = time.time()

best_ind = tools.selBest(pop, k=1)[0]
best_n_neighbors = int(max(min(best_ind[0], max_neighbors), 1))
best_p = int(max(min(best_ind[1], 2), 1))

print("Best hyperparameters found:")
print({"n_neighbors": best_n_neighbors, "p": best_p})

final_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, p=best_p)
final_model.fit(X_train, TrainLabel)
y_pred_final = final_model.predict(X_test)
accuracy_final = accuracy_score(TestLabel, y_pred_final)

print(f"Final model accuracy: {accuracy_final:.4f}")
print(f"Genetic optimization runtime: {end_time - start_time:.4f} seconds")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import time

start_time = time.time()

DataCrop = np.genfromtxt('DataCrop.txt', delimiter=',')
LabelCrop = np.genfromtxt('LabelCrop.txt', dtype=str)

X_train, X_test, y_train, y_test = train_test_split(
    DataCrop, LabelCrop, test_size=0.2
)

model = SVC()

C_range = np.arange(10**-3, 10**3, 0.01)
param_grid = {
    'C': C_range,
    'kernel': ['linear']
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")

best_C = grid_search.best_params_['C']
best_accuracy = grid_search.best_score_

print(f"Best value of C: {best_C}")
print(f"Best cross-validation accuracy: {best_accuracy:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))

end_time = time.time()
total_time = end_time - start_time

print(f"Total runtime: {total_time:.2f} seconds")

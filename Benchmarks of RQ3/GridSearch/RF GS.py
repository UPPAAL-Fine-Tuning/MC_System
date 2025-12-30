import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time

start_time = time.time()

DataCrop = np.genfromtxt('DataCrop.txt', delimiter=',')
LabelCrop = np.genfromtxt('LabelCrop.txt', dtype=str)

scaler = StandardScaler()
DataCrop = scaler.fit_transform(DataCrop)

X_train, X_test, y_train, y_test = train_test_split(
    DataCrop, LabelCrop, test_size=0.2, stratify=LabelCrop
)

model = RandomForestClassifier()

param_grid = {
    'n_estimators': np.arange(1, 1001)
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

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation accuracy: {best_accuracy:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))

end_time = time.time()
total_time = end_time - start_time

print(f"Total runtime: {total_time:.2f} seconds")

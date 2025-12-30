import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

output_dir = "C:/Users/syrine/Downloads/loso_npy_splits_final_window"
fold_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
fold_results = []
class_names = ["Base", "TSST", "Fun"]
test_subject = "fold_S2"
fold_path = os.path.join(output_dir, test_subject)

X_train = np.load(os.path.join(fold_path, "X_train.npy"))
X_test = np.load(os.path.join(fold_path, "X_test.npy"))
y_train = np.load(os.path.join(fold_path, "y_train.npy"))
y_test = np.load(os.path.join(fold_path, "y_test.npy"))

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

print(f"\n{' CLASSIFICATION REPORT ':=^60}")
print(f"Fold: {test_subject} | Accuracy: {accuracy:.4f}\n")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix ({test_subject})\nAccuracy: {accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
print(f"\nKey Metrics:")
print(f"- Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
print(f"- Weighted Avg Precision: {report['weighted avg']['precision']:.3f}")
print(f"- Worst Class Recall: {min([report[cls]['recall'] for cls in class_names]):.3f}")

predicted_classes = Counter(y_pred)
majority_class = predicted_classes.most_common(1)[0][0]
print(f"Subject {test_subject} is classified as: {majority_class}")

results = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Subject': test_subject
})
print("\nSample-wise predictions:")
print(results.head())

results.to_csv(f'predictions_fold_{test_subject}.csv', index=False)

param_grid = {
    'randomforestclassifier__n_estimators': list(np.linspace(1, 1000, 500, dtype=int))
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print("Best Parameters:", grid_search.best_params_)
print(f"Fold: {test_subject} | Test Accuracy: {accuracy:.4f}")

print(f"\n{' CLASSIFICATION REPORT ':=^60}")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix ({test_subject})\nAccuracy: {accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
print(f"\nKey Metrics:")
print(f"- Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
print(f"- Weighted Avg Precision: {report['weighted avg']['precision']:.3f}")
print(f"- Worst Class Recall: {min([report[cls]['recall'] for cls in class_names]):.3f}")

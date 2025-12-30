Model-based Approach for Guided Parameter Exploration in Machine Learning Classifiers

This repository implements a model-based framework for guided hyperparameter exploration that integrates machine learning 
classifiers with a formal model specified in UPPAAL. The approach supports multiple classifiers, including Support Vector 
Machines (SVM), k-Nearest Neighbors (KNN), and Random Forests (RF), and enables structured, logic-driven adaptation of their 
key hyperparameters through an feedback loop between machine learning execution and formal model checking.

The framework allows the exploration of:

the regularization parameter C for SVM,

the number of neighbors k for KNN,

the number of trees (n_estimators) for Random Forest.

* Repository Structure
.
├── controller.py   # Machine learning module (SVM / KNN / RF training and evaluation)
├── script.py       # Python control script coordinating ML execution and UPPAAL
├── modele.xml      # UPPAAL formal model
├── trace.uctr      # UPPAAL execution trace
└── README_GitHub.txt

* Core Components
-Python Control Script :

The control script orchestrates the interaction between the machine learning module and the formal modeling component. It:

Extracts hyperparameter values from UPPAAL trace files.

Dynamically updates the configuration of the selected classifier.

Executes the corresponding machine learning model.

Computes evaluation metrics, including accuracy, precision, recall, F1-score, and AUC.

-Machine Learning Module (Controller.py) :

This module implements and evaluates one of multiple machine learning classifiers:

Support Vector Machine (SVM), with exploration of the regularization parameter C,

k-Nearest Neighbors (KNN), with exploration of the number of neighbors k,

Random Forest (RF), with exploration of the number of trees (n_estimators).

For each classifier :

Training and evaluation are performed on predefined datasets.

Performance metrics are computed and provided as feedback to the formal model.

-UPPAAL Model (modele.xml):

The UPPAAL model encodes the formal control logic of the hyperparameter exploration process. It:

Governs parameter updates and iteration flow.

Encodes termination conditions and performance thresholds.

Ensures that the exploration process respects predefined constraints and target performance objectives.

-Optimization Workflow :

The guided exploration process follows an iterative closed-loop structure:

Extract initial hyperparameter values from a UPPAAL execution trace.

Execute the selected classifier with the extracted configuration.

Evaluate the model and compute performance metrics.

Update the UPPAAL model with the obtained results.

Repeat the process until the formal termination conditions are satisfied.

-Objective :

The objective of the framework is to reach a target classification performance close to a predefined reference value(threshold), 
while ensuring that all explored hyperparameter configurations satisfy the constraints encoded in the formal model.

-Usage:

Install the required Python dependencies.

Ensure that modele.xml and trace.uctr are correctly referenced in the control script.

Select the desired classifier (SVM, KNN, or RF) and run the control script to start the guided hyperparameter exploration process.

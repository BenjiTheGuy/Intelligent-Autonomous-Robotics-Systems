"""
Machine Learning for Robotic Navigation

This involves training a neural network to select local navigation actions.
"""
# Importing libraries to use in file to handle data, train models, evaluation and plotting.
import json
from pathlib import Path

# Selecting a non-interactive plotting backend so figures can be saved without opening a GUI.
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Creating Project Paths and Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Ensuring rhe project output directories exist before writing generated files.
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "navigation_model.joblib"
METRICS_PATH = RESULTS_DIR / "ml_metrics.json"

# A fixed random seed makes generated samples and the train/test split repeatable.
RANDOM_STATE = 42
NUMBER_OF_SAMPLES = 6000

# The four possible classes which the classifier uses to predict.
ACTIONS = ["FORWARD", "LEFT", "RIGHT", "YIELD"]

# Generating synthetic labelled training examples for supervised learning.
def generate_training_data(number_samples=NUMBER_OF_SAMPLES):
    """
    Generating synthetic navigation states.

    Features include:
    1. Heading error in radians.
    2. Distance to the next waypoint in metres.
    3. Distance to the nearest robot in metres.
    4. Bearing of the nearest robot in relation to the current robot's heading, measured in radians.
    """

    # Using NumPy's random generator to create repeatable simulated sensor or navigation states.
    rng = np.random.default_rng(RANDOM_STATE)

    # Generating the four navigation features
    heading_error = rng.uniform(-np.pi, np.pi, number_samples)
    waypoint_distance = rng.uniform(0.05, 5.0, number_samples)
    nearest_robot_distance = rng.uniform(0.4, 5.0, number_samples)
    neighbour_bearing = rng.uniform(-np.pi, np.pi, number_samples)

    # Each row represents one navigation state, and each column is one input feature.
    X = np.column_stack([
        heading_error,
        waypoint_distance,
        nearest_robot_distance,
        neighbour_bearing
    ])

    # Assigning FORWARD to every sample, rules below will override it when needed.
    y = np.full(number_samples, "FORWARD", dtype=object)

    # Giving collision-avoidance priority when another robot is close and in front.
    should_yield = (
        (nearest_robot_distance < 1.2) & (np.abs(neighbour_bearing) < 0.9)
    )

    y[should_yield] = "YIELD"

    # Turning left when the heading error is positive, will not override YIELD.
    should_turn_left = (
        (heading_error > 0.18) & ~should_yield
    )
    y[should_turn_left] = "LEFT"

    # Turning right when the heading error is negative, will not override YIELD.
    should_turn_right = (
            (heading_error < -0.18) & ~should_yield
    )
    y[should_turn_right] = "RIGHT"

    # Returning the feature matrix X and the rule-generated class labels y.
    return X, y

# Training and Evaluating the Neural Network
def train_and_evaluate():
    X, y = generate_training_data()

    print("\nAction Distribution:")
    for action in ACTIONS:
        count = int(np.sum(y == action))
        print(f"{action}: {count}")

    # Keeping 20% of the data aside for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scaling features before training because the inputs use different numeric ranges. This model has two hidden layers
    # (32 and 16 neurons) and uses early stopping to stop training when validation performance no longer improves.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("network", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=RANDOM_STATE
        ))
    ])

    print("\nTraining Neural Network...")

    # Fitting the scaler and neural network using only the training partition.
    model.fit(X_train, y_train)

    # Predicting the labels for unseen test samples and calculating overall accuracy.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n---Model Evaluation---")
    print(f"Training Examples: {len(X_train)}")
    print(f"Test Examples: {len(X_test)}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=ACTIONS,
        zero_division=0
    ))

    # Saving the fitted pipeline, including its scaler and trained neural network.
    joblib.dump(model, MODEL_PATH)

    # Confusion Matrix compares actual labels with predicted labels for each action.
    matrix = confusion_matrix(y_test, predictions, labels=ACTIONS)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=ACTIONS)

    display.plot(values_format="d", xticks_rotation=35)

    plt.title("Navigation Action Confusion Matrix")

    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "ml_confusion_matrix.png", dpi=200)

    plt.close()

    # Plotting the network loss recorded during training to inspect learning progress.
    network = model.named_steps["network"]

    plt.figure(figsize=(10, 10))
    plt.plot(
        range(1, len(network.loss_curve_) + 1),
        network.loss_curve_
    )

    plt.xlabel("Training Iteration")
    plt.ylabel("Training Loss")
    plt.title("Neural Network Training Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "ml_training_curve.png", dpi=200)

    plt.close()

    # Storing the evaluation results in JSON so that they can be used in the report or other tools.
    metrics = {
        "data_source": "synthetic_rule_labelled",
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_accuracy": float(accuracy),
        "actions": ACTIONS,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=ACTIONS,
            zero_division=0,
            output_dict=True
        ),
        "model_path": str(MODEL_PATH)
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    print("\nSaved Files:")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Confusion Matrix: {RESULTS_DIR / 'ml_confusion_matrix.png'}")
    print(f"Training Curve: {RESULTS_DIR / 'ml_training_curve.png'}")

# Running training only when the file is executed directly, not when imported as a module.
if __name__ == "__main__":
    train_and_evaluate()
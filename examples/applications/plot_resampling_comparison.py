"""
=========================================================
Comparing resampling methods on an imbalanced dataset
=========================================================

This example compares several resampling strategies for binary
classification using the Breast Cancer Wisconsin dataset.
"""

# Authors: Kazi Samina Maraj Mumu
# License: MIT

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.datasets import make_imbalance
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 42


def evaluate_resampling_methods():
    """Compare classification performance with different resampling methods."""
    data = load_breast_cancer()

    # Create an imbalanced version of the real-world dataset.
    X, y = make_imbalance(
        data.data,
        data.target,
        sampling_strategy={0: 60, 1: 357},
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "No resampling": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Random oversampling": make_pipeline(
            StandardScaler(),
            RandomOverSampler(random_state=RANDOM_STATE),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Random undersampling": make_pipeline(
            StandardScaler(),
            RandomUnderSampler(random_state=RANDOM_STATE),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "SMOTE": make_pipeline(
            StandardScaler(),
            SMOTE(random_state=RANDOM_STATE),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
    }

    results = []

    for method_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append(
            {
                "Method": method_name,
                "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
                # Class 0 is malignant and is the minority class in this experiment.
                "F1 score": f1_score(y_test, predictions, pos_label=0),
                "Geometric mean": geometric_mean_score(
                    y_test, predictions, average="binary"
                ),
            }
        )

    return pd.DataFrame(results)


def main():
    """Run the comparison and display the results."""
    results = evaluate_resampling_methods()

    print(results.to_string(index=False))

    ax = results.plot(
        x="Method",
        y=["Balanced accuracy", "F1 score", "Geometric mean"],
        kind="bar",
        figsize=(10, 6),
        rot=0,
    )

    ax.set_title("Comparison of Resampling Methods")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0.9, 1.0)
    ax.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()

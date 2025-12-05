import pandas as pd
import numpy as np
from joblib import dump
from time import sleep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from constants import *

class MetaTrainerRegression:
    def __init__(self, path):

        print("[ INIT ] Initialising meta-trainer regression")
        self.df = pd.read_csv(path)

        # Drop task_type + label column
        self.X = self.df.drop(columns=["task_type", "best_model"])
        self.y = self.df["best_model"]   # KEEP AS STRINGS

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=1/6, random_state=42
        )

    # ----------------------------------------------------------------------

    def train_all_models(self):

        print("\n==============================")
        print(" TRAINING BASE RANDOM FOREST")
        print("==============================")

        # Base model
        base_model = RandomForestClassifier()
        base_model.fit(self.X_train, self.y_train)

        base_pred = base_model.predict(self.X_test)
        base_acc = accuracy_score(self.y_test, base_pred)

        print(f"➡ Base RF Accuracy = {base_acc:.4f}")

        # ----------------------------------------------
        # GRID SEARCH (Deterministic Tuning)
        # ----------------------------------------------

        print("\n==============================")
        print(" GRID SEARCH TUNED RANDOM FOREST")
        print("==============================")

        param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", None],
            "bootstrap": [True, False]
        }

        grid = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(self.X_train, self.y_train)
        tuned_model = grid.best_estimator_

        print("\n🏆 Best Tuned Params:")
        print(grid.best_params_)

        # Tuned accuracy
        tuned_pred = tuned_model.predict(self.X_test)
        tuned_acc = accuracy_score(self.y_test, tuned_pred)

        print(f"➡ Tuned RF Accuracy = {tuned_acc:.4f}")

        # ------------------------------------------------------------
        # Pick BEST MODEL (base or tuned)
        # ------------------------------------------------------------
        if tuned_acc >= base_acc:
            print("\n⭐ Using TUNED RandomForest as final model")
            final_model = tuned_model
        else:
            print("\n⭐ Using BASE RandomForest as final model")
            final_model = base_model

        dump(final_model,META_REGRESSION_MODEL)
        print("✔ Model saved as meta_regression_model.pkl")

        return final_model



if __name__ == "__main__":
    path = "meta_model/meta_learning/meta_regression/meta_features_regression.csv"
    trainer = MetaTrainerRegression(path)
    trainer.train_all_models()

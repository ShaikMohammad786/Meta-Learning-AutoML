import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.getcwd()))

from constants import *
import pandas as pd
import numpy as np
from joblib import dump
from components.preprocessing import Preproccessor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from lime.lime_tabular import LimeTabularExplainer
from user_section.prediction.classification_prediction import MetaClassificationPredictor
from user_section.training.bundle_exporter import export_user_bundle
from user_section.training.model_explanations import describe_model


class UserClassificationTrainer:

    def __init__(
        self,
        predictor:MetaClassificationPredictor,
        user_id,
        dataset_name,
        task_type,
        status_tracker=None
    ):

        self.best_models = predictor.top_models
        self.X_train = predictor.X_train
        self.y_train = predictor.y_train
        self.X_test = predictor.X_test
        self.y_test = predictor.y_test
        self.X_val = predictor.X_val
        self.y_val = predictor.y_val
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.preprocessor = predictor.preprocessor
        self.status_tracker = status_tracker

        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "SVC": SVC(probability=True),
            "RandomForest": RandomForestClassifier(n_jobs=-1),
            "GradientBoosting": HistGradientBoostingClassifier(early_stopping=False),
        }

        self.param_grids = {
            "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
            "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "DecisionTree": {"max_depth": [5, 10, None]},
            "SVC": {"C": [0.1, 1, 10]},
            "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
            "GradientBoosting": {
                "learning_rate": [0.05, 0.1],
                "max_iter": [200, 300],
                "max_leaf_nodes": [31, 63],
            },
        }

        self.feature_names = list(self.X_train.columns)
        self.metadata_dir = Path(f"{USERS_FOLDER}/{self.user_id}/models/classification")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _generate_lime_explanations(self, estimator):
        try:
            sample_frame = self.X_test if len(self.X_test) else self.X_train
            sample = sample_frame.iloc[0].values

            explainer = LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                class_names=[str(cls) for cls in np.unique(self.y_train)],
                discretize_continuous=True,
                mode="classification",
            )
            predict_fn = (
                estimator.predict_proba
                if hasattr(estimator, "predict_proba")
                else None
            )

            if predict_fn is None:
                return []

            explanation = explainer.explain_instance(
                data_row=sample,
                predict_fn=predict_fn,
                num_features=min(8, len(self.feature_names)),
            )

            return [
                {"feature": feat, "weight": float(weight)}
                for feat, weight in explanation.as_list()
            ]
        except Exception as error:
            print(f"[LIME] Failed to generate explanation: {error}")
            return []

    def _persist_metadata(self, model_label, test_accuracy, explanations, model_reason, human_metric_text):
        metadata = {
            "model_name": model_label,
            "metric_name": "accuracy",
            "metric_value": float(test_accuracy),
            "explanations": explanations,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_reason": model_reason,
            "human_metric": human_metric_text,
        }
        meta_path = self.metadata_dir / f"{self.dataset_name}.meta.json"
        with meta_path.open("w", encoding="utf-8") as handler:
            json.dump(metadata, handler, indent=2)

    def train_and_tune_model(self, Tuning=False):

        if(self.task_type!="classification"):
            raise ValueError("Task type is not classification !")
            

        all_results = []

        for model in self.best_models:

            print(f"[TRAIN] {model} ")

            base_model = self.models[model]

            params = self.param_grids[model]

            if Tuning:

                grid = GridSearchCV(
                    estimator=base_model,
                    param_grid=params,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                )

                grid.fit(self.X_train, self.y_train)

                best_est = grid.best_estimator_
                val_acc = best_est.score(self.X_val, self.y_val)

                all_results.append({"Estimator": best_est, "Accuracy": val_acc, "Model": model})
            else:
                base_model.fit(self.X_train, self.y_train)
                new_test_x = pd.concat([self.X_val, self.X_test], axis=0)
                new_test_y = pd.concat([self.y_val, self.y_test], axis=0)
                val_acc = base_model.score(new_test_x, new_test_y)
                all_results.append({"Estimator": base_model, "Accuracy": val_acc, "Model": model})

        final_model = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False)
        best_row = final_model.iloc[0]
        best_model_name = best_row["Model"]
        best_estimator = best_row["Estimator"]

        print(
            f"\n[SELECT] Best Model: {best_row['Estimator']} (Accuracy={best_row['Accuracy']:.4f})"
        )

        path = f"{USERS_FOLDER}/{self.user_id}/models/classification"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/{self.dataset_name}.pkl"
        dump(best_estimator, save_path)
        print(f"💾 Saved Best Model → {save_path}")

        eval_X = self.X_test
        eval_y = self.y_test
        if not Tuning:
            eval_X = pd.concat([self.X_val, self.X_test], axis=0)
            eval_y = pd.concat([self.y_val, self.y_test], axis=0)

        test_accuracy = float(best_estimator.score(eval_X, eval_y))
        explanations = self._generate_lime_explanations(best_estimator)
        model_story = describe_model("classification", best_model_name, test_accuracy)
        self._persist_metadata(
            best_model_name,
            test_accuracy,
            explanations,
            model_story["explanation"],
            model_story["metric_text"],
        )
        if self.status_tracker:
            self.status_tracker.update("packaging", "Saving classifier and building the bundle.")
        bundle_path = export_user_bundle(
            task_type="classification",
            user_id=self.user_id,
            dataset_name=self.dataset_name,
            model_path=save_path,
            preprocessor=self.preprocessor,
        )

        return {
            "best_model_name": best_model_name,
            "best_model_path": save_path,
            "bundle_path": str(bundle_path),
            "all_results": all_results,
            "test_accuracy": test_accuracy,
            "explanations": explanations,
            "model_reason": model_story["explanation"],
            "human_metric": model_story["metric_text"],
        }




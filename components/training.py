import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import pandas as pd
import numpy as np
import random
from components.preprocessing import Preproccessor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Lasso,
    Ridge,
    ElasticNet,
)
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
)
from typing import Dict, Any, Optional, Iterable

# from meta.meta_features_extraction import meta_features_extract_class,meta_features_extract_reg,meta_features_extract_clust
from components.meta_features_extraction import (
    meta_features_extract_class,
    meta_features_extract_reg,
    meta_features_extract_clust,
)


class Classification_Training:

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        dataset_path,
        target_col,
        tuning: bool = False,
    ):  

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.tuning = tuning  # <── store flag
        self.results = []
        n_samples = self.X_train.shape[0]
        LARGE_DATASET_THRESHOLD = 50000

        print(f"[INFO] Training samples: {n_samples}")

        # Auto switch between SVR and LinearSVR
        if n_samples > LARGE_DATASET_THRESHOLD:
            print("[AUTO] Dataset is large → Switching SVC to LinearSVC")
            svc_model = LinearSVC()
            svm_name = "LinearSVC"
        else:
            svc_model = SVC()
            svm_name = "SVC"
        # Base models
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1),
            "KNN": KNeighborsClassifier(n_jobs=-1),
            "DecisionTree": DecisionTreeClassifier(),
            svm_name: svc_model,
            "RandomForest": RandomForestClassifier(n_jobs=-1),
            "GradientBoosting": HistGradientBoostingClassifier(early_stopping=False),
        }

        # Param grids for tuning
        self.param_grids = {
            "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
            "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "DecisionTree": {"max_depth": [5, 10, None]},
            "SVM": {"C": [0.1, 1, 10]},
            "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
            "GradientBoosting": {
                "learning_rate": [0.05, 0.1],
                "max_iter": [200, 300],
                "max_leaf_nodes": [31, 63],
            },
        }

    # ===========================================================
    # TRAINING PIPELINE
    # ===========================================================
    def train_model(self):

        print("\n[START] Classification Training Pipeline Initiated...\n")
        print(f"[INFO] Tuning mode = {self.tuning}\n")

        all_results = []

        # --------------------------------------------------------
        # Loop through models
        # --------------------------------------------------------
        for name, model in self.models.items():
            print(f"[TRAIN] {name}")

            # -------------------------
            # TUNING MODE
            # -------------------------
            if self.tuning and name in self.param_grids:
                print(f"[TUNE] GridSearchCV for {name}")

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=self.param_grids[name],
                    scoring="accuracy",
                    cv=3,
                    n_jobs=-1,
                )

                start = time.time()
                grid.fit(self.X_train, self.y_train)
                end = time.time()

                best_est = grid.best_estimator_
                val_acc = accuracy_score(self.y_val, best_est.predict(self.X_val))

                all_results.append(
                    {
                        "Model": name,
                        "Accuracy": val_acc,
                        "Time": round(end - start, 4),
                        "TrainedModel": best_est,
                    }
                )

                print(f"  -> Tuned Accuracy={val_acc:.4f}\n")
                continue

            # -------------------------
            # FAST MODE (NO TUNING)
            # -------------------------
            start = time.time()
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_val)
            end = time.time()

            acc = accuracy_score(self.y_val, preds)
            train_time = round(end - start, 4)

            all_results.append(
                {
                    "Model": name,
                    "Accuracy": acc,
                    "Time": train_time,
                    "TrainedModel": model,
                }
            )

            print(f"  -> Accuracy={acc:.4f}, Time={train_time}s\n")

        # --------------------------------------------------------
        # SELECT BEST MODEL
        # --------------------------------------------------------
        df_tmp = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False)
        best_row = df_tmp.iloc[0]

        print(
            f"\n[SELECT] Best Model: {best_row['Model']} (Accuracy={best_row['Accuracy']:.4f})"
        )

        # --------------------------------------------------------
        # SAVE META ROW
        # --------------------------------------------------------
        dataset_name = os.path.basename(self.dataset_path)

        save_path = "meta_dataset_results.csv"

        # --------------------------------------------------------
        # META-FEATURE EXTRACTION
        # --------------------------------------------------------
        meta_features_extract_class(
            self.X_train,
            self.y_train,
            best_row["Model"],
            pd.read_csv(self.dataset_path),
        )

        # --------------------------------------------------------
        # REFIT BEST MODEL ON FULL TRAINING DATA
        # --------------------------------------------------------
        print("[FINAL TRAIN] Re-training best model on full training data...")
        best_model_instance = best_row["TrainedModel"]
        best_model_instance.fit(self.X_train, self.y_train)

        # --------------------------------------------------------
        # PREDICT ON TEST SET
        # --------------------------------------------------------
        print("[PREDICT] Making test predictions...")
        test_predictions = best_model_instance.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)

        # print(f"[TEST] Final Test Accuracy = {test_accuracy:.4f}")
        save_row = pd.DataFrame(
            [
                {
                    "dataset_name": dataset_name,
                    "task_type": "Classification",
                    "best_model": best_row["Model"],
                    "score": test_accuracy,
                    "train_time_sec": best_row["Time"],
                }
            ]
        )

        if os.path.exists(save_path):
            save_row.to_csv(save_path, mode="a", header=False, index=False)
        else:
            save_row.to_csv(save_path, index=False)

        print("[META] Saved minimal meta row.\n")

        return self


class Regression_Training:

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        dataset_path,
        target_col,
        tuning: bool = False,
    ):

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.tuning = tuning  # <── save the flag
        self.results = []

    # --------------------
    # Tuning helper
    # --------------------
    def evaluate_model(self, grid: GridSearchCV, name):
        grid.fit(self.X_train, self.y_train)
        best_model = grid.best_estimator_
        preds = best_model.predict(self.X_val)

        self.results.append(
            {
                "Model": name,
                "Best Params": grid.best_params_,
                "R2": r2_score(self.y_val, preds),
                "MAE": mean_absolute_error(self.y_val, preds),
                "RMSE": root_mean_squared_error(self.y_val, preds),
                "TrainedModel": best_model,
            }
        )

        print(self.results[-1])
        return self

    # --------------------
    # Main pipeline
    # --------------------
    def train_model(self):
        print("\n[START] Regression Training Pipeline Initiated...\n")

        n_samples = self.X_train.shape[0]
        LARGE_DATASET_THRESHOLD = 50000

        print(f"[INFO] Training samples: {n_samples}")

        # Auto switch between SVR and LinearSVR
        if n_samples > LARGE_DATASET_THRESHOLD:
            print("[AUTO] Dataset is large → Switching SVR to LinearSVR")
            svr_model = LinearSVR()
            svm_name = "LinearSVR"
        else:
            svr_model = SVR()
            svm_name = "SVR"

        # ===============================
        # BASE MODELS
        # ===============================
        model_pipelines = {
            "LinearRegression": Pipeline([("regressor", LinearRegression())]),
            "PolynomialRegression": Pipeline(
                [
                    ("PolyFeatures", PolynomialFeatures(degree=2)),
                    ("regressor", LinearRegression()),
                ]
            ),
            "Ridge": Pipeline([("regressor", Ridge())]),
            "Lasso": Pipeline([("regressor", Lasso(max_iter=10000))]),
            "ElasticNet": Pipeline([("regressor", ElasticNet(max_iter=10000))]),
            "KNN": Pipeline([("regressor", KNeighborsRegressor())]),
            "DecisionTree": Pipeline([("regressor", DecisionTreeRegressor())]),
            svm_name: Pipeline([("regressor", svr_model)]),
            "RandomForest": Pipeline([("regressor", RandomForestRegressor(n_jobs=-1))]),
            "GradientBoosting": Pipeline(
                [("regressor", HistGradientBoostingRegressor(early_stopping=False))]
            ),
        }

        # ===============================
        # PARAM GRIDS (used only if tuning=True)
        # ===============================
        param_grids = {
            "Ridge": {"regressor__alpha": [0.1, 1, 10]},
            "Lasso": {"regressor__alpha": [0.001, 0.01, 0.1, 1]},
            "ElasticNet": {
                "regressor__alpha": [0.01, 0.1],
                "regressor__l1_ratio": [0.1, 0.5],
            },
            "KNN": {
                "regressor__n_neighbors": [3, 5, 7],
                "regressor__weights": ["uniform", "distance"],
            },
            "DecisionTree": {"regressor__max_depth": [5, 10, None]},
            "SVR_or_LinearSVR": {"regressor__C": [0.1, 1, 10]},
            "RandomForest": {"regressor__n_estimators": [100, 200]},
            "HistGradientBoosting": {
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_iter": [200, 300],
                "regressor__max_leaf_nodes": [31, 63],
            },
        }

        print(f"[INFO] Tuning mode = {self.tuning}\n")

        all_results = []

        for name, model in model_pipelines.items():
            print(f"[TRAIN] {name}")

            # -----------------------------
            # TUNING MODE
            # -----------------------------
            if self.tuning and name in param_grids:
                print(f"[TUNE] Running GridSearchCV for {name}")

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[name],
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                )

                self.evaluate_model(grid, name)
                continue

            # -----------------------------
            # FAST MODE (NO TUNING)
            # -----------------------------
            start = time.time()
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_val)
            end = time.time()

            R2 = r2_score(self.y_val, preds)
            train_time = round(end - start, 4)

            result = {
                "Model": name,
                "R2": R2,
                "Time": train_time,
                "TrainedModel": model,
            }

            all_results.append(result)

            print(f"  -> R2={R2:.4f}, Time={train_time}s\n")

        # -------------------------
        # Select best model
        # -------------------------
        df_tmp = pd.DataFrame(all_results).sort_values("R2", ascending=False)
        best_row = df_tmp.iloc[0]

        print(
            f"\n[SELECT] Best Model = {best_row['Model']} (R2={best_row['R2']:.4f})\n"
        )

        # -------------------------
        # Save meta row
        # -------------------------
        dataset_name = os.path.basename(self.dataset_path)

        save_path = "meta_dataset_results.csv"

        # --------------------------------------------------------
        # META-FEATURE EXTRACTION
        # --------------------------------------------------------
        meta_features_extract_reg(
            self.X_train,
            self.y_train,
            best_row["Model"],
            pd.read_csv(self.dataset_path),
        )

        # -------------------------
        # Final train on full X_train
        # -------------------------
        print("[FINAL TRAIN] Training best model on full training set...")
        best_model_instance = best_row["TrainedModel"]
        best_model_instance.fit(self.X_train, self.y_train)

        # -------------------------
        # Predict on test set
        # -------------------------
        test_predictions = best_model_instance.predict(self.X_test)
        test_r2 = r2_score(self.y_test, test_predictions)

        print(f"[TEST] Final Test R2 = {test_r2:.4f}")
        save_row = pd.DataFrame(
            [
                {
                    "dataset_name": dataset_name,
                    "task_type": "Regression",
                    "best_model": best_row["Model"],
                    "score": best_row["R2"],
                    "train_time_sec": best_row["Time"],
                }
            ]
        )
        if os.path.exists(save_path):
            save_row.to_csv(save_path, mode="a", header=False, index=False)
        else:
            save_row.to_csv(save_path, index=False)

        print("[META] Minimal model summary saved.\n")
        return self


class ClusteringTrainer:

    def __init__(self, X, random_state: int = 42):

        self.X = X if not isinstance(X, pd.DataFrame) else X.values
        self.random_state = random_state

        # Candidate algorithms (factory functions returning new estimators)
        self.algos = {
            "KMeans": lambda **p: KMeans(random_state=self.random_state, **p),
            "DBSCAN": lambda **p: DBSCAN(**p),
            "Agglomerative": lambda **p: AgglomerativeClustering(**p),
        }

        # Default parameter grids (small, extend as needed)
        self.param_grids = {
            "KMeans": {"n_clusters": [2, 3, 4, 5, 8, 10]},
            "DBSCAN": {"eps": [0.3, 0.5, 0.8, 1.0], "min_samples": [3, 5, 8]},
            "Agglomerative": {
                "n_clusters": [2, 3, 4, 5, 8],
                "linkage": ["ward", "complete", "average"],
            },
        }

        # results containers
        self.baselines: Dict[str, Any] = {}
        self.tuned: Dict[str, Any] = {}
        self.results: pd.DataFrame = pd.DataFrame()
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.best_score: float = -np.inf
        self.best_metric: str = "silhouette"  # default selection metric

    # helpers: scoring

    def _score_labels(self, X, labels):
        """Compute internal clustering metrics. Labels of -1 (noise) are accepted."""
        # Need at least 2 clusters for silhouette and CH; DB requires >=1 cluster.
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        scores = {"n_clusters": n_clusters}

        if n_clusters >= 2:
            try:
                scores["silhouette"] = float(silhouette_score(X, labels))
            except Exception:
                scores["silhouette"] = float("nan")
            try:
                scores["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            except Exception:
                scores["calinski_harabasz"] = float("nan")
        else:
            scores["silhouette"] = float("nan")
            scores["calinski_harabasz"] = float("nan")

        # Davies-Bouldin exists for n_clusters >= 2 as well, smaller is better
        if n_clusters >= 2:
            try:
                scores["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            except Exception:
                scores["davies_bouldin"] = float("nan")
        else:
            scores["davies_bouldin"] = float("nan")

        return scores

    # baseline fitting

    def fit_baselines(self, verbose: bool = True):
        """
        Fit baseline versions of each algorithm (first grid value or sensible defaults)
        and compute cluster metrics.
        """
        rows = []
        for name, factory in self.algos.items():
            # pick a default param set: first item in param grid or empty
            grid = self.param_grids.get(name, {})
            default_params = {k: v[0] for k, v in grid.items()} if grid else {}
            model = factory(**default_params)
            # KMeans/Agglomerative need n_clusters; ensure sensible default if not present
            try:
                labels = model.fit_predict(self.X)
            except Exception:
                # Some algorithms (e.g. DBSCAN) may not have fit_predict method for edge cases
                model.fit(self.X)
                labels = (
                    model.labels_
                    if hasattr(model, "labels_")
                    else model.predict(self.X)
                )
            scores = self._score_labels(self.X, labels)
            rows.append({"algo": name, "params": default_params, **scores})
            self.baselines[name] = {
                "model": clone(model),
                "labels": labels,
                "scores": scores,
            }
            if verbose:
                print(
                    f"[BASELINE] {name} params={default_params} -> n_clusters={scores['n_clusters']}, silhouette={scores['silhouette']}"
                )
        self.results = pd.DataFrame(rows)
        return self.results.sort_values(by="silhouette", ascending=False).reset_index(
            drop=True
        )

    def search(
        self,
        algo_name: str,
        search_type: str = "grid",
        n_iter: int = 20,
        score_metric: str = "silhouette",
        random_state: Optional[int] = None,
        verbose: bool = True,
    ):

        if random_state is None:
            random_state = self.random_state

        if algo_name not in self.algos:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        grid = self.param_grids.get(algo_name, {})
        if not grid:
            raise ValueError(f"No parameter grid defined for {algo_name}")

        # build list of candidate param dicts
        keys = list(grid.keys())
        all_candidates = []
        if search_type == "grid":
            # cartesian product (simple)
            import itertools

            for vals in itertools.product(*(grid[k] for k in keys)):
                all_candidates.append(dict(zip(keys, vals)))
        else:
            # random sampling without replacement
            candidates_set = set()
            attempts = 0
            max_attempts = max(n_iter * 10, 1000)
            while len(all_candidates) < n_iter and attempts < max_attempts:
                cand = {}
                for k in keys:
                    cand[k] = random.choice(grid[k])
                tup = tuple(sorted(cand.items()))
                if tup not in candidates_set:
                    candidates_set.add(tup)
                    all_candidates.append(cand)
                attempts += 1

        rows = []
        best_score = -np.inf if score_metric != "davies_bouldin" else np.inf
        best_model = None
        best_labels = None
        best_params = None

        for params in all_candidates:
            # instantiate model
            model = self.algos[algo_name](**params)
            try:
                labels = model.fit_predict(self.X)
            except Exception:
                model.fit(self.X)
                labels = getattr(model, "labels_", None)
                if labels is None and hasattr(model, "predict"):
                    labels = model.predict(self.X)
                if labels is None:
                    # skip if we can't get labels
                    continue

            scores = self._score_labels(self.X, labels)
            # choose metric comparison
            metric_val = scores.get(score_metric)
            if score_metric == "davies_bouldin":
                is_better = metric_val < best_score
            else:
                is_better = metric_val > best_score

            if is_better or best_model is None:
                best_score = metric_val
                best_model = clone(model)
                best_labels = labels
                best_params = params

            rows.append({"algo": algo_name, "params": params, **scores})

            if verbose:
                print(
                    f"[TRY] {algo_name} params={params} -> n_clusters={scores['n_clusters']}, silhouette={scores['silhouette']}, DB={scores['davies_bouldin']:.4f}"
                )

        df_results = (
            pd.DataFrame(rows)
            .sort_values(by="silhouette", ascending=False)
            .reset_index(drop=True)
        )
        # store best
        self.tuned[algo_name] = {
            "model": best_model,
            "params": best_params,
            "labels": best_labels,
            "score": best_score,
            "metric": score_metric,
        }
        # also append to global results
        self.results = pd.concat(
            [self.results, df_results], ignore_index=True, sort=False
        ).reset_index(drop=True)
        if verbose:
            print(
                f"[BEST] {algo_name} best_params={best_params} best_{score_metric}={best_score}"
            )
        return df_results

    # select best across algorithms

    def select_best(self, metric: str = "silhouette"):
        """
        Select the best tuned model (or baseline if not tuned) based on metric.
        metric: 'silhouette', 'davies_bouldin' (lower is better), or 'calinski_harabasz'
        """
        best = None
        best_val = -np.inf if metric != "davies_bouldin" else np.inf
        best_name = None
        best_obj = None

        # check tuned first, then baselines
        candidates = []
        for name in set(list(self.tuned.keys()) + list(self.baselines.keys())):
            if name in self.tuned and self.tuned[name]["model"] is not None:
                entry = self.tuned[name]
                score = entry["score"]
            else:
                entry = self.baselines.get(name)
                score = (
                    entry["scores"].get(metric) if entry is not None else float("nan")
                )

            if score is None or (isinstance(score, float) and np.isnan(score)):
                continue

            if metric == "davies_bouldin":
                better = score < best_val
            else:
                better = score > best_val

            if better or best is None:
                best = entry
                best_val = score
                best_name = name

        self.best_model_name = best_name
        self.best_model = best["model"] if best is not None else None
        self.best_score = best_val
        self.best_metric = metric
        print(
            f"[SELECT] Best algorithm: {self.best_model_name} (metric={metric}, value={best_val})"
        )
        return {
            "best_name": self.best_model_name,
            "best_model": self.best_model,
            "best_score": self.best_score,
        }

    def get_labels(self, model_obj, X=None):

        if X is None:
            X = self.X
        try:
            return model_obj.fit_predict(X)
        except Exception:
            try:
                model_obj.fit(X)
                return getattr(model_obj, "labels_", None)
            except Exception:
                raise RuntimeError("Model cannot produce labels on given data.")


if __name__ == "__main__":
    # Example usage of the Preproccessor
    # Uses run_preprocessing() which is a legacy method maintained for backward compatibility
    # It internally calls fit() and transform() following sklearn conventions
    
    # dataset_path = "datasets/regression/synthetic_car_prices.csv"
    # preprocessor = Preproccessor(dataset_path, "Price")
    # X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
    #     preprocessor.run_preprocessing()
    # )
    # trainer = Regression_Training(X_train, y_train, X_test, y_test, X_val, y_val, dataset_path, "Price")
    # trainer.train_model()

    dataset_path = "gender_classification.csv"
    preprocessor = Preproccessor(dataset_path, "gender")
    X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
        preprocessor.run_preprocessing()
    )
    trainer = Classification_Training(
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        dataset_path=dataset_path,
        target_col="gender",
    )
    trainer.train_model()

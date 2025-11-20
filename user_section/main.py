import pandas as pd
import os
import sys
from typing import Optional

sys.path.append(os.path.abspath(os.getcwd()))
from user_section.training.user_regression_training import UserRegressionTrainer
from user_section.training.user_classification_training import UserClassificationTrainer
from user_section.prediction.regression_prediction import MetaRegressionPredictor
from user_section.prediction.classification_prediction import (
    MetaClassificationPredictor,
)
from user_section.training.status_tracker import TrainingStatusTracker
from constants import *


class User:
    def __init__(
        self, dataset_path, user_id, target_col, tuning, task_type, dataset_name, status_tracker: Optional[TrainingStatusTracker] = None
    ):
        self.dataset_path = dataset_path
        self.user_id = user_id
        self.target_col = target_col
        self.tuning = tuning
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.status_tracker = status_tracker

    def _status(self, phase, message):
        if self.status_tracker:
            self.status_tracker.update(phase, message)

    def regression(self):
        self._status("preprocessing", "Cleaning data and preparing train/validation/test splits.")
        predictor = MetaRegressionPredictor(
            dataset_path=self.dataset_path,
            tuning=self.tuning,
            target_col=self.target_col,
        )

        predictor.run_pipeline(task_type=self.task_type)
        self._status(
            "meta_learning",
            f"Meta learner shortlisted {len(predictor.top_models)} regression models.",
        )

        trainer = UserRegressionTrainer(
            predictor, self.user_id, self.tuning, self.dataset_name, status_tracker=self.status_tracker
        )

        output = trainer.train_and_tune_model()
        self._status("training_complete", "Finished training and evaluating the candidate regressors.")
        print(output)
        return output

    def classification(self):
        self._status("preprocessing", "Processing dataset for classification.")
        predictor = MetaClassificationPredictor(
            dataset_path=self.dataset_path,
            tuning=self.tuning,
            target_col=self.target_col,
        )

        predictor.run_pipeline(task_type=self.task_type)
        self._status(
            "meta_learning",
            f"Meta learner shortlisted {len(predictor.top_models)} classification models.",
        )

        trainer = UserClassificationTrainer(
            predictor, self.user_id, self.dataset_name, self.task_type, status_tracker=self.status_tracker
        )

        output = trainer.train_and_tune_model(self.tuning)
        self._status("training_complete", "Finished training and evaluating the candidate classifiers.")
        print(output)
        return output

    def start(self):
        try:
            if self.task_type == "regression":
                output = self.regression()
                new_entry = {
                    "user_id": self.user_id,
                    "dataset_name": self.dataset_name,
                    "dataset_path": self.dataset_path,
                    "target_col": self.target_col,
                }

                os.makedirs(
                    os.path.dirname(PENDING_DATSETS_REGRESSION_FILE), exist_ok=True
                )

                df = pd.read_csv(PENDING_DATSETS_REGRESSION_FILE)
                df = pd.concat([df, pd.DataFrame([new_entry])], axis=0, ignore_index=True)
                df.to_csv(PENDING_DATSETS_REGRESSION_FILE, index=False)
            elif self.task_type == "classification":
                output = self.classification()
                new_entry = {
                    "user_id": self.user_id,
                    "dataset_name": self.dataset_name,
                    "dataset_path": self.dataset_path,
                    "target_col": self.target_col,
                }

                os.makedirs(
                    os.path.dirname(PENDING_DATSETS_CLASSIFICATION_FILE), exist_ok=True
                )

                df = pd.read_csv(PENDING_DATSETS_CLASSIFICATION_FILE)
                df = pd.concat([df, pd.DataFrame([new_entry])], axis=0, ignore_index=True)
                df.to_csv(PENDING_DATSETS_CLASSIFICATION_FILE, index=False)
            else:
                raise ValueError("Error with task!")

            if self.status_tracker:
                self.status_tracker.complete("Model ready and bundle packaged.")
            return output
        except Exception as e:
            if self.status_tracker:
                self.status_tracker.error(str(e))
            print(
                "We are not able to process your dataset right now!! Our best minds are working on it "
            )
            print(e)
            raise



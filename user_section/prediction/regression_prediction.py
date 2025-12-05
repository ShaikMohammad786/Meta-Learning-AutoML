print("Hello")
import pandas as pd
import numpy as np
import joblib
import sys, os
from joblib import load

sys.path.append(os.path.abspath(os.getcwd()))
from components.preprocessing import Preproccessor
from components.meta_features_extraction import meta_features_extract_reg

from constants import *

print("Hello")


class MetaRegressionPredictor:
    def __init__(self, dataset_path, tuning=False, target_col="target"):
        self.df = pd.read_csv(dataset_path)
        self.dataset_path = dataset_path
        self.tuning = tuning
        self.target_col = target_col
        self.top_models = None
        self.top_conf = None
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.preprocessor = None
        self.task_type = None
        
    def check_task_type(self, task_type):
        if self.task_type != task_type:
            raise ValueError(f"This dataset is not suitable for {task_type.value}")

    def preprocess(self):
        """Preprocess the dataset using the new sklearn-style API."""
        try:
            preprocessor = Preproccessor(
                dataframe=self.dataset_path, target_col=self.target_col
            )
            self.preprocessor = preprocessor
            # Use the legacy method for backward compatibility
            # This internally calls fit() and transform()
            (
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.X_val,
                self.y_val,
                self.task_type,
            ) = preprocessor.run_preprocessing()
            print("[ SUCCESS ] Preprocessing complete")
        except Exception as e:
            print(f"[ ERROR ] Preprocessing Failed ! {e}")

    def extract_features(self):
        try:
            print("[ TRY ] Trying to Extract Metafeatures")
            self.meta_data = meta_features_extract_reg(
                self.X_train, self.y_train, None, pd.read_csv(self.dataset_path)
            )
            print("[ SUCCESS ] Meta Features Extracted Successfully")
        except Exception as e:
            print(f"[ ERROR ] Meta Features Extraction Failed ! {e}")

    def predict(self):
        try:
            print("[ TRY ] Trying to load the model")
            model = load(META_REGRESSION_MODEL)
            print("[ SUCCESS ] Model Loaded Successfully!")
            probs = model.predict_proba(
                self.meta_data.drop(columns=["task_type", "best_model"])
            )[0]
            top_idx = probs.argsort()[-2:][::-1]
            self.top_models = model.classes_[top_idx]
            self.top_conf = probs[top_idx]
            print(f"Predicted Models: {self.top_models}")
        except Exception as e:
            print(f"[ ERROR ] Model Prediction Failed ! {e}")

    def run_pipeline(self,task_type):
        self.preprocess()
        self.check_task_type(task_type)
        self.extract_features()
        self.predict()

        return self


if __name__ == "__main__":
    print("[ START ] Starting Model")
    meta_model = MetaRegressionPredictor(
        r"datasets\regression\aqi.csv", target_col="aqi_value"
    )
    print("[ RUN ] RUNNING Model")
    meta_model.run()

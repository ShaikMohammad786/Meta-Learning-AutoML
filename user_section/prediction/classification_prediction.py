import sys, os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
from joblib import load
from components.meta_features_extraction import meta_features_extract_class
from components.preprocessing import Preproccessor

import numpy as np
from constants import *


class MetaClassificationPredictor:
    def __init__(self, dataset_path, target_col, tuning):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.meta_model_path = META_CLASSIFICATION_MODEL

        self.user_dataset = pd.read_csv(dataset_path)
        self.preprocessor = Preproccessor(dataframe=dataset_path, target_col=target_col)

        self.meta_model = None
        self.meta_row = None
        self.explainer = None
        self.tuning=tuning

    def preprocess(self):
        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.X_val,
            self.y_val,
            self.task_type,
        ) = self.preprocessor.run_preprocessing()

        return self
    
    def check_task_type(self,task_type):
        if(self.task_type !=task_type):
            raise ValueError(f"This dataset is not suitable for {task_type.value}")


    def extract_meta_features(self):
        self.meta_row = meta_features_extract_class(
            X_train=self.X_train,
            y_train=self.y_train,
            best_model=None,
            raw_df=self.user_dataset,
            save=False,
        )
        # keep as DataFrame with a single row
        if isinstance(self.meta_row, pd.Series):
            self.meta_row = self.meta_row.to_frame().T

        self.meta_row = self.meta_row.drop(
            columns=["best_model", "task_type"], errors="ignore"
        )

    def load_meta_model(self):
        self.meta_model = load(self.meta_model_path)

    def get_probabilities(self):
        probs = self.meta_model.predict_proba(self.meta_row)[0]
        classes = self.meta_model.classes_
        out = [(c, p) for c, p in zip(classes, probs)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def predict_top2(self):
        probs = self.get_probabilities()
        return probs[:2]

    def run_pipeline(self,task_type):
        self.preprocess()
        self.check_task_type(task_type)
        self.extract_meta_features()
        self.load_meta_model()
        self.get_probabilities()
        top2models = self.predict_top2()
        models = []
        for i in top2models:
            models.append(i[0])
        self.top_models = models
        return models

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
from joblib import dump

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from constants import *

class MetaTrainingClassification:

    def __init__(self, path: str):
        df = pd.read_csv(path)

    
        self.X = df.drop(columns=["task_type", "best_model"])
        self.y = df["best_model"]

      
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
        )

    def train_model(self):
       
        rf = RandomForestClassifier(random_state=42)
        rf.fit(self.X_train, self.y_train)
        score = rf.score(self.X_test, self.y_test)
        return score

    def tuning_model(self):
        
        param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2", None],
    }


        base_model = RandomForestClassifier(random_state=42)

        gridsearch = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="accuracy",  
            cv=5,
            n_jobs=-1
        )

        gridsearch.fit(self.X_train, self.y_train)

        best_model = gridsearch.best_estimator_
        best_params = gridsearch.best_params_
        best_cv_score = gridsearch.best_score_
        test_score = best_model.score(self.X_test, self.y_test)


        dump(best_model , META_CLASSIFICATION_MODEL)

        return {
            "best_model": best_model,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "test_score": test_score
        }



if __name__ == "__main__":

    path="meta_model/meta_learning/meta_classification/meta_features_classification.csv"
   

    trainer = MetaTrainingClassification(path)
    score=trainer.train_model()
    cv = trainer.tuning_model()

    print(score)
    print(cv)



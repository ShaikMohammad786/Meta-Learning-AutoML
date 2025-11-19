
from components.preprocessing import Preproccessor
from components.training import Regression_Training,Classification_Training


dataset_path = "datasets/regression/aqi.csv"
target_col='aqi_value'
preprocessor = Preproccessor(dataset_path, target_col)
X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
    preprocessor.run_preprocessing()
)
trainer = Regression_Training(X_train, y_train, X_test, y_test, X_val, y_val,dataset_path,target_col)
trainer.train_model()


# dataset_path = "datasets/classification/uber.csv"
# target_col="passenger_count"
# preprocessor = Preproccessor(dataset_path, target_col=target_col)
# X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
#     preprocessor.run_preprocessing()
# )
# trainer = Classification_Training(X_train, y_train, X_test, y_test, X_val, y_val,dataset_path=dataset_path,target_col=target_col)
# trainer.train_model()

  
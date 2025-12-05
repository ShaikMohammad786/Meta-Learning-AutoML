import pandas as pd
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.preprocessing import Preproccessor
from components.training import Regression_Training, Classification_Training
from components.meta_features_extraction import meta_features_extract_reg, meta_features_extract_class
from constants import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PENDING_DIR = os.path.join("user_section", "pending_datasets")
REGRESSION_CSV = os.path.join(PENDING_DIR, "regression.csv")
CLASSIFICATION_CSV = os.path.join(PENDING_DIR, "classification.csv")

def process_datasets(task_type, csv_path):
    if not os.path.exists(csv_path):
        logging.warning(f"Pending {task_type} datasets file not found: {csv_path}")
        return

    try:
        df_pending = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read {csv_path}: {e}")
        return

    if df_pending.empty:
        logging.info(f"No pending {task_type} datasets found.")
        return

    # Normalize column names
    df_pending.columns = [c.lower().strip() for c in df_pending.columns]
    
    # Check required columns
    required = {"dataset_path", "target_col"}
    if not required.issubset(df_pending.columns):
        logging.error(f"CSV must contain columns: {required}. Found: {df_pending.columns}")
        return

    logging.info(f"Found {len(df_pending)} pending {task_type} datasets.")

    for idx, row in df_pending.iterrows():
        dataset_path = row["dataset_path"]
        target_col = row["target_col"]

        # Handle relative paths
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)

        if not os.path.exists(dataset_path):
            logging.error(f"Dataset file not found: {dataset_path}")
            continue

        logging.info(f"Processing {task_type} dataset: {dataset_path} (Target: {target_col})")

        try:
            # 1. Preprocessing
            logging.info("Running preprocessing...")
            preprocessor = Preproccessor(dataframe=dataset_path, target_col=target_col)
            X_train, y_train, X_test, y_test, X_val, y_val, detected_task = preprocessor.run_preprocessing()

            if X_train is None:
                logging.error("Preprocessing failed (returned None). Skipping.")
                continue

            # 2. Training
            logging.info("Running training...")
            best_model = None
            
            if task_type == "regression":
                trainer = Regression_Training(
                    X_train, y_train, X_test, y_test, X_val, y_val, 
                    dataset_path=dataset_path, target_col=target_col
                )
                # Assuming train_model returns the best model or we can extract it
                # The existing train_model prints results but might not return the object directly
                # We might need to inspect the trainer class to see how to get the best model
                # For now, let's assume it saves it or we can access it.
                # Looking at user_regression_training, it seems complex. 
                # Let's check components/training.py if possible, but for now we run it.
                trainer.train_model()
                # In the original code, train_model usually sets self.best_model or similar
                # If not, we might pass a placeholder or need to update training.py
                # For meta-features, we need the model object if possible.
                if hasattr(trainer, "best_model"):
                    best_model = trainer.best_model
                
                # 3. Meta-Feature Extraction
                logging.info("Extracting meta-features...")
                # Load raw df for meta-features if needed
                raw_df = pd.read_csv(dataset_path)
                meta_features_extract_reg(X_train, y_train, best_model=best_model, raw_df=raw_df)

            elif task_type == "classification":
                trainer = Classification_Training(
                    X_train, y_train, X_test, y_test, X_val, y_val, 
                    dataset_path=dataset_path, target_col=target_col
                )
                trainer.train_model()
                if hasattr(trainer, "best_model"):
                    best_model = trainer.best_model

                logging.info("Extracting meta-features...")
                raw_df = pd.read_csv(dataset_path)
                meta_features_extract_class(X_train, y_train, best_model=best_model, raw_df=raw_df)

            logging.info(f"Successfully processed {dataset_path}")

        except Exception as e:
            logging.error(f"Error processing {dataset_path}: {e}", exc_info=True)

if __name__ == "__main__":
    print("🚀 Starting Meta-Model Training from Pending Datasets...")
    
    process_datasets("regression", REGRESSION_CSV)
    process_datasets("classification", CLASSIFICATION_CSV)
    
    print("✅ Batch processing complete.")

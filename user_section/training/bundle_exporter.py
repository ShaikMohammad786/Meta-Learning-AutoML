import shutil
import zipfile
from pathlib import Path
from textwrap import dedent

import cloudpickle
import numpy as np
import pandas as pd
import sklearn
import joblib
import sys
from sklearn.base import BaseEstimator, TransformerMixin
try:
    import imblearn
except ImportError:
    imblearn = None

try:
    import streamlit
    STREAMLIT_VERSION = streamlit.__version__
except ImportError:
    STREAMLIT_VERSION = "1.28.0"  # Default fallback version

from constants import USERS_FOLDER


def _generate_streamlit_app(task_type: str) -> str:
    """
    Generate a clean Streamlit app that loads the serialized preprocessor.
    Uses cloudpickle for preprocessor (handles class serialization) and joblib for model.
    """

    return dedent(
        f"""
        from pathlib import Path
        import cloudpickle
        import joblib
        import pandas as pd
        import streamlit as st

        # Paths to serialized artifacts
        MODEL_PATH = Path(__file__).parent / "model.pkl"
        PREPROCESSOR_PATH = Path(__file__).parent / "preprocessor.pkl"
        TASK_TYPE = "{task_type}"


        @st.cache_resource
        def load_artifacts():
            '''Load the trained model and preprocessor from disk.'''
            model = joblib.load(MODEL_PATH)
            with open(PREPROCESSOR_PATH, "rb") as f:
                preprocessor = cloudpickle.load(f)
            return model, preprocessor


        def main():
            st.set_page_config(page_title="ML Model Inference", layout="wide")
            st.title("🚀 Your Trained ML Model")
            st.write("Upload a CSV file to get predictions from your trained model.")
            
            st.sidebar.info(
                "**Setup Instructions:**\\n\\n"
                "1. Install dependencies: `pip install -r requirements.txt`\\n"
                "2. Run this app: `streamlit run app.py`\\n\\n"
                "**Note:** The preprocessing is handled by the bundled preprocessor. "
                "No external source code is needed."
            )

            # Load artifacts
            model, preprocessor = load_artifacts()
            
            # File upload
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded_file is None:
                st.info("⬆️ Upload a CSV file to begin")
                return

            # Read uploaded data
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {{raw_df.shape[0]}} rows × {{raw_df.shape[1]}} columns")
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {{e}}")
                return

            # Apply preprocessing
            try:
                processed_df = preprocessor.transform(raw_df)
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {{e}}")
                st.info("Make sure your CSV has the same columns as the training data.")
                return

            # Make predictions
            try:
                predictions = model.predict(processed_df)
            except Exception as e:
                st.error(f"❌ Prediction failed: {{e}}")
                return

            # Display results
            result_df = raw_df.copy()
            result_df["prediction"] = predictions
            
            st.subheader("📊 Predictions")
            st.dataframe(result_df.head(100), use_container_width=True)

            # Show probabilities for classification
            if TASK_TYPE == "classification" and hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(processed_df)
                    prob_df = pd.DataFrame(
                        probabilities,
                        columns=[f"prob_class_{{i}}" for i in range(probabilities.shape[1])]
                    )
                    st.subheader("📈 Class Probabilities")
                    st.dataframe(prob_df.head(100), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate probabilities: {{e}}")

            # Download button
            csv_output = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_output,
                file_name="predictions.csv",
                mime="text/csv",
            )


        if __name__ == "__main__":
            main()
        """
    ).strip()


def _generate_requirements() -> str:
    """Generate requirements.txt with pinned versions"""
    
    # Handle optional imblearn
    if imblearn:
        imblearn_line = f"imbalanced-learn=={imblearn.__version__}"
    else:
        imblearn_line = "imbalanced-learn>=0.10.0"

    return dedent(
        f"""
        numpy=={np.__version__}
        pandas=={pd.__version__}
        scikit-learn=={sklearn.__version__}
        joblib=={joblib.__version__}
        {imblearn_line}
        cloudpickle>={cloudpickle.__version__}
        pyarrow>=17.0.0
        streamlit>=1.40.0
        """
    ).strip()


def _generate_readme(task_type: str, dataset_name: str) -> str:
    """Generate README.md with setup and usage instructions"""
    return dedent(
        f"""
        # {dataset_name} - ML Model Bundle
        
        This bundle contains a trained **{task_type}** model ready for inference.
        
        ## Contents
        
        - `model.pkl` - Trained machine learning model
        - `preprocessor.pkl` - Serialized preprocessor (self-contained, no source code needed)
        - `app.py` - Streamlit web application for predictions
        - `requirements.txt` - Python dependencies with pinned versions
        
        ## Important Notes
        
        > **No Source Code Included**: The preprocessing logic is fully embedded in the `preprocessor.pkl` file.
        > You do NOT need any preprocessing source code to run this bundle. The preprocessor is completely self-contained.
        
        ## Setup
        
        1. **Install dependencies** (recommended: use a virtual environment):
           ```bash
           pip install -r requirements.txt
           ```
        
        2. **Run the Streamlit app**:
           ```bash
           streamlit run app.py
           ```
        
        3. **Upload your CSV file** in the web interface to get predictions
        
        ## Usage Notes
        
        - The CSV file must have the same schema (column names and types) as the training data
        - The target column will be automatically excluded if present
        - Missing columns will be filled with appropriate default values
        - Predictions will be added as a new column called `prediction`
        
        ## How It Works
        
        The bundled preprocessor uses its `.transform()` method to apply all preprocessing steps:
        - Imputation of missing values
        - Encoding of categorical features
        - Scaling of numerical features
        - Dimensionality reduction (if applied during training)
        
        All these transformations are stored in the `preprocessor.pkl` file and applied automatically.
        
        ## Troubleshooting
        
        **NumPy Version Error**: If you see a `BitGenerator` error, make sure you installed the exact versions from `requirements.txt`:
        ```bash
        pip install -r requirements.txt --force-reinstall
        ```
        
        **Missing Columns**: The model expects the same features used during training. Check the error message for details.
        
        **Preprocessing Errors**: Ensure your CSV has similar data types and value ranges as the training data.
        """
    ).strip()


def export_user_bundle(task_type: str, user_id: str, dataset_name: str, model_path: str, preprocessor, status_tracker=None) -> Path:
    """
    Save the trained model, preprocessing artifact and a ready-to-run Streamlit app
    into storage/<user_id>/templates/<task_type>/<dataset_name>.zip
    """

    if status_tracker:
        status_tracker.update("packaging", "Packaging: Creating inference artifact...")

    base_dir = Path(USERS_FOLDER) / user_id / "templates" / task_type
    bundle_dir = base_dir / dataset_name

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if status_tracker:
        status_tracker.update("packaging", "Packaging: Copying model file...")
    model_dst = bundle_dir / "model.pkl"
    shutil.copy2(model_path, model_dst)

def _create_inference_preprocessor(original_preprocessor):
    """
    Creates a minimal, inference-only preprocessor instance.
    This new instance contains ONLY the artifacts and the transform method.
    It has NO dependency on the original components.preprocessing module.
    """
    
    # Define the class LOCALLY so it's fully self-contained when pickled
    class InferencePreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self, artifacts):
            self.artifacts = artifacts
            # Unpack artifacts for easy access in transform
            self.dropped_bad_columns = artifacts.get("dropped_bad_columns", [])
            self.pre_encoding_columns = artifacts.get("pre_encoding_columns", [])
            self.columns_after_encoding = artifacts.get("columns_after_encoding", [])
            self.imputation_values = artifacts.get("imputation_values", {})
            self.encoders = artifacts.get("encoders", {})
            self.scale_cols = artifacts.get("scale_cols", [])
            self.scaler = artifacts.get("scaler", None)
            self.high_corr_drop_cols = artifacts.get("high_corr_drop_cols", [])
            self.final_feature_columns = artifacts.get("final_feature_columns", [])
            self.pca = artifacts.get("pca", None)
            self.target_col = artifacts.get("target_col", None)

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """Apply learned preprocessing to new data (inference)."""
            Xc = X.copy().reset_index(drop=True)

            # drop duplicates
            Xc = Xc.drop_duplicates().reset_index(drop=True)

            # drop target column if present
            if self.target_col and self.target_col in Xc.columns:
                Xc = Xc.drop(columns=[self.target_col])

            # drop bad columns (learned)
            if self.dropped_bad_columns:
                Xc = Xc.drop(columns=[c for c in self.dropped_bad_columns if c in Xc.columns], errors="ignore")

            # ensure pre-encoding columns exist
            for col in self.pre_encoding_columns:
                if col not in Xc.columns:
                    Xc[col] = np.nan
            Xc = Xc.reindex(columns=self.pre_encoding_columns)

            # impute using stored values
            for col, info in self.imputation_values.items():
                if col in Xc.columns:
                    Xc[col] = Xc[col].fillna(info["value"])

            # apply encoders learned
            if self.encoders:
                for col, meta in self.encoders.items():
                    et = meta["type"]
                    if et == "onehot":
                        expected = meta["columns"]
                        if col in Xc.columns:
                            d = pd.get_dummies(Xc[col], prefix=col, drop_first=False)
                            d = d.reindex(columns=expected, fill_value=0)
                            Xc = pd.concat([Xc.drop(columns=[col]), d], axis=1)
                        else:
                            # add zero columns for expected
                            zeros = pd.DataFrame(0, index=Xc.index, columns=expected)
                            Xc = pd.concat([Xc, zeros], axis=1)

                    elif et == "target":
                        mapping = meta["mapping"]
                        default = meta.get("default", 0.0)
                        if col not in Xc.columns:
                            Xc[col] = default
                        else:
                            Xc[col] = Xc[col].map(mapping).fillna(default)

                    elif et == "frequency":
                        mapping = meta["mapping"]
                        if col not in Xc.columns:
                            Xc[col] = 0.0
                        else:
                            Xc[col] = Xc[col].map(mapping).fillna(0.0)

            # align to columns after encoding
            if self.columns_after_encoding:
                for col in self.columns_after_encoding:
                    if col not in Xc.columns:
                        Xc[col] = 0
                Xc = Xc.reindex(columns=self.columns_after_encoding, fill_value=0)

            # scaling
            if self.scaler is not None and self.scale_cols:
                for col in self.scale_cols:
                    if col not in Xc.columns:
                        Xc[col] = 0.0
                Xc[self.scale_cols] = self.scaler.transform(Xc[self.scale_cols])

            # drop high corr
            if self.high_corr_drop_cols:
                Xc = Xc.drop(columns=[c for c in self.high_corr_drop_cols if c in Xc.columns], errors="ignore")

            # apply PCA (if learned)
            if self.pca is not None:
                Xc = pd.DataFrame(self.pca.transform(Xc), columns=[f"PC{i+1}" for i in range(self.pca.n_components_)], index=Xc.index)

            # align to final features
            for col in self.final_feature_columns:
                if col not in Xc.columns:
                    Xc[col] = 0
            Xc = Xc.reindex(columns=self.final_feature_columns, fill_value=0)

            return Xc

    # Extract artifacts
    artifacts = original_preprocessor.get_artifacts()
    # Add target_col manually as it might be missing from get_artifacts
    if hasattr(original_preprocessor, "target_col"):
        artifacts["target_col"] = original_preprocessor.target_col
        
    return InferencePreprocessor(artifacts)


def export_user_bundle(task_type: str, user_id: str, dataset_name: str, model_path: str, preprocessor, status_tracker=None) -> Path:
    """
    Save the trained model, preprocessing artifact and a ready-to-run Streamlit app
    into storage/<user_id>/templates/<task_type>/<dataset_name>.zip
    """

    if status_tracker:
        status_tracker.update("packaging", "Packaging: Creating inference artifact...")

    base_dir = Path(USERS_FOLDER) / user_id / "templates" / task_type
    bundle_dir = base_dir / dataset_name

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if status_tracker:
        status_tracker.update("packaging", "Packaging: Copying model file...")
    model_dst = bundle_dir / "model.pkl"
    shutil.copy2(model_path, model_dst)

    if status_tracker:
        status_tracker.update("packaging", "Packaging: Saving preprocessor artifact...")
    
    # Create a minimal inference-only preprocessor
    inference_preprocessor = _create_inference_preprocessor(preprocessor)
    
    # Save using cloudpickle
    artifact_path = bundle_dir / "preprocessor.pkl"
    with open(artifact_path, "wb") as f:
        cloudpickle.dump(inference_preprocessor, f)

    if status_tracker:
        status_tracker.update("packaging", "Packaging: Generating Streamlit app...")
    
    app_path = bundle_dir / "app.py"
    app_content = _generate_streamlit_app(task_type)
    app_path.write_text(app_content, encoding="utf-8")

    # Generate requirements.txt with pinned versions
    requirements_path = bundle_dir / "requirements.txt"
    requirements_path.write_text(_generate_requirements(), encoding="utf-8")

    # Generate README.md with setup instructions
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(_generate_readme(task_type, dataset_name), encoding="utf-8")

    if status_tracker:
        status_tracker.update("packaging", "Packaging: Zipping bundle...")
    
    # Create zip using shutil.make_archive which is more robust
    zip_base_name = str(base_dir / dataset_name)
    zip_path_str = shutil.make_archive(zip_base_name, 'zip', root_dir=bundle_dir)
    zip_path = Path(zip_path_str)
    
    if zip_path:
        status_tracker.update("completed", "Packaging completed. Bundle ready.", completed=True)

    return zip_path

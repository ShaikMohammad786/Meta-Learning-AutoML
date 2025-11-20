import shutil
import zipfile
from pathlib import Path
from textwrap import dedent

from joblib import dump

from constants import USERS_FOLDER


def _generate_streamlit_app(task_type: str) -> str:
    """
    Build a ready-to-run Streamlit application that:
    1. Loads the exported model + preprocessing artifact
    2. Lets end users upload CSV files
    3. Applies the stored preprocessing steps
    4. Runs predictions and offers a download option
    """

    return dedent(
        f"""
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        import streamlit as st

        TASK_TYPE = "{task_type}"
        MODEL_PATH = Path(__file__).parent / "model.pkl"
        PREPROCESSOR_PATH = Path(__file__).parent / "preprocessor.joblib"


        @st.cache_resource
        def load_artifacts():
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            return model, preprocessor


        def preprocess_with_artifact(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
            df = df.copy().drop_duplicates().reset_index(drop=True)

            target_col = artifact.get("target_col")
            if target_col and target_col in df.columns:
                df = df.drop(columns=[target_col])

            dropped_cols = artifact.get("dropped_bad_columns") or []
            df = df.drop(columns=[c for c in dropped_cols if c in df.columns], errors="ignore")

            base_cols = artifact.get("pre_encoding_columns") or list(df.columns)
            for col in base_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df.reindex(columns=base_cols)

            imputation_values = artifact.get("imputation_values") or {{}}
            for col, info in imputation_values.items():
                if col not in df.columns:
                    continue
                fill_value = info.get("value", 0)
                df[col] = df[col].fillna(fill_value)

            encoders = artifact.get("encoders") or {{}}
            for col, meta in encoders.items():
                enc_type = meta.get("type")
                if enc_type == "onehot":
                    expected_cols = meta.get("columns", [])
                    if col in df.columns:
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        dummies = dummies.reindex(columns=expected_cols, fill_value=0)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    else:
                        zeros = pd.DataFrame(0, index=df.index, columns=expected_cols)
                        df = pd.concat([df, zeros], axis=1)
                elif enc_type == "target":
                    mapping = meta.get("mapping", {{}})
                    default = float(np.mean(list(mapping.values()))) if mapping else 0.0
                    if col not in df.columns:
                        df[col] = default
                    else:
                        df[col] = df[col].map(mapping).fillna(default)
                elif enc_type == "frequency":
                    mapping = meta.get("mapping", {{}})
                    if col not in df.columns:
                        df[col] = 0
                    else:
                        df[col] = df[col].map(mapping).fillna(0)

            encoded_cols = artifact.get("columns_after_encoding")
            if encoded_cols:
                for col in encoded_cols:
                    if col not in df.columns:
                        df[col] = 0
                df = df.reindex(columns=encoded_cols, fill_value=0)

            scale_cols = artifact.get("scale_cols") or []
            scaler = artifact.get("scaler")
            if scaler is not None and scale_cols:
                for col in scale_cols:
                    if col not in df.columns:
                        df[col] = 0
                df[scale_cols] = scaler.transform(df[scale_cols])

            drop_cols = artifact.get("high_corr_drop_cols") or []
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            if artifact.get("pca") is not None:
                pca = artifact["pca"]
                transformed = pca.transform(df)
                df = pd.DataFrame(transformed, columns=[f"PC{{i+1}}" for i in range(pca.n_components_)])

            final_cols = artifact.get("final_features") or list(df.columns)
            for col in final_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df.reindex(columns=final_cols, fill_value=0)

            return df


        def main():
            st.set_page_config(page_title="Trained Model Tester", layout="wide")
            st.title("🚀 Your Customized ML Model")
            st.write("Upload a CSV file with the same schema used during training to generate predictions instantly.")

            model, artifact = load_artifacts()
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

            if uploaded_file is None:
                st.info("Waiting for a CSV upload to begin inference.")
                return

            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded file with {{raw_df.shape[0]}} rows and {{raw_df.shape[1]}} columns.")

            try:
                processed_df = preprocess_with_artifact(raw_df, artifact)
            except Exception as exc:
                st.error(f"Failed to preprocess the uploaded file: {{exc}}")
                return

            try:
                predictions = model.predict(processed_df)
            except Exception as exc:
                st.error(f"Model inference failed: {{exc}}")
                return

            result_df = raw_df.copy()
            result_df["prediction"] = predictions
            st.subheader("Prediction Preview")
            st.dataframe(result_df.head(50))

            if TASK_TYPE == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(processed_df)
                proba_df = pd.DataFrame(proba, columns=[f"class_{{idx}}" for idx in range(proba.shape[1])])
                st.subheader("Class Probabilities (first 50 rows)")
                st.dataframe(proba_df.head(50))

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name="model_predictions.csv",
                mime="text/csv",
            )


        if __name__ == "__main__":
            main()
        """
    ).strip()


def export_user_bundle(task_type: str, user_id: str, dataset_name: str, model_path: str, preprocessor) -> Path:
    """
    Save the trained model, preprocessing artifact and a ready-to-run Streamlit app
    into storage/<user_id>/templates/<task_type>/<dataset_name>.zip
    """

    artifact = preprocessor.create_inference_artifact()

    base_dir = Path(USERS_FOLDER) / user_id / "templates" / task_type
    bundle_dir = base_dir / dataset_name

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_dst = bundle_dir / "model.pkl"
    shutil.copy2(model_path, model_dst)

    artifact_path = bundle_dir / "preprocessor.joblib"
    dump(artifact, artifact_path)

    app_path = bundle_dir / "app.py"
    app_path.write_text(_generate_streamlit_app(task_type), encoding="utf-8")

    zip_path = base_dir / f"{dataset_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handler:
        handler.write(model_dst, arcname=model_dst.name)
        handler.write(artifact_path, arcname=artifact_path.name)
        handler.write(app_path, arcname=app_path.name)

    return zip_path


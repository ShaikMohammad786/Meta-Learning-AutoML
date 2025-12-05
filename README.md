# Meta-Learning AutoML

> **Meta-learning–driven AutoML system for tabular ML**  
> From raw CSV → clean features → meta-features → recommended algorithms → tuned models → exportable bundle (model, preprocessor, UI).

---

## 🧠 What Is This Project?

**Meta-Learning AutoML** is a full-stack system that:

- Learns from **many past datasets** (meta-learning) to understand _which algorithms work well on what kind of data_.
- Uses that knowledge to **recommend and train the best models** for **new** tabular datasets.
- Automatically exports:
  - A **trained model** (`model.pkl`)
  - A **preprocessing pipeline** (`preprocessor.pkl`)
  - A **ready-to-run UI** (Streamlit app) so non-ML users can run predictions.

Stack overview:

- **Backend:** Python, FastAPI, scikit-learn, pandas, numpy, LIME
- **Frontend:** React + Vite (in `Frontend/automl`)
- **Meta-Learning:** custom meta-model stored in `meta_model/`
- **Data schemas:** Pydantic models in `pydantic_models/`
- **User-facing training helpers:** `user_section/training/`
- **Utility functions:** `utils/`

---

## ✨ Key Features

### 1. End-to-end Data & Model Pipeline

- Dataset upload (UI or API)
- Automated:
  - Missing-value handling
  - Categorical encoding
  - Numeric scaling
  - Outlier handling
  - Train/validation/test splits
  - Optional PCA / correlation-based feature pruning
- Trains & evaluates ML models with proper metrics and reproducible seeds.

### 2. Meta-Learning for Algorithm Recommendation

Instead of blindly trying everything for every new dataset, the system:

1. Extracts **meta-features** from the dataset (size, feature types, imbalance, stats).
2. Uses a **meta-model** (stored in `meta_model/`) that was trained on many past datasets.
3. Predicts the **top algorithms** likely to perform best.
4. Trains **only those** (e.g., top-2) with hyperparameter tuning.

This makes the AutoML process:

- **Faster** (smaller search space)
- **Smarter** (learns from history)
- **More scalable** across many datasets.

### 3. Exportable Model Bundle

For each ML task, the system can export:

- `model.pkl` – the final trained estimator.
- `preprocessor.pkl` – all preprocessing steps (encoders, scalers, etc.).
- A **Streamlit-based inference UI** that loads the above and serves predictions.

This makes production deployment simple:

```bash
streamlit run app.py  # after placing model.pkl + preprocessor.pkl
```

### 4. Frontend UI (React + Vite)

Located in `Frontend/automl/`, the frontend:

- Lets users upload datasets.
- Shows preprocessing & training status.
- Visualises:
  - Dataset summary
  - Chosen algorithms
  - Performance metrics
- Exposes options to trigger meta-learning–guided training and download the exported bundle.

### 5. Interpretability with LIME

- Uses **LIME** (Local Interpretable Model-agnostic Explanations) to:
  - Explain individual predictions.
  - Help users understand why the model decided something.
- Helper logic lives in `user_section/training/model_explanations.py`.

---

## 🧩 High-Level Architecture

```text
                           ┌────────────────────────────┐
                           │        Frontend UI         │
                           │      (React + Vite)        │
                           └────────────┬───────────────┘
                                        │ HTTP (JSON)
                                        ▼
                           ┌────────────────────────────┐
                           │        FastAPI API         │
                           │        main.py             │
                           └────────────┬───────────────┘
                                        │
         ┌───────────────────────────────┴───────────────────────────────────────┐
         │                                                                   │
         ▼                                                                   ▼
┌───────────────────────┐                                        ┌───────────────────────┐
│  Data & Preprocessing │                                        │    Meta-Learning      │
│      (utils/,         │                                        │    (meta_model/)      │
│       datasets/)       │                                        │ - Meta-features       │
│ - Cleaning            │                                        │ - Meta-model          │
│ - Encoding            │                                        │ - Algorithm ranking   │
│ - Splits              │                                        └───────────────────────┘
└───────────┬───────────┘                                                   │
            │                                                               │
            ▼                                                               ▼
   ┌───────────────────────┐                                   ┌─────────────────────────┐
   │  Candidate Trainers   │                                   │  Export & Explain      │
   │  (RandomForest, SVM,  │                                   │  (user_section/,       │
   │   XGBoost, etc.)      │                                   │   Streamlit app)       │
   └───────────┬───────────┘                                   └─────────────────────────┘
               │
               ▼
       Best model & config
       + metrics + bundle
```

---

## 📁 Project Structure

> Note: some filenames are summarised for readability – check your repo for exact names.

```text
Meta-Learning-AutoML/
├── Frontend/
│   └── automl/
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       ├── public/
│       │   └── vite.svg
│       └── src/
│           ├── main.jsx
│           ├── App.jsx
│           ├── App.css
│           ├── index.css
│           ├── assets/
│           │   └── react.svg
│           ├── components/
│           │   ├── Layout.jsx
│           │   ├── Navigation.jsx
│           │   └── Footer.jsx
│           ├── context/
│           │   └── SessionContext.jsx
│           ├── lib/
│           │   └── api.js
│           └── pages/
│               ├── Auth.jsx
│               ├── Home.jsx
│               ├── Models.jsx
│               └── Workspace.jsx
│
├── components/                 # Backend-side helper modules (Python)
│   └── ...                     # (e.g., orchestration, pipeline composition)
│
├── constants/                  # Shared constants, enums, choices
│   └── ...                     # (e.g., algorithm names, metric keys)
│
├── datasets/
│   └── classification/         # Sample datasets for experimentation / meta-training
│       └── ...                 # (CSV files)
│
├── meta_model/                 # Meta-learning model & meta-feature utilities
│   └── ...                     # e.g., meta_features.py, meta_learner.py, loaders
│
├── pydantic_models/            # Request/response schemas for FastAPI
│   └── ...                     # e.g., DatasetConfig, TrainRequest, TrainResponse
│
├── user_section/
│   └── training/
│       ├── bundle_exporter.py  # Exports model + preprocessor + app bundle
│       ├── model_explanations.py  # LIME and other explanation helpers
│       └── status_tracker.py   # Job progress, logs, status updates
│
├── utils/
│   ├── data_preprocessing.py   # Imputation, encoding, scaling, splits
│   ├── model_training.py       # Training logic, hyperparameter search
│   ├── evaluation.py           # Metrics calculation & reporting
│   ├── meta_feature_utils.py   # (If present) meta-feature extraction helpers
│   └── ...
│
├── config.py                   # Global configuration (paths, constants, settings)
├── main.py                     # FastAPI application entrypoint
├── requirements.txt            # Python dependencies
├── __init__.py
├── .gitignore
└── README.md                   # This file
```

If some filenames differ, feel free to update them in this tree.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ShaikMohammad786/Meta-Learning-AutoML.git
cd Meta-Learning-AutoML
```

### 2. Backend Setup (FastAPI + ML)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### Run the FastAPI server

```bash
uvicorn main:app --reload
```

By default this usually runs on `http://127.0.0.1:8000`.

- Interactive API docs: `http://127.0.0.1:8000/docs`
- Alternative docs (ReDoc): `http://127.0.0.1:8000/redoc`

### 3. Frontend Setup (React + Vite)

```bash
cd Frontend/automl
npm install
npm run dev
```

By default Vite runs on `http://127.0.0.1:5173` (or whichever port is shown in the console).

Make sure the frontend is configured to call the backend base URL (`http://127.0.0.1:8000`) in `src/lib/api.js`.

---

## 📌 Typical Usage Flow

1. **Open the UI** (Vite dev server or production build).
2. **Upload a dataset** (CSV) through the frontend.
3. Backend:
   - Validates the file (extension, size, basic checks).
   - Loads into pandas.
   - Performs preprocessing using utilities in `utils/`.
4. **Meta-feature extraction** runs on this dataset.
5. **Meta-model** (in `meta_model/`) predicts a ranked list of algorithms.
6. **AutoML trainer**:
   - Selects top-k algorithms (e.g., 2).
   - Executes hyperparameter search for each.
   - Chooses the best model based on chosen metric (accuracy / F1 / etc.).
7. The system:
   - Saves `model.pkl` and `preprocessor.pkl`.
   - Updates training status via `status_tracker.py`.
   - Optionally creates a Streamlit inference app bundle using `bundle_exporter.py`.
8. Frontend shows:
   - Final performance metrics.
   - Algorithm chosen.
   - Links/buttons to download the bundle.

---

## 🔍 Backend Design Details

### `main.py` (FastAPI Entrypoint)

Typical responsibilities include:

- Mounting routes for:
  - `/upload_dataset` – upload and register a dataset.
  - `/train` – trigger meta-learning–guided training.
  - `/status/{job_id}` – fetch job status from `status_tracker.py`.
  - `/download/{artifact}` – download model bundle or other artifacts.
- Dependency injection (e.g., authentication, user session).
- Wiring utilities from:
  - `utils/`
  - `meta_model/`
  - `user_section/training/`
  - `pydantic_models/`

### `pydantic_models/`

Defines Pydantic classes for:

- Upload responses
- Training requests (e.g., target column, task type: classification/regression)
- Training results (metrics, chosen algorithm, artifact paths)
- Status responses

These provide strict types and automatic validation for API inputs/outputs.

### `utils/` Package

Common responsibilities:

- **Data preprocessing**:
  - Detect numeric vs categorical columns.
  - Impute missing values:
    - Numeric: mean/median.
    - Categorical: most frequent / constant.
  - Encode categoricals (One-Hot, Ordinal, etc.).
  - Scale numeric features (StandardScaler/MinMaxScaler).
- **Splitting**:
  - Train/validation/test splitting with reproducible random seeds.
  - Stratified splitting for classification.
- **Metrics**:
  - Accuracy, F1, precision, recall, ROC-AUC (classification).
  - MSE, MAE, R² (regression, if supported).
- **Meta-feature helpers**:
  - Compute dataset-level descriptors used by the meta-model.

---

## 🧠 Meta-Learning Pipeline

Though implementation details live in `meta_model/` and related utilities, conceptually:

1. **Historical dataset collection**  
   Multiple classification datasets (e.g., from `datasets/classification/`).

2. **Base learner evaluation**  
   For each dataset:
   - Train multiple algorithms (e.g., LogisticRegression, RandomForest, XGBoost, SVC, KNN).
   - Record performance metrics.

3. **Meta-feature extraction**  
   For each dataset, extract meta-features such as:
   - Number of samples / features
   - Proportion of numeric vs categorical columns
   - Class imbalance ratios
   - Skewness / kurtosis of numeric features
   - Entropy of target
   - Correlation statistics

4. **Meta-model training**  
   - Input: meta-features.
   - Target: best-performing algorithm (or ranking).
   - Train a classifier/regressor that maps meta-features → recommended algorithms.

5. **At inference (user dataset)**  
   - Compute meta-features.
   - Run meta-model.
   - Get top algorithms.
   - Train only those algorithms with tuning.

This pipeline allows the system to **reuse previous experience** instead of starting from scratch for each new dataset.

---

## 📊 Model Training & Evaluation

The training pipeline (via `utils/` + `user_section/training/`) typically:

1. Builds an sklearn `Pipeline`:
   - Preprocessor (`ColumnTransformer` or custom).
   - Estimator (RandomForest, XGBoost, etc.).
2. Uses techniques like:
   - K-fold or stratified k-fold cross-validation.
   - Grid search / random search over hyperparameters.
3. For each candidate algorithm:
   - Evaluate on validation folds.
   - Track scores and training time.
4. Compare algorithms recommended by the meta-model.
5. Persist:
   - Final chosen model.
   - Preprocessor.
   - Performance summary (JSON/CSV).

---

## 🧾 Explainability (LIME)

`user_section/training/model_explanations.py`:

- Wraps LIME to generate local explanations for single predictions.
- Flow:
  1. User (or internal API) selects an instance.
  2. System passes instance + model to LIME.
  3. LIME returns feature importance for that particular prediction.
- Can be integrated into:
  - Streamlit inference UI.
  - Frontend (via API, rendering plots or tables).

This is important for:

- Decision transparency.
- Debugging misclassified examples.
- Communicating with non-technical stakeholders.

---

## 📦 Bundle Exporter

`user_section/training/bundle_exporter.py`:

- Packages the final artifacts into a deployable bundle:
  - `model.pkl`
  - `preprocessor.pkl`
  - Optional `config.json`
  - Streamlit `app.py` (or similar) that:
    - Loads both pickles.
    - Builds input forms from feature schema.
    - Outputs predictions and explanation plots.
- The exported bundle can be:
  - Run locally with `streamlit run app.py`.
  - Deployed to platforms like Streamlit Cloud, Hugging Face Spaces, or any server.

---

## 📈 Status Tracking

`user_section/training/status_tracker.py`:

- Tracks long-running training jobs:
  - QUEUED → PREPROCESSING → META_FEATURES → TRAINING → EXPORTING → DONE / FAILED
- Stores progress (could be in memory, DB, or file-based).
- Enables the frontend to:
  - Poll for job status.
  - Show progress bars / logs to the user.

---

## 🌐 Frontend Details (Frontend/automl)

The Vite + React app is structured as:

- `src/pages/`
  - `Auth.jsx` – handles login/auth (if implemented) and basic access.
  - `Home.jsx` – landing page (project overview, quick actions).
  - `Models.jsx` – list of previous models, metrics, download links.
  - `Workspace.jsx` – main working page for uploading datasets and running AutoML.
- `src/components/`
  - `Layout.jsx` – common shell (header, footer, sidebar).
  - `Navigation.jsx` – main navbar/side menu.
  - `Footer.jsx` – footer with credits, links, etc.
- `src/context/SessionContext.jsx`
  - Provides user/session/global state across the app.
- `src/lib/api.js`
  - Central API client.
  - Functions like `uploadDataset`, `startTraining`, `getStatus`, `getModels`, etc.

Styling:

- Controlled via `App.css` and `index.css`.
- React/Vite default assets (e.g., `react.svg`) for logos or replaced by project branding.

---

## 🧪 Testing (Suggested)

If you want to extend the project with tests:

- **Backend tests**:
  - Use `pytest` + `httpx` or `requests` to test FastAPI endpoints.
  - Test:
    - Dataset upload
    - Training pipeline (with small toy datasets)
    - Status endpoints
- **Frontend tests**:
  - Use `vitest` / `jest` + `@testing-library/react` to test UI components and API hooks.

---

## 🧱 Configuration

`config.py` may contain:

- Paths (e.g., where to store:
  - Uploaded datasets
  - Intermediate artifacts
  - Model bundles
- Default hyperparameters or search spaces.
- Feature flags (e.g., whether to enable LIME, meta-learning, etc.).
- Logging configuration.

You can customize these to match your environment (local vs server).


### Frontend

- Build:

  ```bash
  cd Frontend/automl
  npm run build
  ```

- Serve `dist/` using:
  - nginx
  - Vercel / Netlify / any static hosting
- Make sure the frontend `api.js` points to the deployed backend URL.

### All-in-one

- You can create a `docker-compose.yml` that:
  - Spins up backend container.
  - Spins up frontend container (or a simple nginx serving the built files).
  - Optionally attaches volumes for storing datasets/model bundles.



## 📜 License

Choose a license that matches your needs. A common option is MIT:

```text
MIT License

Copyright (c) 2025 Shaik Mohammad
```

If you add a `LICENSE` file, update this section accordingly.

---

## 👤 Author & Contact

**Author:** Shaik Mohammad  
**GitHub:** [@ShaikMohammad786](https://github.com/ShaikMohammad786)  
**Project Repo:** [Meta-Learning AutoML](https://github.com/ShaikMohammad786/Meta-Learning-AutoML)

If you use this project or build on top of it, a star ⭐ on the repo is always appreciated!

---

_This README is generated to be extremely detailed so that any new contributor, recruiter, or teammate can understand the full story of the project from top to bottom._

Meta-Learning-AutoML

An end-to-end meta-learning-driven AutoML system for tabular datasets: ingestion → preprocessing → meta-feature extraction → algorithm recommendation → model training → ready-to-deploy package.

Built with Python (scikit‐learn, pandas, numpy, Lime), React + FastAPI frontend/back-end, and containerised for production.

Table of Contents

Project Motivation

Key Features & Highlights

Architecture & Workflow

Component Breakdown

Getting Started

Prerequisites

Installation

Running the system

Usage Example

Meta-Learning Pipeline Details

Frontend / API Overview

Deployment & Dockerisation

Project Structure

Contributing

License

Acknowledgements

Contact

<a name="motivation"></a>1. Project Motivation

Traditional AutoML tools often treat each dataset independently, performing full exploration for each new task.

Meta-learning allows leveraging prior experience over many datasets (1000+ in this project) to predict which algorithms or configurations will perform well, dramatically reducing search time.

Business and research demands: deliver a scalable, production-ready AutoML solution that can be deployed by non-ML experts (via UI) yet offers power and transparency for ML engineers.

This project fulfils that by designing a full stack system: ingestion, preprocessing, meta-learning recommendation, model training, and packaged output (model + preprocessor + UI).

<a name="features"></a>2. Key Features & Highlights

Scalable pipeline for 1000+ tabular datasets: ingestion, cleaning, encoding, outlier detection/removal, PCA/correlation-based feature pruning.

Meta-feature extraction: from each dataset generate meta-features describing size, distribution, skewness, feature types, target complexity, etc.

Algorithm recommendation: meta-model predicts best performing algorithm(s) for a new dataset, narrowing search to top-2.

Automated training of top algorithms with hyperparameter tuning (10+ candidate algorithms).

Packaged deliverable: produces model.pkl, preprocessor.pkl, and a ready-to-run Streamlit app for deployment.

Frontend + API: interactive web UI (React) with backend API (FastAPI) for dataset upload, pipeline execution, results visualization.

Transparency & interpretability: uses LIME (Local Interpretable Model-agnostic Explanations) to explain model predictions, and meta-feature insights to show why certain algorithms were recommended.

Production readiness: containerised via Docker for easy deployment, reproducibility, and scalability.

<a name="architecture"></a>3. Architecture & Workflow
USER → upload dataset (UI)  
    ↓  
Backend API → Dataset ingestion & preprocessing pipeline  
    ↓  
Meta-feature extractor → Generate meta-features  
    ↓  
Meta-model (trained offline) → Recommend top-2 algorithms  
    ↓  
AutoML engine → Train & tune recommended algorithms  
    ↓  
Select best model → Save model.pkl + preprocessor.pkl  
    ↓  
Deployable Streamlit app package  
    ↓  
User can now serve predictions or download the package for deployment  


Additional monitoring dashboards provide: dataset metrics, meta-feature summary, algorithm performance, LIME explanations.

<a name="components"></a>4. Component Breakdown
a) Data Pipeline

Handles ingestion of raw tabular data (CSV, Excel).

Performs cleaning: missing value imputation, categorical encoding, numeric scaling, outlier detection.

Applies dimensionality reduction / feature-pruning via PCA or correlation thresholds.

Produces a model-ready dataset and persists the preprocessor.pkl.

b) Meta-Learning Engine

Processes historical datasets (>1000) to extract meta-features like: number of instances, number of features, feature‐to‐instance ratio, average skewness, entropy of target, feature type counts (numeric, categorical), etc.

Trains a meta-model (e.g., RandomForest / XGBoost) to map meta-features → best algorithm(s).

For a new dataset, predicts top-2 algorithms from the candidate pool.

c) AutoML Module

Candidate algorithms: e.g., LogisticRegression, RandomForest, XGBClassifier, SVC, KNeighbors, etc.

Hyperparameter search (GridSearchCV / RandomSearchCV).

Selects best algorithm, trains full model.

Saves: model.pkl, preprocessor.pkl, along with metadata (performance scores, training report).

d) Interpretability Layer

Uses LIME to generate explanations of model predictions.

Meta-feature reports show why certain algorithms were recommended (feature importance from meta-model).

e) Frontend & API

Backend: FastAPI for dataset upload endpoint, status polling, result download.

Frontend: React SPA with upload UI, progress indicators, visualisation of dataset stats, algorithm recommendations, training metrics, final download.

The packaged Streamlit app can be launched by end-user for inference.

f) Deployment

Dockerfile provided.

Compose for multi-service (API + frontend).

CI/CD pipeline optional (you can integrate GitHub Actions).

Logging & monitoring hooks are included (e.g., training logs, evaluation metrics).

<a name="getting-started"></a>5. Getting Started
Prerequisites

Python 3.8+

Node.js 16+ (for frontend)

Docker (if containerising)

Git

Installation
# Clone the repo
git clone https://github.com/ShaikMohammad786/Meta-Learning-AutoML.git
cd Meta-Learning-AutoML

# Backend setup
cd backend  # or where main.py lives
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install

Running the system locally
# In one terminal: backend
cd backend
uvicorn main:app --reload

# In another terminal: frontend
cd frontend
npm start


Navigate to http://localhost:3000 (or configured port) to use UI.

Docker mode
docker build -t meta-learning‐automl .
docker run -p 8000:8000 meta-learning-automl

<a name="usage"></a>6. Usage Example

Upload a new tabular dataset via UI (.csv) – e.g., heart_disease.csv.

The system displays: uploaded dataset stats, missing values summary, basic plots.

Meta-feature extraction runs automatically.

Meta-model recommends: RandomForestClassifier and XGBClassifier.

AutoML module tunes both, picks the best (e.g., XGB with 85% accuracy).

You are prompted to download: model.pkl, preprocessor.pkl, inference_app_streamlit.zip.

Run the Streamlit app:

streamlit run inference_app.py --model model.pkl --preprocessor preprocessor.pkl


Use the UI to input feature values and get predictions + LIME explanation.

<a name="meta-learning-pipeline"></a>7. Meta-Learning Pipeline Details

Historical dataset pool: 1000+ datasets from open sources (UCI, Kaggle) covering classification tasks.

Meta-feature list includes:

Number of samples, number of features, ratio of categorical to numeric features

Class imbalance ratio

Mean/variance/skewness/kurtosis of numerical features

Number of unique values in categorical features

Entropy of target distribution

Correlation statistics between features & target

Meta-model training:

Algorithm pool: LogisticRegression, RandomForest, XGBoost, SVM, KNN, etc.

Performance label: best algorithm by accuracy (or other metric) on dataset after standard pipeline.

Meta-model: RandomForestClassifier trained on (meta-features → best algorithm).

At runtime, given new dataset meta-features → recommend top-2.

Advantages: significantly reduces search space and training time compared to blind AutoML.

<a name="frontend-api"></a>8. Frontend & API Overview

Backend (FastAPI)

POST /upload – upload dataset

GET /status/{job_id} – poll job status

GET /results/{job_id} – retrieve results (recommended algorithms, metrics, download links)

Frontend (React)

Upload form & drag-&-drop

Real-time progress bar for preprocessing → meta-features → training

Visualisations: dataset summary, training metrics, meta-feature importance, best model metrics

Download button for packaged deliverable

Streamlit App Package

Simple UI for inference: load model + preprocessor → provide input → see prediction + LIME explainability plot

<a name="deployment"></a>9. Deployment & Dockerisation

Dockerfile in root.

Use docker-compose.yml if you have separate services (frontend + backend + database).

Logging: logs sent to logs/ folder (or optionally to cloud).

Monitoring: can plug into Prometheus/Grafana if extended.

CI/CD suggestion: Use GitHub Actions to build, test, lint (Python & JavaScript) and push container to Docker Hub or GitHub Packages.

<a name="project-structure"></a>10. Project Structure
Meta-Learning-AutoML/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── pydantic_models/
│   ├── utils/
│   ├── meta_model/
│   ├── datasets/
│   ├── requirements.txt
│   └── …
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── lib/
│   └── …
├── user_section/         # contains user-written training modules, custom scripts
│   ├── training/
│   │   ├── bundle_exporter.py
│   │   ├── model_explanations.py
│   │   └── status_tracker.py
├── .gitignore
├── README.md
└── Dockerfile


Note: adjust names as per actual repo.

<a name="contributing"></a>11. Contributing

Contributions are very welcome! If you’d like to contribute:

Fork the repository.

Create a new branch (git checkout -b feature/your-feature).

Make your changes and add tests/documentation.

Commit with meaningful message and push to your fork.

Open a Pull Request describing the feature/bug.

Ensure your code passes linting and tests.

<a name="license"></a>12. License

This project is licensed under the MIT License – see the LICENSE
 file for details.
(If you prefer a different license, update accordingly.)

<a name="acknowledgements"></a>13. Acknowledgements

Datasets from UCI, Kaggle, OpenML.

Meta-learning inspiration from “Metalearning for AutoML” research papers.

Tools: scikit-learn, pandas, numpy, LIME, FastAPI, React.

Thanks to the open-source community for libraries and utilities.

<a name="contact"></a>14. Contact

Created and maintained by Shaik Mohammad.
GitHub: @ShaikMohammad786

Email: your-email@example.com
 (replace with your actual email if you choose)

Feel free to raise issues, open discussions, or send suggestions via GitHub.

Thank you for exploring this project. I hope you find Meta-Learning-AutoML useful, inspiring, and ready-to-deploy for real-world data-science workflows. 🧠🚀

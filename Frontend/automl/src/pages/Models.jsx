import { useEffect, useState } from "react";
import {
  fetchModels,
  downloadModel,
  deleteModel,
  fetchBundles,
  downloadBundle,
  deleteBundle,
} from "../lib/api";
import { useSession } from "../context/SessionContext";

const Models = () => {
  const { token, profile } = useSession();
  const [models, setModels] = useState({ classification: [], regression: [] });
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [bundles, setBundles] = useState({
    classification: [],
    regression: [],
  });

  useEffect(() => {
    if (!token) return;
    const hydrate = async () => {
      setLoading(true);
      try {
        const [modelResp, bundleResp] = await Promise.all([
          fetchModels(token),
          fetchBundles(token),
        ]);
        setModels(modelResp);
        setBundles(bundleResp);
      } catch (error) {
        setStatus({ type: "error", message: error.message });
      } finally {
        setLoading(false);
      }
    };
    hydrate();
  }, [token]);

  const handleDelete = async (modelPath) => {
    if (!token) return;
    try {
      await deleteModel(token, modelPath);
      setStatus({ type: "success", message: "Model deleted." });
      const [refreshedModels, refreshedBundles] = await Promise.all([
        fetchModels(token),
        fetchBundles(token),
      ]);
      setModels(refreshedModels);
      setBundles(refreshedBundles);
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  };

  const handleDeleteBundle = async (bundlePath) => {
    if (!token) return;
    try {
      await deleteBundle(token, bundlePath);
      setStatus({ type: "success", message: "Bundle deleted." });
      const [refreshedModels, refreshedBundles] = await Promise.all([
        fetchModels(token),
        fetchBundles(token),
      ]);
      setModels(refreshedModels);
      setBundles(refreshedBundles);
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  };

  // Auto-dismiss success alerts
  useEffect(() => {
    if (status?.type === "success") {
      const timer = setTimeout(() => setStatus(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [status]);

  const formatMetric = (metric) => {
    if (metric === null || metric === undefined) return "N/A";
    return Number(metric).toFixed(3);
  };

  if (!token) {
    return (
      <section className="page models">
        <header>
          <p className="eyebrow">Model Registry</p>
          <h1>Your Trained Models</h1>
          <p className="lead">
            Sign in to view and download your production-ready machine learning
            models.
          </p>
        </header>
        <div className="card muted">
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>
            <i className="fas fa-lock"></i>
          </div>
          <h3 style={{ marginBottom: "1rem", color: "var(--gray-300)" }}>
            Authentication Required
          </h3>
          <p style={{ marginBottom: "1.5rem" }}>
            Please sign in to access your trained models and deployment
            packages.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="page models">
      <header>
        <p className="eyebrow">Model Registry</p>
        <h1>
          <i className="fas fa-database"></i> Your Trained Models
        </h1>
        <p className="lead">
          Welcome, {profile?.username}! Download your trained models and
          deployment packages. Each model is optimized and ready for production
          use.
        </p>
      </header>

      {status && (
        <div className={`alert ${status.type}`}>
          <i
            className={`fas fa-${
              status.type === "success"
                ? "check-circle"
                : "exclamation-triangle"
            }`}
          ></i>
          {status.message}
        </div>
      )}
      {loading && (
        <span className="badge">
          <i className="fas fa-sync fa-spin"></i> Loading models...
        </span>
      )}

      <div className="models-section">
        <div className="card">
          <h3 style={{ marginBottom: "1rem" }}>
            <i className="fas fa-bullseye"></i> Classification Models
          </h3>
          {models.classification?.length ? (
            <ul className="model-list">
              {models.classification.map((model) => (
                <li key={model.name} className="model-row">
                  <div className="model-meta">
                    <strong>{model.name}</strong>
                    <small>{model.model_label || "Auto-selected model"}</small>
                    {model.model_reason && <p>{model.model_reason}</p>}
                    {model.human_metric ? (
                      <span className="badge success">
                        <i className="fas fa-check"></i> {model.human_metric}
                      </span>
                    ) : model.metric_name ? (
                      <span className="badge">
                        <i className="fas fa-chart-line"></i>{" "}
                        {`${model.metric_name}: ${formatMetric(
                          model.metric_value
                        )}`}
                      </span>
                    ) : (
                      <span className="badge muted">
                        <i className="fas fa-hourglass-half"></i> Evaluating...
                      </span>
                    )}
                    {model.explanations?.length ? (
                      <div className="explanations">
                        {model.explanations.slice(0, 4).map((item) => (
                          <span
                            key={`${model.name}-${item.feature}`}
                            className="pill"
                          >
                            {item.feature}: {Number(item.weight).toFixed(2)}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p className="muted" style={{ fontSize: "0.875rem" }}>
                        Feature importance analysis unavailable
                      </p>
                    )}
                  </div>
                  <div className="model-actions">
                    <button
                      type="button"
                      className="icon-btn"
                      aria-label={`Download ${model.name}`}
                      onClick={() =>
                        downloadModel(model.download_url, model.name)
                      }
                    >
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M12 4v10m0 0 4-4m-4 4-4-4m-4 9h16"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                    <button
                      type="button"
                      className="icon-btn danger"
                      aria-label={`Delete ${model.name}`}
                      onClick={() => handleDelete(model.path)}
                    >
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M18 6l-1 14H7L6 6m3 0V4h6v2m-9 0h12"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p
              className="muted"
              style={{ padding: "2rem", textAlign: "center" }}
            >
              <i className="fas fa-info-circle"></i> No classification models
              yet. Train your first model in the Workspace!
            </p>
          )}
        </div>

        <div className="card">
          <h3 style={{ marginBottom: "1rem" }}>
            <i className="fas fa-chart-line"></i> Regression Models
          </h3>
          {models.regression?.length ? (
            <ul className="model-list">
              {models.regression.map((model) => (
                <li key={model.name} className="model-row">
                  <div className="model-meta">
                    <strong>{model.name}</strong>
                    <small>{model.model_label || "Auto-selected model"}</small>
                    {model.model_reason && <p>{model.model_reason}</p>}
                    {model.human_metric ? (
                      <span className="badge success">
                        <i className="fas fa-check"></i> {model.human_metric}
                      </span>
                    ) : model.metric_name ? (
                      <span className="badge">
                        <i className="fas fa-chart-line"></i>{" "}
                        {`${model.metric_name}: ${formatMetric(
                          model.metric_value
                        )}`}
                      </span>
                    ) : (
                      <span className="badge muted">
                        <i className="fas fa-hourglass-half"></i> Evaluating...
                      </span>
                    )}
                    {model.explanations?.length ? (
                      <div className="explanations">
                        {model.explanations.slice(0, 4).map((item) => (
                          <span
                            key={`${model.name}-${item.feature}`}
                            className="pill"
                          >
                            {item.feature}: {Number(item.weight).toFixed(2)}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p className="muted" style={{ fontSize: "0.875rem" }}>
                        Feature importance analysis unavailable
                      </p>
                    )}
                  </div>
                  <div className="model-actions">
                    <button
                      type="button"
                      className="icon-btn"
                      aria-label={`Download ${model.name}`}
                      onClick={() =>
                        downloadModel(model.download_url, model.name)
                      }
                    >
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                      >
                        <path
                          d="M12 4v10m0 0 4-4m-4 4-4-4m-4 9h16"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                    <button
                      type="button"
                      className="icon-btn danger"
                      aria-label={`Delete ${model.name}`}
                      onClick={() => handleDelete(model.path)}
                    >
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                      >
                        <path
                          d="M18 6l-1 14H7L6 6m3 0V4h6v2m-9 0h12"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p
              className="muted"
              style={{ padding: "2rem", textAlign: "center" }}
            >
              <i className="fas fa-info-circle"></i> No regression models yet.
              Train your first model in the Workspace!
            </p>
          )}
        </div>
      </div>

      <section className="bundles">
        <header>
          <h2>
            <i className="fas fa-box"></i> Ready-to-Deploy Packages
          </h2>
          <p className="lead">
            Complete deployment packages with trained models, preprocessing
            pipelines, and interactive apps. Download and run instantly!
          </p>
        </header>
        <div className="bundles-grid">
          <div className="card">
            <h3 style={{ marginBottom: "1rem" }}>
              <i className="fas fa-bullseye"></i> Classification Packages
            </h3>
            {bundles.classification?.length ? (
              <ul className="model-list">
                {bundles.classification.map((bundle) => (
                  <li key={bundle.path} className="model-row">
                    <div className="model-meta">
                      <strong>{bundle.name}</strong>
                      <small>
                        Updated{" "}
                        {new Date(bundle.modified_ts * 1000).toLocaleString()} ·{" "}
                        {(bundle.size_bytes / (1024 * 1024)).toFixed(2)} MB
                      </small>
                    </div>
                    <div className="model-actions">
                      <button
                        type="button"
                        className="icon-btn"
                        aria-label={`Download bundle ${bundle.name}`}
                        onClick={() =>
                          downloadBundle(
                            token,
                            bundle.download_url,
                            `${bundle.name}.zip`
                          )
                        }
                      >
                        <svg
                          width="18"
                          height="18"
                          viewBox="0 0 24 24"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M12 4v10m0 0 4-4m-4 4-4-4m-4 9h16"
                            stroke="currentColor"
                            strokeWidth="1.6"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                      <button
                        type="button"
                        className="icon-btn danger"
                        aria-label={`Delete bundle ${bundle.name}`}
                        onClick={() => handleDeleteBundle(bundle.path)}
                      >
                        <svg
                          width="18"
                          height="18"
                          viewBox="0 0 24 24"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M18 6l-1 14H7L6 6m3 0V4h6v2m-9 0h12"
                            stroke="currentColor"
                            strokeWidth="1.6"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p
                className="muted"
                style={{ padding: "2rem", textAlign: "center" }}
              >
                <i className="fas fa-info-circle"></i> No classification
                packages yet. Complete packages will be available after
                training!
              </p>
            )}
          </div>

          <div className="card">
            <h3 style={{ marginBottom: "1rem" }}>
              <i className="fas fa-chart-line"></i> Regression Packages
            </h3>
            {bundles.regression?.length ? (
              <ul className="model-list">
                {bundles.regression.map((bundle) => (
                  <li key={bundle.path} className="model-row">
                    <div className="model-meta">
                      <strong>{bundle.name}</strong>
                      <small>
                        Updated{" "}
                        {new Date(bundle.modified_ts * 1000).toLocaleString()} ·{" "}
                        {(bundle.size_bytes / (1024 * 1024)).toFixed(2)} MB
                      </small>
                    </div>
                    <div className="model-actions">
                      <button
                        type="button"
                        className="icon-btn"
                        aria-label={`Download bundle ${bundle.name}`}
                        onClick={() =>
                          downloadBundle(
                            token,
                            bundle.download_url,
                            `${bundle.name}.zip`
                          )
                        }
                      >
                        <svg
                          width="18"
                          height="18"
                          viewBox="0 0 24 24"
                          fill="none"
                        >
                          <path
                            d="M12 4v10m0 0 4-4m-4 4-4-4m-4 9h16"
                            stroke="currentColor"
                            strokeWidth="1.6"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                      <button
                        type="button"
                        className="icon-btn danger"
                        aria-label={`Delete bundle ${bundle.name}`}
                        onClick={() => handleDeleteBundle(bundle.path)}
                      >
                        <svg
                          width="18"
                          height="18"
                          viewBox="0 0 24 24"
                          fill="none"
                        >
                          <path
                            d="M18 6l-1 14H7L6 6m3 0V4h6v2m-9 0h12"
                            stroke="currentColor"
                            strokeWidth="1.6"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p
                className="muted"
                style={{ padding: "2rem", textAlign: "center" }}
              >
                <i className="fas fa-info-circle"></i> No regression packages
                yet. Complete packages will be available after training!
              </p>
            )}
          </div>
        </div>
      </section>
    </section>
  );
};

export default Models;

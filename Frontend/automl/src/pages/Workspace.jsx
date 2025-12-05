import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import {
  previewColumns,
  uploadDataset,
  fetchDatasets,
  downloadDataset,
  deleteDataset,
  fetchTrainingStatus,
  fetchActiveTrainingRuns,
} from "../lib/api";
import { useSession } from "../context/SessionContext";

const initialPayload = {
  task_type: "classification",
  target_col: "",
  tuning: "false",
};

const Workspace = () => {
  const { token, profile } = useSession();
  const [datasetFile, setDatasetFile] = useState(null);
  const [form, setForm] = useState(initialPayload);
  const [columns, setColumns] = useState([]);
  const [status, setStatus] = useState(null);
  const [catalogue, setCatalogue] = useState({
    classification: [],
    regression: [],
  });
  const [loadingCatalogue, setLoadingCatalogue] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [columnLoading, setColumnLoading] = useState(false);
  const [activeRun, setActiveRun] = useState(null);
  const [statusFeed, setStatusFeed] = useState([]);
  const [statusError, setStatusError] = useState(null);
  const [currentStatus, setCurrentStatus] = useState(null);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [showTracking, setShowTracking] = useState(false);

  const updateForm = (field, value) =>
    setForm((prev) => ({ ...prev, [field]: value }));

  const formatElapsedTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) return `${mins}m ${secs}s`;
    const hours = Math.floor(mins / 60);
    const remainingMins = mins % 60;
    return `${hours}h ${remainingMins}m`;
  };

  const loadCatalogue = () => {
    if (!token) return;
    setLoadingCatalogue(true);
    fetchDatasets(token)
      .then(setCatalogue)
      .catch(() => setCatalogue({ classification: [], regression: [] }))
      .finally(() => setLoadingCatalogue(false));
  };

  useEffect(() => {
    loadCatalogue();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  // Restore active runs on mount
  useEffect(() => {
    if (!token) return;

    const restoreActiveRuns = async () => {
      try {
        // Fetch active runs from backend
        const response = await fetchActiveTrainingRuns(token);
        const backendRuns = response?.active_runs || [];

        // Also check localStorage for any runs
        const storedRun = localStorage.getItem("smartml_active_run");

        if (backendRuns.length > 0) {
          // Use the first active run from backend
          const run = backendRuns[0];
          setActiveRun({
            datasetId: run.dataset_id,
            name: run.name,
          });
        } else if (storedRun) {
          // Fallback to localStorage if backend has no active runs
          try {
            const parsed = JSON.parse(storedRun);
            setActiveRun(parsed);
          } catch (e) {
            localStorage.removeItem("smartml_active_run");
          }
        }
      } catch (error) {
        console.error("Failed to restore active runs:", error);
      }
    };

    restoreActiveRuns();
  }, [token]);

  // Auto-dismiss success alerts after 5 seconds
  useEffect(() => {
    if (status?.type === "success") {
      const timer = setTimeout(() => {
        setStatus(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [status]);

  const hydrateColumns = async (file) => {
    if (!token || !file) return;
    setColumnLoading(true);
    try {
      const response = await previewColumns(token, file);
      setColumns(response);
      setForm((prev) => ({
        ...prev,
        target_col: response?.[0] ?? "",
      }));
    } catch (error) {
      setColumns([]);
      setForm((prev) => ({ ...prev, target_col: "" }));
      setStatus({ type: "error", message: error.message });
    } finally {
      setColumnLoading(false);
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    setDatasetFile(file || null);
    setColumns([]);
    setForm((prev) => ({ ...prev, target_col: "" }));
    if (file) {
      await hydrateColumns(file);
    }
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!datasetFile) {
      setStatus({
        type: "error",
        message: "Attach a dataset before submitting.",
      });
      return;
    }
    setSubmitting(true);
    setStatus(null);
    try {
      const response = await uploadDataset(token, {
        ...form,
        file: datasetFile,
      });

      // Set activeRun immediately to start polling
      if (response?.status?.dataset_id) {
        const newRun = {
          datasetId: response.status.dataset_id,
          name: response.dataset?.original_name ?? "dataset",
        };
        setActiveRun(newRun);
        // Persist to localStorage
        localStorage.setItem("smartml_active_run", JSON.stringify(newRun));
        setStatusFeed([]);
        setStatusError(null);
        setSubmitting(false); // Reset button immediately
      }

      setStatus({
        type: "success",
        message: "Dataset uploaded successfully. Training started.",
      });
      setDatasetFile(null);
      setColumns([]);
      setForm(initialPayload);
      loadCatalogue();
    } catch (error) {
      setStatus({ type: "error", message: error.message });
      setSubmitting(false);
    }
  };

  const handleDeleteDataset = async (filePath) => {
    if (!token) return;
    try {
      await deleteDataset(token, filePath);
      setStatus({ type: "success", message: "Dataset deleted." });
      loadCatalogue();
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  };

  useEffect(() => {
    if (!token || !activeRun?.datasetId) return;
    let cancelled = false;

    const statusPollRef = { current: null };

    const poll = async () => {
      try {
        const res = await fetchTrainingStatus(token, activeRun.datasetId);
        if (!cancelled) {
          setStatusFeed(res?.history ?? []);
          setCurrentStatus(res?.current ?? null);
          setStatusError(null);
          if (
            res?.current?.state === "completed" ||
            res?.current?.state === "error"
          ) {
            setActiveRun((prev) =>
              prev ? { ...prev, terminalState: res.current.state } : prev
            );
            // Clean up localStorage when training completes
            localStorage.removeItem("smartml_active_run");
            return true;
          }
        }
      } catch (error) {
        if (!cancelled) {
          setStatusError(error.message);
        }
      }
      return false;
    };

    const kickoff = async () => {
      const done = await poll();
      if (done) return;
      const interval = setInterval(async () => {
        const finished = await poll();
        if (finished) {
          clearInterval(interval);
        }
      }, 4000);
      statusPollRef.current = interval;
    };

    kickoff();

    return () => {
      cancelled = true;
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current);
      }
    };
  }, [token, activeRun?.datasetId]);

  // Track elapsed time in current stage
  useEffect(() => {
    if (!currentStatus?.timestamp) {
      setTimeElapsed(0);
      return;
    }

    // Don't update timer if in terminal state
    if (
      currentStatus.state === "completed" ||
      currentStatus.state === "error"
    ) {
      const start = new Date(currentStatus.timestamp);
      const now = new Date();
      const diff = Math.floor((now - start) / 1000);
      setTimeElapsed(diff);
      return;
    }

    const updateElapsed = () => {
      const start = new Date(currentStatus.timestamp);
      const now = new Date();
      const diff = Math.floor((now - start) / 1000);
      setTimeElapsed(diff);
    };

    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);

    return () => clearInterval(interval);
  }, [currentStatus?.timestamp, currentStatus?.state]);

  const formattedStatus = useMemo(
    () =>
      statusFeed
        .slice()
        .reverse()
        .map((event, idx) => ({
          ...event,
          id: `${event.phase}-${event.timestamp}-${idx}`,
          time: new Date(event.timestamp || Date.now()).toLocaleTimeString(),
        })),
    [statusFeed]
  );

  if (!token) {
    return (
      <section className="page workspace">
        <header>
          <p className="eyebrow">Your Workspace</p>
          <h1>Ready to Build Your First Model?</h1>
          <p className="lead">
            Sign in to unlock the full power of automated machine learning.
            Upload datasets, train models, and download production-ready AI in
            minutes.
          </p>
        </header>
        <div className="card muted">
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🔐</div>
          <h3 style={{ marginBottom: "1rem", color: "var(--gray-300)" }}>
            Authentication Required
          </h3>
          <p style={{ marginBottom: "1.5rem" }}>
            Please sign in to access your personal workspace and start building
            models.
          </p>
          <NavLink className="btn primary" to="/auth">
            Sign In to Continue
          </NavLink>
        </div>
      </section>
    );
  }
  return (
    <section className="page workspace">
      <header>
        <p className="eyebrow">Your Workspace</p>
        <h1>Build & Train Your Models</h1>
        <p className="lead">
          Welcome back, {profile?.fname ?? "there"}! 👋 Upload your dataset and
          watch as our AI automatically processes, trains, and delivers a
          production-ready model.
        </p>
      </header>

      {status && (
        <div className={`alert ${status.type}`}>
          {status.type === "success" && "✓ "}
          {status.type === "error" && "⚠ "}
          <span>{status.message}</span>
        </div>
      )}

      {activeRun && (
        <div className="training-status-container">
          <div className="training-status-header">
            <div className="status-info">
              <h3>
                <i className="fas fa-robot"></i> Training: {activeRun.name}
              </h3>
              <div className="status-badges">
                {activeRun.terminalState ? (
                  <span
                    className={`badge ${
                      activeRun.terminalState === "completed"
                        ? "success"
                        : "error"
                    }`}
                  >
                    <i
                      className={`fas fa-${
                        activeRun.terminalState === "completed"
                          ? "check"
                          : "times"
                      }`}
                    ></i>
                    {activeRun.terminalState === "completed"
                      ? "Completed"
                      : "Failed"}
                  </span>
                ) : (
                  currentStatus && (
                    <span className="badge running">
                      <i className="fas fa-spinner fa-spin"></i>
                      {currentStatus.phase}
                    </span>
                  )
                )}
                {currentStatus && !activeRun.terminalState && (
                  <span className="badge info">
                    <i className="far fa-clock"></i>
                    {formatElapsedTime(timeElapsed)}
                  </span>
                )}
              </div>
            </div>
            <div className="status-actions">
              {activeRun.terminalState === "completed" && (
                <NavLink to="/models" className="btn primary btn-sm">
                  <i className="fas fa-eye"></i> View Model
                </NavLink>
              )}
              <button
                type="button"
                className="btn ghost btn-sm"
                onClick={() => setShowTracking(!showTracking)}
              >
                <i
                  className={`fas fa-chevron-${showTracking ? "up" : "down"}`}
                ></i>
                {showTracking ? "Hide" : "Show"} Tracking
              </button>
            </div>
          </div>

          {statusError && (
            <p className="muted" style={{ padding: "1rem", margin: 0 }}>
              <i className="fas fa-exclamation-triangle"></i> Status temporarily
              unavailable: {statusError}
            </p>
          )}

          {showTracking && (
            <div className="tracking-details">
              {currentStatus && (
                <div className="current-stage-detail">
                  <p className="current-message">
                    <i className="fas fa-info-circle"></i>
                    {currentStatus.message}
                  </p>
                </div>
              )}

              {formattedStatus.length ? (
                <div className="tracking-timeline">
                  <h4 className="timeline-heading">
                    <i className="fas fa-list-check"></i> Progress Timeline
                  </h4>
                  <div className="timeline-wrapper">
                    {formattedStatus.map((event, index) => {
                      const isActive =
                        index === 0 && currentStatus?.phase === event.phase;
                      const isCompleted =
                        event.state === "completed" ||
                        (!isActive && index !== 0);
                      const isError = event.state === "error";

                      return (
                        <div
                          key={event.id}
                          className={`timeline-item ${
                            isActive ? "active" : ""
                          } ${isCompleted ? "completed" : ""} ${
                            isError ? "error" : ""
                          }`}
                        >
                          <div className="timeline-marker">
                            <i
                              className={`fas fa-${
                                isError
                                  ? "times"
                                  : isCompleted
                                  ? "check"
                                  : "circle"
                              }`}
                            ></i>
                          </div>
                          <div className="timeline-content">
                            <div className="timeline-header">
                              <span className="timeline-phase">
                                {event.phase}
                              </span>
                              <span className="timeline-time">
                                {event.time}
                              </span>
                            </div>
                            <p className="timeline-message">{event.message}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : !currentStatus ? (
                <p
                  className="muted"
                  style={{ padding: "1rem", textAlign: "center" }}
                >
                  <i className="fas fa-hourglass-start"></i> Waiting for
                  training to begin...
                </p>
              ) : null}
            </div>
          )}
        </div>
      )}

      <form onSubmit={handleUpload} className="card form">
        <div className="form-heading">
          <h3>
            <i className="fas fa-cloud-upload-alt"></i> Upload Your Dataset
          </h3>
          <p>
            Start by uploading a CSV file. We'll analyze it, train the optimal
            model, and have it ready for download in minutes.
          </p>
        </div>
        <label className="file-input">
          <span>
            <i className="fas fa-file-csv"></i> Choose CSV File
          </span>
          <input type="file" accept=".csv" onChange={handleFileChange} />
        </label>
        {columnLoading && (
          <span className="badge">
            <i className="fas fa-sync fa-spin"></i> Analyzing columns...
          </span>
        )}

        <div className="form-grid">
          <label>
            What type of prediction?
            <select
              value={form.task_type}
              onChange={(e) => updateForm("task_type", e.target.value)}
            >
              <option value="classification">
                Classification (Categories)
              </option>
              <option value="regression">Regression (Numbers)</option>
            </select>
          </label>
          <label>
            Target column (what to predict)
            <select
              value={form.target_col}
              onChange={(e) => updateForm("target_col", e.target.value)}
              disabled={!columns.length}
            >
              <option value="">
                {columns.length ? "Choose target column" : "Upload file first"}
              </option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </label>
          <label>
            Optimize performance?
            <select
              value={form.tuning}
              onChange={(e) => updateForm("tuning", e.target.value)}
            >
              <option value="true">Yes (Recommended)</option>
              <option value="false">No (Faster)</option>
            </select>
          </label>
        </div>

        <div className="actions">
          <button
            type="submit"
            className="btn primary"
            disabled={submitting || !form.target_col}
          >
            {submitting ? (
              <>
                <i className="fas fa-spinner fa-spin"></i> Uploading...
              </>
            ) : (
              <>
                <i className="fas fa-rocket"></i> Start Training
              </>
            )}
          </button>
        </div>
      </form>

      <section className="catalogue">
        <header>
          <h2>
            <i className="fas fa-database"></i> Your Datasets
          </h2>
          {loadingCatalogue && (
            <span className="badge">
              <i className="fas fa-sync fa-spin"></i> Refreshing...
            </span>
          )}
        </header>
        <div className="grid two">
          {["classification", "regression"].map((type) => (
            <div key={type} className="card">
              <h3 style={{ textTransform: "capitalize", marginBottom: "1rem" }}>
                <i
                  className={`fas fa-${
                    type === "classification" ? "tag" : "chart-line"
                  }`}
                ></i>{" "}
                {type}
              </h3>
              {catalogue[type]?.length ? (
                <ul
                  style={{
                    listStyle: "none",
                    padding: 0,
                    margin: 0,
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.75rem",
                  }}
                >
                  {catalogue[type].map((file) => (
                    <li key={file.download_url} className="catalogue-item">
                      <span style={{ flex: 1, color: "var(--gray-200)" }}>
                        {file.name}
                      </span>
                      <div className="catalogue-actions">
                        <button
                          type="button"
                          className="icon-btn"
                          onClick={() =>
                            downloadDataset(token, file.download_url, file.name)
                          }
                          aria-label={`Download ${file.name}`}
                          title="Download dataset"
                        >
                          <svg
                            width="16"
                            height="16"
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
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p
                  className="muted"
                  style={{ padding: "2rem", textAlign: "center" }}
                >
                  No {type} datasets yet. Upload your first one above! 👆
                </p>
              )}
            </div>
          ))}
        </div>
      </section>
    </section>
  );
};

export default Workspace;

import { useEffect, useMemo, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  previewColumns,
  uploadDataset,
  fetchDatasets,
  downloadDataset,
  deleteDataset,
  fetchTrainingStatus,
} from '../lib/api'
import { useSession } from '../context/SessionContext'

const initialPayload = {
  task_type: 'classification',
  target_col: '',
  tuning: 'false',
}

const Workspace = () => {
  const { token, profile } = useSession()
  const [datasetFile, setDatasetFile] = useState(null)
  const [form, setForm] = useState(initialPayload)
  const [columns, setColumns] = useState([])
  const [status, setStatus] = useState(null)
  const [catalogue, setCatalogue] = useState({ classification: [], regression: [] })
  const [loadingCatalogue, setLoadingCatalogue] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [columnLoading, setColumnLoading] = useState(false)
  const [activeRun, setActiveRun] = useState(null)
  const [statusFeed, setStatusFeed] = useState([])
  const [statusError, setStatusError] = useState(null)

  const updateForm = (field, value) => setForm((prev) => ({ ...prev, [field]: value }))

  const loadCatalogue = () => {
    if (!token) return
    setLoadingCatalogue(true)
    fetchDatasets(token)
      .then(setCatalogue)
      .catch(() => setCatalogue({ classification: [], regression: [] }))
      .finally(() => setLoadingCatalogue(false))
  }

  useEffect(() => {
    loadCatalogue()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  const hydrateColumns = async (file) => {
    if (!token || !file) return
    setColumnLoading(true)
    try {
      const response = await previewColumns(token, file)
      setColumns(response)
      setForm((prev) => ({
        ...prev,
        target_col: response?.[0] ?? '',
      }))
    } catch (error) {
      setColumns([])
      setForm((prev) => ({ ...prev, target_col: '' }))
      setStatus({ type: 'error', message: error.message })
    } finally {
      setColumnLoading(false)
    }
  }

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0]
    setDatasetFile(file || null)
    setColumns([])
    setForm((prev) => ({ ...prev, target_col: '' }))
    if (file) {
      await hydrateColumns(file)
    }
  }

  const handleUpload = async (event) => {
    event.preventDefault()
    if (!datasetFile) {
      setStatus({ type: 'error', message: 'Attach a dataset before submitting.' })
      return
    }
    setSubmitting(true)
    setStatus(null)
    try {
      const response = await uploadDataset(token, { ...form, file: datasetFile })
      setStatus({
        type: 'success',
        message: 'Dataset received. Sit tight while we preprocess and train.',
      })
      setDatasetFile(null)
      setColumns([])
      setForm(initialPayload)
      if (response?.status?.dataset_id) {
        setActiveRun({
          datasetId: response.status.dataset_id,
          name: response.dataset?.original_name ?? 'dataset',
        })
        setStatusFeed([])
        setStatusError(null)
      }
      loadCatalogue()
    } catch (error) {
      setStatus({ type: 'error', message: error.message })
    } finally {
      setSubmitting(false)
    }
  }

  const handleDeleteDataset = async (filePath) => {
    if (!token) return
    try {
      await deleteDataset(token, filePath)
      setStatus({ type: 'success', message: 'Dataset deleted.' })
      loadCatalogue()
    } catch (error) {
      setStatus({ type: 'error', message: error.message })
    }
  }

  useEffect(() => {
    if (!token || !activeRun?.datasetId) return
    let cancelled = false

    const statusPollRef = { current: null }

    const poll = async () => {
      try {
        const res = await fetchTrainingStatus(token, activeRun.datasetId)
        if (!cancelled) {
          setStatusFeed(res?.history ?? [])
          setStatusError(null)
          if (res?.current?.state === 'completed' || res?.current?.state === 'error') {
            setActiveRun((prev) => (prev ? { ...prev, terminalState: res.current.state } : prev))
            return true
          }
        }
      } catch (error) {
        if (!cancelled) {
          setStatusError(error.message)
        }
      }
      return false
    }

    const kickoff = async () => {
      const done = await poll()
      if (done) return
      const interval = setInterval(async () => {
        const finished = await poll()
        if (finished) {
          clearInterval(interval)
        }
      }, 4000)
      statusPollRef.current = interval
    }

    kickoff()

    return () => {
      cancelled = true
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current)
      }
    }
  }, [token, activeRun?.datasetId])

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
  )

  if (!token) {
    return (
      <section className="page workspace">
        <header>
          <h1>Workspace</h1>
          <p className="lead">Authenticate to stream datasets to FastAPI.</p>
        </header>
        <div className="card muted">
          <p>Bring your JWT token by logging in first. Don’t have one yet? Create an account below.</p>
          <NavLink className="btn primary" to="/auth">
            Login / Sign up
          </NavLink>
        </div>
      </section>
    )
  }
  return (
    <section className="page workspace">
      <header>
        <p className="eyebrow">Pipelines</p>
        <h1>Upload datasets & orchestrate training</h1>
        <p className="lead">
          Hi {profile?.fname ?? 'there'}, ship CSV datasets straight to <code>/users/send_dataset</code>. We automatically
          persist the file under your user namespace before triggering the AutoML user worker.
        </p>
      </header>

      {status && (
        <div className={`alert ${status.type}`}>
          <span>{status.message}</span>
        </div>
      )}

      {activeRun && (
        <div className="card status-feed">
          <div className="form-heading">
            <h3>Live training status</h3>
            <p>Tracking: {activeRun.name}</p>
            {activeRun.terminalState && (
              <span className={`badge ${activeRun.terminalState === 'completed' ? 'success' : 'error'}`}>
                {activeRun.terminalState === 'completed' ? 'Finished' : 'Failed'}
              </span>
            )}
          </div>
          {statusError && <p className="muted">Status temporarily unavailable: {statusError}</p>}
          {formattedStatus.length ? (
            <ol className="status-timeline">
              {formattedStatus.map((event) => (
                <li key={event.id}>
                  <div className="status-header">
                    <strong>{event.phase}</strong>
                    <span>{event.time}</span>
                  </div>
                  <p>{event.message}</p>
                </li>
              ))}
            </ol>
          ) : (
            <p className="muted">Waiting for the first update…</p>
          )}
        </div>
      )}

      <form onSubmit={handleUpload} className="card form">
        <div className="form-heading">
          <h3>Dataset intake</h3>
          <p>We store the artifact and immediately kick off preprocessing, meta-feature extraction and model search.</p>
        </div>
        <label className="file-input">
          <span>Dataset (CSV)</span>
          <input type="file" accept=".csv" onChange={handleFileChange} />
        </label>
        {columnLoading && <span className="badge">Detecting columns…</span>}

        <div className="form-grid">
          <label>
            Task type
            <select value={form.task_type} onChange={(e) => updateForm('task_type', e.target.value)}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </label>
          <label>
            Target column
            <select
              value={form.target_col}
              onChange={(e) => updateForm('target_col', e.target.value)}
              disabled={!columns.length}
            >
              <option value="">{columns.length ? 'Select a target column' : 'Upload a dataset first'}</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </label>
          <label>
            Hyperparameter tuning
            <select value={form.tuning} onChange={(e) => updateForm('tuning', e.target.value)}>
              <option value="true">Enabled</option>
              <option value="false">Disabled</option>
            </select>
          </label>
        </div>

        <div className="actions">
          <button type="submit" className="btn primary" disabled={submitting || !form.target_col}>
            {submitting ? 'Uploading…' : 'Send dataset'}
          </button>
        </div>
      </form>

      <section className="catalogue">
        <header>
          <h2>Your dataset catalogue</h2>
          {loadingCatalogue && <span className="badge">Refreshing…</span>}
        </header>
        <div className="grid two">
          {['classification', 'regression'].map((type) => (
            <div key={type} className="card">
              <h3>{type}</h3>
              {catalogue[type]?.length ? (
                <ul>
                  {catalogue[type].map((file) => (
                    <li key={file.download_url} className="catalogue-item">
                      <span>{file.name}</span>
                      <div className="catalogue-actions">
                        <button
                          type="button"
                          className="icon-btn"
                          onClick={() => downloadDataset(token, file.download_url, file.name)}
                          aria-label={`Download ${file.name}`}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
                          onClick={() => handleDeleteDataset(file.path)}
                          aria-label={`Delete ${file.name}`}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
                <p className="muted">No datasets yet.</p>
              )}
            </div>
          ))}
        </div>
      </section>
    </section>
  )
}

export default Workspace


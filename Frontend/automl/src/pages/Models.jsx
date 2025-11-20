import { useEffect, useState } from 'react'
import { fetchModels, downloadModel, deleteModel, fetchBundles, downloadBundle } from '../lib/api'
import { useSession } from '../context/SessionContext'

const Models = () => {
  const { token, profile } = useSession()
  const [models, setModels] = useState({ classification: [], regression: [] })
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [bundles, setBundles] = useState({ classification: [], regression: [] })

  useEffect(() => {
    if (!token) return
    const hydrate = async () => {
      setLoading(true)
      try {
        const [modelResp, bundleResp] = await Promise.all([fetchModels(token), fetchBundles(token)])
        setModels(modelResp)
        setBundles(bundleResp)
      } catch (error) {
        setStatus({ type: 'error', message: error.message })
      } finally {
        setLoading(false)
      }
    }
    hydrate()
  }, [token])

  const handleDelete = async (modelPath) => {
    if (!token) return
    try {
      await deleteModel(token, modelPath)
      setStatus({ type: 'success', message: 'Model deleted.' })
      const [refreshedModels, refreshedBundles] = await Promise.all([fetchModels(token), fetchBundles(token)])
      setModels(refreshedModels)
      setBundles(refreshedBundles)
    } catch (error) {
      setStatus({ type: 'error', message: error.message })
    }
  }

  const formatMetric = (metric) => {
    if (metric === null || metric === undefined) return 'N/A'
    return Number(metric).toFixed(3)
  }

  if (!token) {
    return (
      <section className="page models">
        <header>
          <h1>Model registry</h1>
          <p className="lead">Authenticate to view your trained models.</p>
        </header>
      </section>
    )
  }

  return (
    <section className="page models">
      <header>
        <p className="eyebrow">Model Registry</p>
        <h1>Download-ready model artifacts</h1>
        <p className="lead">
          Models are stored under <code>{profile?.username}</code> in the FastAPI storage layer. Use this console to
          download the packaged artifacts and integrate them into downstream workloads.
        </p>
      </header>

      {status && <div className={`alert ${status.type}`}>{status.message}</div>}
      {loading && <span className="badge">Syncing models…</span>}

      <div className="grid two">
        {['classification', 'regression'].map((type) => (
          <div key={type} className="card">
            <h3>{type} models</h3>
            {models[type]?.length ? (
              <ul className="model-list">
                {models[type].map((model) => (
                  <li key={model.name} className="model-row">
                    <div className="model-meta">
                      <strong>{model.name}</strong>
                      <small>{model.model_label || 'Auto-selected model'}</small>
                      {model.model_reason && <p>{model.model_reason}</p>}
                      {model.human_metric ? (
                        <span className="badge success">{model.human_metric}</span>
                      ) : model.metric_name ? (
                        <span className="badge">{`${model.metric_name}: ${formatMetric(model.metric_value)}`}</span>
                      ) : (
                        <span className="badge muted">Awaiting evaluation</span>
                      )}
                      {model.explanations?.length ? (
                        <div className="explanations">
                          {model.explanations.slice(0, 4).map((item) => (
                            <span key={`${model.name}-${item.feature}`} className="pill">
                              {item.feature}: {Number(item.weight).toFixed(2)}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="muted">LIME explanation not available.</p>
                      )}
                    </div>
                    <div className="model-actions">
                      <button
                        type="button"
                        className="icon-btn"
                        aria-label={`Download ${model.name}`}
                        onClick={() => downloadModel(model.download_url, model.name)}
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
              <p className="muted">No models for {type} yet.</p>
            )}
          </div>
        ))}
      </div>

      <section className="bundles">
        <header>
          <h2>Downloadable Streamlit bundles</h2>
          <p className="lead">Each zip includes the trained model, preprocessing artifact, and a ready-to-run app.</p>
        </header>
        <div className="grid two">
          {['classification', 'regression'].map((type) => (
            <div key={`bundle-${type}`} className="card">
              <h3>{type}</h3>
              {bundles[type]?.length ? (
                <ul className="model-list">
                  {bundles[type].map((bundle) => (
                    <li key={bundle.path} className="model-row">
                      <div className="model-meta">
                        <strong>{bundle.name}</strong>
                        <small>
                          Updated {new Date(bundle.modified_ts * 1000).toLocaleString()} ·{' '}
                          {(bundle.size_bytes / (1024 * 1024)).toFixed(2)} MB
                        </small>
                      </div>
                      <div className="model-actions">
                        <button
                          type="button"
                          className="icon-btn"
                          aria-label={`Download bundle ${bundle.name}`}
                          onClick={() => downloadBundle(token, bundle.download_url, `${bundle.name}.zip`)}
                        >
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
                <p className="muted">No bundles yet.</p>
              )}
            </div>
          ))}
        </div>
      </section>
    </section>
  )
}

export default Models


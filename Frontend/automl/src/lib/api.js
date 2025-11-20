const apiBase = import.meta.env?.VITE_API_BASE
const API_BASE = (apiBase ? apiBase.replace(/\/$/, '') : 'http://localhost:8000')

const withAuth = (token, headers = {}) => {
  if (!token) return headers
  return { ...headers, Authorization: `Bearer ${token}` }
}

const handleResponse = async (res) => {
  if (!res.ok) {
    const text = await res.text()
    let message = text
    try {
      const parsed = JSON.parse(text)
      message = parsed.detail || parsed.msg || text
    } catch {
      // ignore parse error
    }
    throw new Error(message || 'Unexpected server error')
  }
  const contentType = res.headers.get('content-type') ?? ''
  if (contentType.includes('application/json')) {
    return res.json()
  }
  return res.blob()
}

export const persistToken = (token) => localStorage.setItem('metaml_token', token)
export const clearTokenStorage = () => localStorage.removeItem('metaml_token')

export const registerUser = (payload) =>
  fetch(`${API_BASE}/users/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(handleResponse)

export const loginUser = (payload) =>
  fetch(`${API_BASE}/users/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(handleResponse)

export const fetchProfile = (token) =>
  fetch(`${API_BASE}/users/profile`, {
    headers: withAuth(token),
  }).then(handleResponse)

export const previewColumns = async (token, file) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/users/get_columns`, {
    method: 'POST',
    headers: withAuth(token),
    body: formData,
  })
  return handleResponse(res)
}

export const uploadDataset = async (token, payload) => {
  const formData = new FormData()
  formData.append('task_type', payload.task_type)
  formData.append('target_col', payload.target_col)
  formData.append('tuning', payload.tuning)
  formData.append('file', payload.file)

  const res = await fetch(`${API_BASE}/users/send_dataset`, {
    method: 'POST',
    headers: withAuth(token),
    body: formData,
  })
  return handleResponse(res)
}

export const fetchModels = (token) =>
  fetch(`${API_BASE}/users/get_models`, {
    headers: withAuth(token),
  }).then(handleResponse)

export const fetchBundles = (token) =>
  fetch(`${API_BASE}/users/get_bundles`, {
    headers: withAuth(token),
  }).then(handleResponse)

export const fetchDatasets = (token) =>
  fetch(`${API_BASE}/users/get_datasets`, {
    headers: withAuth(token),
  }).then(handleResponse)

export const fetchTrainingStatus = (token, datasetId) =>
  fetch(`${API_BASE}/users/training_status?dataset_id=${encodeURIComponent(datasetId)}`, {
    headers: withAuth(token),
  }).then(handleResponse)

const downloadBinary = async (path, filename, token, requiresAuth = false) => {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: requiresAuth ? withAuth(token) : undefined,
  })
  const blob = await handleResponse(res)
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

export const downloadModel = (path, filename) =>
  downloadBinary(path, filename, null, false)

export const downloadDataset = (token, path, filename) =>
  downloadBinary(path, filename, token, true)

export const downloadBundle = (token, path, filename) =>
  downloadBinary(path, filename, token, true)

export const deleteDataset = (token, filePath) =>
  fetch(`${API_BASE}/users/delete_dataset?file_path=${encodeURIComponent(filePath)}`, {
    method: 'DELETE',
    headers: withAuth(token),
  }).then(handleResponse)

export const deleteModel = (token, filePath) =>
  fetch(`${API_BASE}/users/delete_model?file_path=${encodeURIComponent(filePath)}`, {
    method: 'DELETE',
    headers: withAuth(token),
  }).then(handleResponse)


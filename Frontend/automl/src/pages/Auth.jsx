import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { loginUser, registerUser } from '../lib/api'
import { useSession } from '../context/SessionContext'

const Auth = () => {
  const { setToken, profile } = useSession()
  const navigate = useNavigate()
  const [loginForm, setLoginForm] = useState({ email: '', password: '' })
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [registerForm, setRegisterForm] = useState({
    fname: '',
    lname: '',
    username: '',
    email: '',
    password: '',
    cpassword: '',
  })
  const [registerStatus, setRegisterStatus] = useState(null)
  const [registerLoading, setRegisterLoading] = useState(false)

  const updateLogin = (field, value) => {
    setLoginForm((prev) => ({ ...prev, [field]: value }))
  }

  const updateRegister = (field, value) => {
    setRegisterForm((prev) => ({ ...prev, [field]: value }))
  }

  const handleLogin = async (event) => {
    event.preventDefault()
    setStatus(null)
    setLoading(true)
    try {
      const response = await loginUser(loginForm)
      setToken(response.token)
      setStatus({ type: 'success', message: 'Logged in successfully.' })
      navigate('/')
    } catch (error) {
      setStatus({ type: 'error', message: error.message })
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async (event) => {
    event.preventDefault()
    setRegisterStatus(null)
    setRegisterLoading(true)
    try {
      const response = await registerUser(registerForm)
      setRegisterStatus({ type: 'success', message: response.msg || 'Account created. You can sign in now.' })
      setRegisterForm({ fname: '', lname: '', username: '', email: '', password: '', cpassword: '' })
    } catch (error) {
      setRegisterStatus({ type: 'error', message: error.message })
    } finally {
      setRegisterLoading(false)
    }
  }

  return (
    <section className="page auth">
      <header>
        <p className="eyebrow">Identity & Governance</p>
        <h1>Secure access to your SmartML control plane</h1>
        <p className="lead">
          All workspace interactions are JWT-gated. Request access from your platform team, then sign in below to pull a
          fresh session token directly from FastAPI.
        </p>
      </header>

      {status && (
        <div className={`alert ${status.type}`}>
          <span>{status.message}</span>
        </div>
      )}

      <form onSubmit={handleLogin}>
        <div className="form-heading">
          <h3>Sign in</h3>
          <p>Use your provisioned credentials to mint a new JWT.</p>
          {profile && <span className="badge success">Active session detected for {profile.email}</span>}
        </div>
        <label>
          Email
          <input
            type="email"
            value={loginForm.email}
            onChange={(e) => updateLogin('email', e.target.value)}
            required
          />
        </label>
        <label>
          Password
          <input
            type="password"
            value={loginForm.password}
            onChange={(e) => updateLogin('password', e.target.value)}
            required
          />
        </label>
        <button className="btn primary full" disabled={loading}>
          {loading ? 'Verifying…' : 'Login'}
        </button>
      </form>

      <form onSubmit={handleRegister} className="card form secondary">
        <div className="form-heading">
          <h3>Create an account</h3>
          <p>Need access? Sign up directly and we’ll provision your workspace profile.</p>
        </div>
        {registerStatus && <div className={`alert ${registerStatus.type}`}>{registerStatus.message}</div>}
        <div className="form-grid">
          <label>
            First name
            <input type="text" value={registerForm.fname} onChange={(e) => updateRegister('fname', e.target.value)} required />
          </label>
          <label>
            Last name
            <input type="text" value={registerForm.lname} onChange={(e) => updateRegister('lname', e.target.value)} required />
          </label>
        </div>
        <div className="form-grid">
          <label>
            Username
            <input type="text" value={registerForm.username} onChange={(e) => updateRegister('username', e.target.value)} required />
          </label>
          <label>
            Email
            <input type="email" value={registerForm.email} onChange={(e) => updateRegister('email', e.target.value)} required />
          </label>
        </div>
        <div className="form-grid">
          <label>
            Password
            <input type="password" value={registerForm.password} onChange={(e) => updateRegister('password', e.target.value)} required />
          </label>
          <label>
            Confirm password
            <input type="password" value={registerForm.cpassword} onChange={(e) => updateRegister('cpassword', e.target.value)} required />
          </label>
        </div>
        <button className="btn ghost full" disabled={registerLoading}>
          {registerLoading ? 'Creating…' : 'Sign up'}
        </button>
      </form>

      <div className="card muted">
        <p>Already have credentials? Use the sign-in form above to grab a fresh token.</p>
      </div>
    </section>
  )
}

export default Auth


import { NavLink, useNavigate } from 'react-router-dom'
import { useSession } from '../context/SessionContext'

const links = [
  { path: '/', label: 'Overview' },
  { path: '/workspace', label: 'Workspace' },
  { path: '/models', label: 'Models' },
]

const Navigation = () => {
  const { profile, logout } = useSession()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/auth')
  }

  return (
    <header className="nav-shell">
      <NavLink to="/" className="brand">
        <span className="brand-mark">MetaML</span>
        <span className="brand-subtitle">Autopilot</span>
      </NavLink>
      <nav>
        {links.map((link) => (
          <NavLink
            key={link.path}
            to={link.path}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            {link.label}
          </NavLink>
        ))}
        {profile ? (
          <>
            <NavLink to="/auth" className="nav-link nav-cta">
              Hi, {profile.fname}
            </NavLink>
            <button type="button" className="nav-link nav-logout" onClick={handleLogout}>
              Logout
            </button>
          </>
        ) : (
          <NavLink to="/auth" className="nav-link nav-cta">
            Access
          </NavLink>
        )}
      </nav>
    </header>
  )
}

export default Navigation


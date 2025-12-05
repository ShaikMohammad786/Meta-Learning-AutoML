import { NavLink, useNavigate } from "react-router-dom";
import { useSession } from "../context/SessionContext";

const links = [
  { path: "/", label: "Home", icon: "fa-home" },
  { path: "/workspace", label: "Workspace", icon: "fa-bolt" },
  { path: "/models", label: "Models", icon: "fa-robot" },
];

const Navigation = () => {
  const { profile, logout } = useSession();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/auth");
  };

  return (
    <header className="nav-shell">
      <NavLink to="/" className="brand">
        <span className="brand-mark">SmartML</span>
        <span className="brand-subtitle">AutoML Platform</span>
      </NavLink>
      <nav>
        {links.map((link) => (
          <NavLink
            key={link.path}
            to={link.path}
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            <i
              className={`fas ${link.icon}`}
              style={{ marginRight: "0.5rem" }}
            ></i>
            {link.label}
          </NavLink>
        ))}
        {profile ? (
          <>
            <NavLink to="/auth" className="nav-link nav-cta">
              <i className="fas fa-user"></i> {profile.fname}
            </NavLink>
            <button
              type="button"
              className="nav-link nav-logout"
              onClick={handleLogout}
            >
              Sign Out
            </button>
          </>
        ) : (
          <NavLink to="/auth" className="nav-link nav-cta">
            Sign In
          </NavLink>
        )}
      </nav>
    </header>
  );
};

export default Navigation;

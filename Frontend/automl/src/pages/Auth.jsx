import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { loginUser, registerUser } from "../lib/api";
import { useSession } from "../context/SessionContext";

const Auth = () => {
  const { setToken, profile } = useSession();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("signin");
  const [loginForm, setLoginForm] = useState({ email: "", password: "" });
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [registerForm, setRegisterForm] = useState({
    fname: "",
    lname: "",
    username: "",
    email: "",
    password: "",
    cpassword: "",
  });
  const [registerStatus, setRegisterStatus] = useState(null);
  const [registerLoading, setRegisterLoading] = useState(false);

  const updateLogin = (field, value) => {
    setLoginForm((prev) => ({ ...prev, [field]: value }));
  };

  const updateRegister = (field, value) => {
    setRegisterForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleLogin = async (event) => {
    event.preventDefault();
    setStatus(null);
    setLoading(true);
    try {
      const response = await loginUser(loginForm);
      setToken(response.token);
      setStatus({ type: "success", message: "Logged in successfully." });
      navigate("/");
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (event) => {
    event.preventDefault();
    setRegisterStatus(null);
    setRegisterLoading(true);
    try {
      const response = await registerUser(registerForm);
      setRegisterStatus({
        type: "success",
        message: response.msg || "Account created. You can sign in now.",
      });
      setRegisterForm({
        fname: "",
        lname: "",
        username: "",
        email: "",
        password: "",
        cpassword: "",
      });
    } catch (error) {
      setRegisterStatus({ type: "error", message: error.message });
    } finally {
      setRegisterLoading(false);
    }
  };

  // If user is already logged in, show a different view
  if (profile) {
    return (
      <section className="page auth">
        <header>
          <p className="eyebrow">Already Authenticated</p>
          <h1>Welcome Back, {profile.fname || profile.username}!</h1>
          <p className="lead">
            You're already signed in. Ready to build amazing ML models?
          </p>
        </header>

        <div className="auth-container">
          <div
            className="card"
            style={{
              textAlign: "center",
              padding: "3rem 2rem",
              background:
                "linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(34, 197, 94, 0.08))",
              border: "1px solid rgba(14, 165, 233, 0.2)",
            }}
          >
            <div
              style={{
                width: "80px",
                height: "80px",
                margin: "0 auto 1.5rem",
                background:
                  "linear-gradient(135deg, var(--primary-500), var(--accent-500))",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "2rem",
                boxShadow: "0 8px 24px rgba(14, 165, 233, 0.3)",
              }}
            >
              <i className="fas fa-user-check"></i>
            </div>
            <h2 style={{ marginBottom: "0.75rem", color: "var(--gray-100)" }}>
              You're All Set!
            </h2>
            <p
              style={{
                color: "var(--gray-400)",
                marginBottom: "2rem",
                fontSize: "1.0625rem",
              }}
            >
              Signed in as{" "}
              <strong style={{ color: "var(--primary-400)" }}>
                {profile.email}
              </strong>
            </p>
            <div
              style={{
                display: "flex",
                gap: "1rem",
                justifyContent: "center",
                flexWrap: "wrap",
              }}
            >
              <button
                onClick={() => navigate("/workspace")}
                className="btn primary"
                style={{ minWidth: "180px" }}
              >
                <i className="fas fa-rocket"></i> Go to Workspace
              </button>
              <button
                onClick={() => navigate("/")}
                className="btn secondary"
                style={{ minWidth: "180px" }}
              >
                <i className="fas fa-home"></i> Back to Home
              </button>
            </div>
          </div>
        </div>

        <div
          className="card"
          style={{
            textAlign: "center",
            padding: "2rem",
            background:
              "linear-gradient(135deg, rgba(168, 85, 247, 0.08), rgba(14, 165, 233, 0.08))",
            border: "1px solid rgba(168, 85, 247, 0.2)",
          }}
        >
          <h3 style={{ marginBottom: "0.75rem" }}>
            <i className="fas fa-lightbulb"></i> Quick Tips
          </h3>
          <p style={{ color: "var(--gray-400)" }}>
            Upload your dataset in the Workspace, let SmartML handle
            preprocessing and model selection, then download your trained models
            ready for deployment.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="page auth">
      <header>
        <p className="eyebrow">Account Access</p>
        <h1>Welcome to SmartML</h1>
        <p className="lead">
          Sign in to access your personal AutoML workspace or create a new
          account to get started with automated machine learning.
        </p>
      </header>

      <div className="auth-container">
        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${activeTab === "signin" ? "active" : ""}`}
            onClick={() => setActiveTab("signin")}
          >
            <i className="fas fa-key"></i> Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${activeTab === "signup" ? "active" : ""}`}
            onClick={() => setActiveTab("signup")}
          >
            <i className="fas fa-user-plus"></i> Create Account
          </button>
        </div>

        {activeTab === "signin" && (
          <form onSubmit={handleLogin} className="auth-form">
            {status && (
              <div className={`alert ${status.type}`}>
                <i
                  className={`fas fa-${
                    status.type === "success"
                      ? "check-circle"
                      : "exclamation-triangle"
                  }`}
                ></i>
                <span>{status.message}</span>
              </div>
            )}
            <label>
              Email Address
              <input
                type="email"
                placeholder="you@example.com"
                value={loginForm.email}
                onChange={(e) => updateLogin("email", e.target.value)}
                required
              />
            </label>
            <label>
              Password
              <input
                type="password"
                placeholder="Enter your password"
                value={loginForm.password}
                onChange={(e) => updateLogin("password", e.target.value)}
                required
              />
            </label>
            <button className="btn primary full" disabled={loading}>
              {loading ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i> Signing In...
                </>
              ) : (
                <>
                  <i className="fas fa-sign-in-alt"></i> Sign In
                </>
              )}
            </button>
          </form>
        )}

        {activeTab === "signup" && (
          <form onSubmit={handleRegister} className="auth-form">
            {registerStatus && (
              <div className={`alert ${registerStatus.type}`}>
                <i
                  className={`fas fa-${
                    registerStatus.type === "success"
                      ? "check-circle"
                      : "exclamation-triangle"
                  }`}
                ></i>
                {registerStatus.message}
              </div>
            )}
            <div className="form-grid">
              <label>
                First Name
                <input
                  type="text"
                  placeholder="John"
                  value={registerForm.fname}
                  onChange={(e) => updateRegister("fname", e.target.value)}
                  required
                />
              </label>
              <label>
                Last Name
                <input
                  type="text"
                  placeholder="Doe"
                  value={registerForm.lname}
                  onChange={(e) => updateRegister("lname", e.target.value)}
                  required
                />
              </label>
            </div>
            <label>
              Username
              <input
                type="text"
                placeholder="johndoe"
                value={registerForm.username}
                onChange={(e) => updateRegister("username", e.target.value)}
                required
              />
            </label>
            <label>
              Email Address
              <input
                type="email"
                placeholder="you@example.com"
                value={registerForm.email}
                onChange={(e) => updateRegister("email", e.target.value)}
                required
              />
            </label>
            <div className="form-grid">
              <label>
                Password
                <input
                  type="password"
                  placeholder="Create password"
                  value={registerForm.password}
                  onChange={(e) => updateRegister("password", e.target.value)}
                  required
                />
              </label>
              <label>
                Confirm Password
                <input
                  type="password"
                  placeholder="Confirm password"
                  value={registerForm.cpassword}
                  onChange={(e) => updateRegister("cpassword", e.target.value)}
                  required
                />
              </label>
            </div>
            <button className="btn primary full" disabled={registerLoading}>
              {registerLoading ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i> Creating Account...
                </>
              ) : (
                <>
                  <i className="fas fa-rocket"></i> Create Free Account
                </>
              )}
            </button>
          </form>
        )}
      </div>

      <div
        className="card"
        style={{
          textAlign: "center",
          padding: "2rem",
          background:
            "linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(34, 197, 94, 0.08))",
          border: "1px solid rgba(14, 165, 233, 0.2)",
        }}
      >
        <h3 style={{ marginBottom: "0.75rem" }}>
          <i className="fas fa-bullseye"></i> Why SmartML?
        </h3>
        <p style={{ color: "var(--gray-400)" }}>
          Join hundreds of users building AI models without code. Our platform
          automates the entire ML pipeline, from data preprocessing to model
          deployment.
        </p>
      </div>
    </section>
  );
};

export default Auth;

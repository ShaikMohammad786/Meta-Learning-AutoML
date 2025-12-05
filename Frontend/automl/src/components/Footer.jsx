const Footer = () => (
  <footer className="site-footer">
    <div>
      <p className="brand-mark" style={{ marginBottom: "0.5rem" }}>
        SmartML
      </p>
      <small style={{ color: "var(--gray-500)" }}>
        Automated Machine Learning Platform
      </small>
      <br />
      <small style={{ color: "var(--gray-600)", fontSize: "0.75rem" }}>
        © {new Date().getFullYear()} SmartML. Empowering data science teams.
      </small>
    </div>
    <div className="footer-links">
      <a
        href="mailto:support@smartml.ai"
        style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
      >
        <i className="fas fa-envelope"></i> Support
      </a>
      <a
        href="https://fastapi.tiangolo.com/"
        target="_blank"
        rel="noreferrer"
        style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
      >
        <i className="fas fa-rocket"></i> Powered by FastAPI
      </a>
      <a
        href="https://react.dev/"
        target="_blank"
        rel="noreferrer"
        style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
      >
        <i className="fab fa-react"></i> Built with React
      </a>
    </div>
  </footer>
);

export default Footer;

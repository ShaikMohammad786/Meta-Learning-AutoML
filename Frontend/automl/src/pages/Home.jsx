import { motion } from "framer-motion";
import { NavLink } from "react-router-dom";
import { useSession } from "../context/SessionContext";

const stats = [
  { label: "Models Trained", value: "500+", icon: "fa-robot" },
  { label: "Data Processed", value: "2.4M+", icon: "fa-chart-bar" },
  { label: "Average Time", value: "<7 min", icon: "fa-bolt" },
  { label: "Accuracy Rate", value: "65%", icon: "fa-bullseye" },
];

const features = [
  {
    title: "Upload & Analyze",
    body: "Simply upload your dataset and let our AI do the heavy lifting. No coding required—just pure insights.",
    icon: "fa-cloud-upload-alt",
  },
  {
    title: "Smart Model Selection",
    body: "Our meta-learning engine automatically selects the best ML algorithm for your specific data patterns.",
    icon: "fa-brain",
  },
  {
    title: "Instant Training",
    body: "Watch your models train in real-time with live progress updates. Get production-ready models in minutes.",
    icon: "fa-bolt",
  },
  {
    title: "Download & Deploy",
    body: "Get your trained models with all preprocessing pipelines packaged and ready for deployment.",
    icon: "fa-box",
  },
  {
    title: "Visual Insights",
    body: "Understand your model decisions with explainable AI features and feature importance visualizations.",
    icon: "fa-chart-line",
  },
  {
    title: "Secure & Private",
    body: "Your data stays yours. All processing happens in your secure workspace with enterprise-grade security.",
    icon: "fa-lock",
  },
];

const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
};

const Home = () => {
  const { profile } = useSession();
  return (
    <section className="page home">
      <div className="hero">
        <motion.p
          className="eyebrow"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          SmartML AutoML Platform
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
        >
          Machine Learning Made Simple for Everyone
        </motion.h1>
        <motion.p
          className="lead"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          {profile?.fname ? `Welcome back, ${profile.fname}! ` : ""}
          Transform your data into powerful predictions without writing a single
          line of code. SmartML automates the entire machine learning
          pipeline—from data preprocessing to model deployment—so you can focus
          on insights, not infrastructure.
        </motion.p>
        <motion.div
          className="hero-cta"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.35, duration: 0.5 }}
        >
          <NavLink to="/workspace" className="btn primary">
            {profile ? "Go to Workspace" : "Get Started Free"}
          </NavLink>
          <NavLink to={profile ? "/models" : "/auth"} className="btn ghost">
            {profile ? "View Models" : "Learn More"}
          </NavLink>
        </motion.div>
      </div>

      <motion.div
        className="stat-grid"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.6 }}
      >
        {stats.map((stat, index) => (
          <motion.article
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 + index * 0.1, duration: 0.4 }}
          >
            <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>
              <i className={`fas ${stat.icon}`}></i>
            </div>
            <strong>{stat.value}</strong>
            <span>{stat.label}</span>
          </motion.article>
        ))}
      </motion.div>

      <motion.div {...fadeInUp} transition={{ delay: 0.6, duration: 0.5 }}>
        <h2 style={{ textAlign: "center", marginBottom: "1rem" }}>
          Everything You Need in One Platform
        </h2>
        <p
          className="lead"
          style={{ textAlign: "center", marginBottom: "2rem" }}
        >
          From data upload to model deployment, we've automated the entire
          workflow
        </p>
      </motion.div>

      <motion.div
        className="grid"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7, duration: 0.6 }}
      >
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            className="card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.75 + index * 0.08, duration: 0.4 }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>
              <i className={`fas ${feature.icon}`}></i>
            </div>
            <h3>{feature.title}</h3>
            <p>{feature.body}</p>
          </motion.div>
        ))}
      </motion.div>

      <motion.div
        className="pipeline"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1, duration: 0.5 }}
      >
        <h2>How It Works</h2>
        <p>
          Our intelligent pipeline handles everything automatically, powered by
          advanced meta-learning algorithms:
        </p>
        <ul>
          <li>
            <strong>Upload Your Data:</strong> Support for CSV files with
            automatic column detection and validation.
          </li>
          <li>
            <strong>Automatic Preprocessing:</strong> Smart handling of missing
            values, outliers, and feature engineering.
          </li>
          <li>
            <strong>Meta-Learning Selection:</strong> Our AI analyzes your data
            characteristics and selects the optimal algorithm.
          </li>
          <li>
            <strong>Hyperparameter Optimization:</strong> Automated tuning for
            peak performance without manual intervention.
          </li>
          <li>
            <strong>Model Training & Validation:</strong> Robust
            cross-validation ensures your model generalizes well.
          </li>
          <li>
            <strong>Instant Deployment Package:</strong> Download ready-to-use
            models with preprocessing pipelines included.
          </li>
        </ul>
      </motion.div>

      <motion.div
        className="card"
        style={{
          textAlign: "center",
          padding: "3rem 2rem",
          background:
            "linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(34, 197, 94, 0.1))",
          border: "1px solid rgba(14, 165, 233, 0.3)",
        }}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1.1, duration: 0.5 }}
      >
        <h2 style={{ marginBottom: "1rem", textAlign: "center" }}>
          Ready to Transform Your Data?
        </h2>
        <p
          className="lead"
          style={{
            marginBottom: "2rem",
            textAlign: "center",
            maxWidth: "600px",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        >
          Join hundreds of users who are already leveraging the power of
          automated machine learning
        </p>
        <div
          style={{
            display: "flex",
            gap: "1rem",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <NavLink to="/workspace" className="btn primary">
            Start Building Now
          </NavLink>
          {!profile && (
            <NavLink to="/auth" className="btn ghost">
              Create Free Account
            </NavLink>
          )}
        </div>
      </motion.div>
    </section>
  );
};

export default Home;

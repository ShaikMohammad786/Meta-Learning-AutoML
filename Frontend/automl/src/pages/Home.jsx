import { motion } from 'framer-motion'
import { NavLink } from 'react-router-dom'
import { useSession } from '../context/SessionContext'

const stats = [
  { label: 'Pipelines automated', value: '300+' },
  { label: 'Datasets processed', value: '2.4M rows' },
  { label: 'Avg. tuning time', value: '<7 mins' },
]

const features = [
  {
    title: 'Opinionated FastAPI backend',
    body: 'Secure auth, dataset routing and model storage backed by MongoDB and JWT.',
  },
  {
    title: 'Hands-free AutoML',
    body: 'Upload a dataset, pick the task type and we orchestrate preprocessing, meta-learning and training.',
  },
  {
    title: 'Model governance',
    body: 'Downloadable model bundles, dataset catalogues and complete audit trails per user.',
  },
]

const Home = () => {
  const { profile } = useSession()
  return (
    <section className="page home">
      <div className="hero">
        <motion.p className="eyebrow" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          SmartML · FastAPI + React deployment
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          Production-grade AutoML workspace with a human touch.
        </motion.h1>
        <motion.p
          className="lead"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          Bridge your FastAPI orchestration and the React experience teams expect. Upload datasets, watch meta-models
          tune, manage artifacts, and ship models faster than ever.
        </motion.p>
        <motion.div className="hero-cta" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}>
          <NavLink to="/workspace" className="btn primary">
            Launch workspace
          </NavLink>
          <NavLink to="/auth" className="btn ghost">
            {profile ? 'Manage access' : 'Authenticate'}
          </NavLink>
        </motion.div>
      </div>

      <div className="stat-grid">
        {stats.map((stat) => (
          <article key={stat.label}>
            <strong>{stat.value}</strong>
            <span>{stat.label}</span>
          </article>
        ))}
      </div>

      <div className="grid">
        {features.map((feature) => (
          <div key={feature.title} className="card">
            <h3>{feature.title}</h3>
            <p>{feature.body}</p>
          </div>
        ))}
      </div>

      <div className="pipeline">
        <h2>Route-first architecture</h2>
        <p>
          Every interaction maps directly to FastAPI routes defined in <code>main.py</code>:
        </p>
        <ul>
          <li>
            <code>/users/register</code> + <code>/users/login</code> secure the workspace with JWT.
          </li>
          <li>
            <code>/users/send_dataset</code> streams files, target columns, and tuning preferences to the AutoML engine.
          </li>
          <li>
            <code>/users/get_models</code> & <code>/users/get_datasets</code> hydrate the catalogues instantly.
          </li>
          <li>
            Signed downloads keep datasets and models auditable.
          </li>
        </ul>
      </div>
    </section>
  )
}

export default Home


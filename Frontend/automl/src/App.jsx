import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Auth from './pages/Auth'
import Workspace from './pages/Workspace'
import Models from './pages/Models'
import { SessionProvider } from './context/SessionContext'
import './App.css'

function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/auth" element={<Auth />} />
            <Route path="/workspace" element={<Workspace />} />
            <Route path="/models" element={<Models />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </SessionProvider>
  )
}

export default App

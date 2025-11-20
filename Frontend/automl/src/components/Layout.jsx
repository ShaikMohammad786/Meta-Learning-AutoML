import Navigation from './Navigation'
import Footer from './Footer'

const Layout = ({ children }) => (
  <div className="app-shell">
    <div className="bg-grid" aria-hidden="true" />
    <Navigation />
    <main>{children}</main>
    <Footer />
  </div>
)

export default Layout


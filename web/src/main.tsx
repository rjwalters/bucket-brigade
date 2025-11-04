import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter as Router } from 'react-router-dom'
import App from './App.tsx'
import './index.css'
import { ThemeProvider } from './contexts/ThemeContext.tsx'

// Get base path from environment (defaults to /bucket-brigade/ for GitHub Pages)
const basename = import.meta.env.BASE_URL || '/bucket-brigade/';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <Router basename={basename}>
        <App />
      </Router>
    </ThemeProvider>
  </React.StrictMode>,
)

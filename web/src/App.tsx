import { Routes, Route, Navigate, useLocation, Link } from 'react-router-dom';
import { Flame, Play, FlaskConical, Trophy, Github, Sun, Moon } from 'lucide-react';
import SimpleDashboard from './pages/SimpleDashboard';
import GameReplay from './pages/GameReplay';
import ScenarioResearch from './pages/ScenarioResearch';
import Tournament from './pages/Tournament';
import { useTheme } from './contexts/ThemeContext';

function App() {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  // Map current path to active tab
  const getActiveTab = () => {
    const path = location.pathname;
    if (path === '/' || path === '/dashboard') return 'dashboard';
    if (path.startsWith('/replay')) return 'replay';
    if (path === '/research') return 'research';
    if (path === '/tournament') return 'tournament';
    return 'dashboard';
  };

  const activeTab = getActiveTab();

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Flame, path: '/' },
    { id: 'replay', label: 'Game Replay', icon: Play, path: '/replay' },
    { id: 'tournament', label: 'Tournament', icon: Trophy, path: '/tournament' },
    { id: 'research', label: 'Research', icon: FlaskConical, path: '/research' }
  ];

  return (
    <div className="min-h-screen bg-surface-primary">
      {/* Header */}
      <header className="bg-surface-secondary shadow-sm border-b border-outline-primary">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-8 h-8 bg-brand-bg rounded-lg">
                <Flame className="w-5 h-5 text-brand-text" />
              </div>
              <h1 className="text-xl font-bold text-content-primary">Bucket Brigade</h1>
              <span className="text-sm text-content-secondary">Multi-Agent Cooperation Tournament</span>
            </div>
            <div className="flex items-center space-x-4">
              <nav className="flex space-x-1">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <Link
                      key={tab.id}
                      to={tab.path}
                      data-testid={`nav-${tab.id}`}
                      className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        activeTab === tab.id
                          ? 'bg-interactive-active text-blue-700 dark:text-blue-300'
                          : 'text-content-secondary hover:text-content-primary hover:bg-interactive-hover'
                      }`}
                    >
                      <Icon className="w-4 h-4 mr-2" />
                      {tab.label}
                    </Link>
                  );
                })}
              </nav>
              <a
                href="https://github.com/rjwalters/bucket-brigade"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center px-3 py-2 text-content-secondary hover:text-content-primary hover:bg-interactive-hover rounded-md transition-colors"
                title="View on GitHub"
              >
                <Github className="w-5 h-5" />
              </a>
              <button
                onClick={toggleTheme}
                className="flex items-center px-3 py-2 text-content-secondary hover:text-content-primary hover:bg-interactive-hover rounded-md transition-colors"
                title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
              >
                {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<SimpleDashboard />} />
          <Route path="/dashboard" element={<Navigate to="/" replace />} />
          <Route path="/tournament" element={<Tournament />} />
          <Route path="/research" element={<ScenarioResearch />} />
          <Route path="/replay/:gameId?" element={<GameReplay />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;

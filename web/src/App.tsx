import { Routes, Route, Navigate, useLocation, Link } from 'react-router-dom';
import { Flame, Play, Settings } from 'lucide-react';
import SimpleDashboard from './pages/SimpleDashboard';
import SettingsPage from './pages/Settings';
import GameReplay from './pages/GameReplay';

function App() {
  const location = useLocation();

  // Map current path to active tab
  const getActiveTab = () => {
    const path = location.pathname;
    if (path === '/' || path === '/dashboard') return 'dashboard';
    if (path.startsWith('/replay')) return 'replay';
    if (path === '/settings') return 'settings';
    return 'dashboard';
  };

  const activeTab = getActiveTab();

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Flame, path: '/' },
    { id: 'replay', label: 'Game Replay', icon: Play, path: '/replay' },
    { id: 'settings', label: 'Settings', icon: Settings, path: '/settings' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-8 h-8 bg-orange-100 rounded-lg">
                <Flame className="w-5 h-5 text-orange-600" />
              </div>
              <h1 className="text-xl font-bold text-gray-900">Bucket Brigade</h1>
              <span className="text-sm text-gray-500">Multi-Agent Cooperation Visualizer</span>
            </div>
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
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    {tab.label}
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<SimpleDashboard />} />
          <Route path="/dashboard" element={<Navigate to="/" replace />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/replay/:gameId?" element={<GameReplay />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;

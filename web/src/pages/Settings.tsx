import React, { useState, useRef } from 'react';
import { Upload, Download, Trash2, Database } from 'lucide-react';
import { GameReplay, BatchResult, STORAGE_KEYS } from '../types';
import { loadFromStorage, saveToStorage, clearStorage } from '../utils/storage';

const Settings: React.FC = () => {
  const [gameCount, setGameCount] = useState(0);
  const [resultCount, setResultCount] = useState(0);
  const gameFileRef = useRef<HTMLInputElement>(null);
  const resultFileRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    updateCounts();
  }, []);

  const updateCounts = () => {
    const games = loadFromStorage<GameReplay[]>(STORAGE_KEYS.GAME_REPLAYS, []);
    const results = loadFromStorage<BatchResult[]>(STORAGE_KEYS.BATCH_RESULTS, []);
    setGameCount(games.length);
    setResultCount(results.length);
  };

  const handleGameFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const gameData: GameReplay = JSON.parse(content);

        // Validate the data structure
        if (!gameData.scenario || !gameData.nights) {
          throw new Error('Invalid game replay format');
        }

        const existingGames = loadFromStorage<GameReplay[]>(STORAGE_KEYS.GAME_REPLAYS, []);
        existingGames.push(gameData);
        saveToStorage(STORAGE_KEYS.GAME_REPLAYS, existingGames);

        updateCounts();
        alert(`Successfully loaded game replay: ${gameData.nights.length} nights`);
      } catch (error) {
        alert(`Error loading game file: ${error}`);
      }
    };
    reader.readAsText(file);
  };

  const handleResultFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const lines = content.trim().split('\n');
        const headers = lines[0].split(',');
        const results: BatchResult[] = [];

        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',');
          if (values.length >= headers.length) {
            const result: any = {};
            headers.forEach((header, index) => {
              const value = values[index];
              if (header === 'team' || header === 'agent_rewards' || header === 'agent_params') {
                result[header] = JSON.parse(value);
              } else if (['game_id', 'scenario_id', 'nights_played', 'saved_houses', 'ruined_houses'].includes(header)) {
                result[header] = parseInt(value);
              } else if (header === 'team_reward') {
                result[header] = parseFloat(value);
              } else {
                result[header] = value;
              }
            });
            results.push(result as BatchResult);
          }
        }

        const existingResults = loadFromStorage<BatchResult[]>(STORAGE_KEYS.BATCH_RESULTS, []);
        existingResults.push(...results);
        saveToStorage(STORAGE_KEYS.BATCH_RESULTS, existingResults);

        updateCounts();
        alert(`Successfully loaded ${results.length} batch results`);
      } catch (error) {
        alert(`Error loading results file: ${error}`);
      }
    };
    reader.readAsText(file);
  };

  const exportData = () => {
    const games = loadFromStorage<GameReplay[]>(STORAGE_KEYS.GAME_REPLAYS, []);
    const results = loadFromStorage<BatchResult[]>(STORAGE_KEYS.BATCH_RESULTS, []);

    const exportData = {
      games,
      results,
      exported_at: new Date().toISOString(),
      version: '1.0'
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bucket-brigade-data-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearAllData = () => {
    if (window.confirm('Are you sure you want to clear all stored data? This cannot be undone.')) {
      clearStorage();
      updateCounts();
      alert('All data cleared');
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Settings</h1>
        <p className="text-lg text-gray-600">
          Manage game data and application settings
        </p>
      </div>

      {/* Data Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <Database className="w-8 h-8 mx-auto mb-2 text-blue-600" />
          <div className="text-2xl font-bold text-gray-900">{gameCount}</div>
          <div className="text-sm text-gray-600">Game Replays</div>
        </div>
        <div className="card text-center">
          <Database className="w-8 h-8 mx-auto mb-2 text-green-600" />
          <div className="text-2xl font-bold text-gray-900">{resultCount}</div>
          <div className="text-sm text-gray-600">Batch Results</div>
        </div>
        <div className="card text-center">
          <Database className="w-8 h-8 mx-auto mb-2 text-purple-600" />
          <div className="text-2xl font-bold text-gray-900">
            {(JSON.stringify(loadFromStorage(STORAGE_KEYS.GAME_REPLAYS, [])).length +
              JSON.stringify(loadFromStorage(STORAGE_KEYS.BATCH_RESULTS, [])).length) / 1024).toFixed(1)} KB
          </div>
          <div className="text-sm text-gray-600">Storage Used</div>
        </div>
      </div>

      {/* Data Import */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Import Data</h2>
        <div className="space-y-4">
          {/* Game Replays */}
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div>
              <h3 className="font-medium text-gray-900">Game Replays</h3>
              <p className="text-sm text-gray-600">
                Upload JSON files from <code className="bg-gray-100 px-1 rounded">replays/</code> directory
              </p>
            </div>
            <div>
              <input
                ref={gameFileRef}
                type="file"
                accept=".json"
                onChange={handleGameFileUpload}
                className="hidden"
              />
              <button
                onClick={() => gameFileRef.current?.click()}
                className="btn-primary"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload JSON
              </button>
            </div>
          </div>

          {/* Batch Results */}
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div>
              <h3 className="font-medium text-gray-900">Batch Results</h3>
              <p className="text-sm text-gray-600">
                Upload CSV files from <code className="bg-gray-100 px-1 rounded">results/batch_results.csv</code>
              </p>
            </div>
            <div>
              <input
                ref={resultFileRef}
                type="file"
                accept=".csv"
                onChange={handleResultFileUpload}
                className="hidden"
              />
              <button
                onClick={() => resultFileRef.current?.click()}
                className="btn-primary"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload CSV
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Data Export & Management */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Data Management</h2>
        <div className="space-y-4">
          {/* Export */}
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div>
              <h3 className="font-medium text-gray-900">Export All Data</h3>
              <p className="text-sm text-gray-600">
                Download all stored data as a backup JSON file
              </p>
            </div>
            <button
              onClick={exportData}
              className="btn-secondary"
            >
              <Download className="w-4 h-4 mr-2" />
              Export
            </button>
          </div>

          {/* Clear Data */}
          <div className="flex items-center justify-between p-4 border border-red-200 bg-red-50 rounded-lg">
            <div>
              <h3 className="font-medium text-red-900">Clear All Data</h3>
              <p className="text-sm text-red-600">
                Permanently delete all stored game data and results
              </p>
            </div>
            <button
              onClick={clearAllData}
              className="btn-primary bg-red-600 hover:bg-red-700"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear All
            </button>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="card bg-blue-50 border-blue-200">
        <h2 className="text-xl font-semibold text-blue-900 mb-4">How to Get Data</h2>
        <div className="space-y-3 text-blue-800">
          <div className="flex items-start space-x-3">
            <span className="font-bold">1.</span>
            <span>Run batch experiments: <code className="bg-blue-100 px-1 rounded">python scripts/run_batch.py --num-games 50</code></span>
          </div>
          <div className="flex items-start space-x-3">
            <span className="font-bold">2.</span>
            <span>Upload the generated <code className="bg-blue-100 px-1 rounded">replays/*.json</code> files using the buttons above</span>
          </div>
          <div className="flex items-start space-x-3">
            <span className="font-bold">3.</span>
            <span>Upload <code className="bg-blue-100 px-1 rounded">results/batch_results.csv</code> for ranking data</span>
          </div>
          <div className="flex items-start space-x-3">
            <span className="font-bold">4.</span>
            <span>Explore games in the Replay tab and view rankings in the Rankings tab</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;

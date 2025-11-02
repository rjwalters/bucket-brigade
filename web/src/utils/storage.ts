import {
  validateGameReplay,
  validateBatchResult,
  safeValidateGameReplay,
  safeValidateBatchResult,
  safeValidateExportData,
  GameReplay,
  BatchResult,
  ExportData
} from './schemas';

// Session storage utilities for Bucket Brigade data management

export function saveToStorage<T>(key: string, data: T): void {
  try {
    const serialized = JSON.stringify(data);
    sessionStorage.setItem(key, serialized);
  } catch (error) {
    console.error(`Failed to save data to sessionStorage for key ${key}:`, error);
  }
}

export function loadFromStorage<T>(key: string, defaultValue: T): T {
  try {
    const item = sessionStorage.getItem(key);
    if (item === null) {
      return defaultValue;
    }
    return JSON.parse(item) as T;
  } catch (error) {
    console.error(`Failed to load data from sessionStorage for key ${key}:`, error);
    return defaultValue;
  }
}

// Validated storage functions for critical data types

export function saveGameReplays(games: GameReplay[]): void {
  // Validate all games before saving
  const validatedGames = games.map(game => validateGameReplay(game));
  saveToStorage('bucket_brigade_replays', validatedGames);
}

export function loadGameReplays(): GameReplay[] {
  const rawData = loadFromStorage<GameReplay[]>('bucket_brigade_replays', []);
  // Validate and filter out invalid data
  return rawData.map(game => safeValidateGameReplay(game)).filter(Boolean) as GameReplay[];
}

export function saveBatchResults(results: BatchResult[]): void {
  // Validate all results before saving
  const validatedResults = results.map(result => validateBatchResult(result));
  saveToStorage('bucket_brigade_results', validatedResults);
}

export function loadBatchResults(): BatchResult[] {
  const rawData = loadFromStorage<BatchResult[]>('bucket_brigade_results', []);
  // Validate and filter out invalid data
  return rawData.map(result => safeValidateBatchResult(result)).filter(Boolean) as BatchResult[];
}

export function saveExportData(data: ExportData): void {
  // Validate export data
  const validatedData = safeValidateExportData(data);
  if (validatedData) {
    saveToStorage('bucket_brigade_export', validatedData);
  } else {
    throw new Error('Invalid export data format');
  }
}

export function loadExportData(): ExportData | null {
  const rawData = loadFromStorage<ExportData | null>('bucket_brigade_export', null);
  return rawData ? safeValidateExportData(rawData) : null;
}

export function removeFromStorage(key: string): void {
  try {
    sessionStorage.removeItem(key);
  } catch (error) {
    console.error(`Failed to remove data from sessionStorage for key ${key}:`, error);
  }
}

export function clearStorage(): void {
  try {
    sessionStorage.clear();
  } catch (error) {
    console.error('Failed to clear sessionStorage:', error);
  }
}

export function getStorageSize(): number {
  try {
    let total = 0;
    for (let key in sessionStorage) {
      if (sessionStorage.hasOwnProperty(key)) {
        total += sessionStorage[key].length + key.length;
      }
    }
    return total;
  } catch (error) {
    console.error('Failed to calculate storage size:', error);
    return 0;
  }
}

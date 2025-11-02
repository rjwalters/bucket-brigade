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

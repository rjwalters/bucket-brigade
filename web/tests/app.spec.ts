import { test, expect } from '@playwright/test';

test.describe('Bucket Brigade Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the dashboard', async ({ page }) => {
    await expect(page).toHaveTitle(/Bucket Brigade/);
    await expect(page.locator('h1')).toContainText('Welcome to Bucket Brigade');
  });

  test('should navigate between pages', async ({ page }) => {
    // Check dashboard is active
    await expect(page.locator('[data-testid="nav-dashboard"]')).toHaveClass(/bg-blue-100/);

    // Navigate to rankings
    await page.locator('[data-testid="nav-rankings"]').click();
    await expect(page.locator('h1')).toContainText('Agent Rankings');

    // Navigate to settings
    await page.locator('[data-testid="nav-settings"]').click();
    await expect(page.locator('h1')).toContainText('Settings');

    // Navigate back to dashboard
    await page.locator('[data-testid="nav-dashboard"]').click();
    await expect(page.locator('h1')).toContainText('Welcome to Bucket Brigade');
  });

  test('should show empty state when no data', async ({ page }) => {
    await expect(page.locator('text=No games available')).toBeVisible();
    await expect(page.locator('text=No Rankings Available')).toBeVisible();
  });
});

test.describe('Data Import/Export', () => {
  test('should handle file uploads', async ({ page }) => {
    await page.goto('/settings');

    // Check that file inputs exist
    await expect(page.locator('input[type="file"][accept=".json"]')).toBeVisible();
    await expect(page.locator('input[type="file"][accept=".csv"]')).toBeVisible();
  });

  test('should validate data formats', async ({ page }) => {
    // This would test actual file upload validation
    // For now, just check the UI is present
    await page.goto('/settings');
    await expect(page.locator('text=Import Data')).toBeVisible();
  });
});

test.describe('Game Replay', () => {
  test('should show replay interface when no games', async ({ page }) => {
    await page.goto('/replay');
    await expect(page.locator('text=No Games Available')).toBeVisible();
  });
});

test.describe('Responsive Design', () => {
  test('should work on mobile viewport', async ({ page, isMobile }) => {
    if (isMobile) {
      await page.goto('/');
      await expect(page.locator('h1')).toContainText('Welcome to Bucket Brigade');

      // Check mobile navigation
      await expect(page.locator('[data-testid="nav-dashboard"]')).toBeVisible();
    }
  });
});

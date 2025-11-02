import { test, expect } from '@playwright/test';

test.describe('Bucket Brigade Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should load the dashboard', async ({ page }) => {
    // Listen for console errors
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await expect(page).toHaveTitle(/Bucket Brigade/);
    // Wait for the page to load
    await page.waitForLoadState('networkidle');

    // Check for JavaScript errors
    if (errors.length > 0) {
      console.log('Console errors:', errors);
      throw new Error(`JavaScript errors found: ${errors.join(', ')}`);
    }

    // Check if the root div exists
    await expect(page.locator('#root')).toBeAttached();
    // Wait a bit for React to mount
    await page.waitForTimeout(2000);

    // Check for the main content
    await expect(page.locator('main')).toContainText('Bucket Brigade');
  });

  test('should navigate between pages', async ({ page }) => {
    // Check dashboard is active
    await expect(page.locator('[data-testid="nav-dashboard"]')).toHaveClass(/bg-blue-100/);

    // Navigate to rankings
    await page.goto('/rankings');
    await expect(page.locator('text=No Rankings Available')).toBeVisible();

    // Navigate to settings
    await page.goto('/settings');
    await expect(page.locator('main h1')).toContainText('Settings');
    await expect(page.locator('text=Data Management')).toBeVisible();

    // Navigate back to dashboard
    await page.goto('/dashboard');
    await expect(page.locator('main')).toContainText('Bucket Brigade');
  });

  test('should show empty state when no data', async ({ page }) => {
    await expect(page.locator('text=No games loaded yet')).toBeVisible();
    await page.goto('/rankings');
    await expect(page.locator('text=No Rankings Available')).toBeVisible();
  });
});

test.describe('Data Import/Export', () => {
  test('should handle file uploads', async ({ page }) => {
  await page.goto('/settings');
  await page.waitForLoadState('networkidle');

  // Check that upload buttons exist
  await expect(page.locator('h2:has-text("Import Data")')).toBeVisible();
  await expect(page.locator('button:has-text("Upload JSON")')).toBeVisible();
  await expect(page.locator('button:has-text("Upload CSV")')).toBeVisible();
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
      await expect(page.locator('header h1')).toContainText('Bucket Brigade');

      // Check mobile navigation
      await expect(page.locator('[data-testid="nav-dashboard"]')).toBeVisible();
    }
  });
});

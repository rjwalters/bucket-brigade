import { test, expect } from '@playwright/test';

test.describe('Research Tab - Basic Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/research');
    await page.waitForLoadState('networkidle');
  });

  test('should load research page successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Bucket Brigade/);
    await expect(page.locator('main').locator('h1')).toContainText('Scenario Research');
  });

  test('should display all three tabs', async ({ page }) => {
    await expect(page.locator('button', { hasText: 'Nash Equilibrium' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'Evolution' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'Heuristics' })).toBeVisible();
  });

  test('should load Nash equilibrium data', async ({ page }) => {
    // Click Nash tab
    await page.locator('button', { hasText: 'Nash Equilibrium' }).click();
    await page.waitForTimeout(1500);

    // Check for Nash-specific content
    const mainContent = page.locator('main');
    await expect(mainContent.locator('text=Nash Equilibrium Analysis')).toBeVisible({ timeout: 10000 });
    await expect(mainContent.locator('text=Equilibrium Type').first()).toBeVisible();
    await expect(mainContent.locator('text=Expected Payoff').first()).toBeVisible();
  });

  test('should display Nash equilibrium strategy details', async ({ page }) => {
    // Click Nash tab
    await page.locator('button', { hasText: 'Nash Equilibrium' }).click();
    await page.waitForTimeout(1500);

    // Check for strategy pool
    await expect(page.locator('h3', { hasText: 'Equilibrium Strategies' })).toBeVisible({ timeout: 10000 });

    // Check for agent parameters
    const mainContent = page.locator('main');
    const parameterNames = ['honesty', 'work tendency', 'coordination', 'altruism'];

    for (const param of parameterNames) {
      await expect(mainContent.locator(`text=${param}`).first()).toBeVisible();
    }
  });

  test('should load Evolution data', async ({ page }) => {
    // Click Evolution tab
    await page.locator('button', { hasText: 'Evolution' }).click();
    await page.waitForTimeout(1500);

    // Check for evolution-specific content
    await expect(page.locator('h2', { hasText: 'Evolutionary Optimization' })).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Best Fitness').first()).toBeVisible();
  });

  test('should load Heuristics data', async ({ page }) => {
    // Click Heuristics tab
    await page.locator('button', { hasText: 'Heuristics' }).click();
    await page.waitForTimeout(2000);

    // Check that main content area exists and has some content
    const mainContent = page.locator('main');
    await expect(mainContent).toBeVisible();

    // The heuristics tab should show either heuristics or comparison data
    // Let's just verify the page didn't error out and content is present
    const text = await mainContent.textContent();
    const hasValidContent = text && text.length > 100; // Should have substantial content

    expect(hasValidContent).toBeTruthy();
  });

  test('should switch between scenarios', async ({ page }) => {
    const selector = page.locator('select');

    // Switch to a different scenario
    await selector.selectOption('deceptive_calm');
    await page.waitForTimeout(1000);

    // Click Nash tab
    await page.locator('button', { hasText: 'Nash Equilibrium' }).click();
    await page.waitForTimeout(1500);

    // Should show content
    await expect(page.locator('text=Nash Equilibrium Analysis')).toBeVisible({ timeout: 10000 });
  });

  test('should display cooperation rate in Nash tab', async ({ page }) => {
    await page.locator('button', { hasText: 'Nash Equilibrium' }).click();
    await page.waitForTimeout(1500);

    // Check for cooperation rate metric card
    await expect(page.locator('text=Cooperation Rate').first()).toBeVisible({ timeout: 10000 });

    // The page should show cooperation and interpretation data
    const mainContent = page.locator('main');
    const hasCooperationData = await mainContent.locator('text=cooperation_rate').isVisible().catch(() =>
      mainContent.textContent().then(text => text?.includes('Cooperation') || false)
    );

    expect(hasCooperationData).toBeTruthy();
  });

  test('should render fitness chart in Evolution tab', async ({ page }) => {
    await page.locator('button', { hasText: 'Evolution' }).click();
    await page.waitForTimeout(1500);

    // Check for chart
    await expect(page.locator('h4', { hasText: 'Fitness Over Generations' })).toBeVisible({ timeout: 10000 });
    await expect(page.locator('svg').first()).toBeVisible();
  });
});

test.describe('Research Tab - Performance', () => {
  test('should load page quickly', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/research');
    await page.waitForLoadState('networkidle');

    const loadTime = Date.now() - startTime;

    // Should load in under 10 seconds (generous for CI)
    expect(loadTime).toBeLessThan(10000);

    // Verify content loaded
    await expect(page.locator('main').locator('h1')).toContainText('Scenario Research');
  });
});

import { test, expect } from '@playwright/test';

/**
 * Basic smoke tests for the Scenario Research page.
 *
 * As of PR #140 / commit fe73a20a, the research page renders all sections
 * vertically (no tabs). These tests assert the page loads and the major
 * sections render correctly when data is available.
 *
 * NOTE: Sections (ComparisonSection, HeuristicsSection, EvolutionSection,
 * NashSection) render conditionally based on whether the selected scenario
 * has data for that method. In environments where only config.json is
 * present (e.g. local dev without generated research artifacts), those
 * sections won't render. Tests that depend on per-section data are skipped
 * gracefully when the data is absent.
 */
test.describe('Research Page - Basic Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/research');
    await page.waitForLoadState('networkidle');
  });

  test('should load research page successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Bucket Brigade/);
    await expect(page.locator('main').locator('h1')).toContainText('Scenario Research');
  });

  test('should display scenario selector', async ({ page }) => {
    await expect(page.locator('select')).toBeVisible();
    await expect(page.locator('label:has-text("Select Scenario")')).toBeVisible();
  });

  test('should display scenario overview parameters', async ({ page }) => {
    // ScenarioOverview renders a grid of parameter labels whenever config is loaded.
    const main = page.locator('main');
    await expect(main.locator('text=Fire Spread (β)').first()).toBeVisible();
    await expect(main.locator('text=Work Cost (c)').first()).toBeVisible();
    await expect(main.locator('text=Agents').first()).toBeVisible();
  });

  test('should display comparison section when data is available', async ({ page }) => {
    const main = page.locator('main');
    const heading = main.locator('h2:has-text("Strategy Comparison")');
    if ((await heading.count()) === 0) {
      test.skip(true, 'Comparison data not available for default scenario');
    }
    await expect(heading).toBeVisible();
    await expect(main.locator('h3:has-text("Tournament Results")')).toBeVisible();
    await expect(main.locator('text=#1').first()).toBeVisible();
  });

  test('should display Nash equilibrium section when data is available', async ({ page }) => {
    const main = page.locator('main');
    const heading = main.locator('h2:has-text("Nash Equilibrium Analysis")');
    if ((await heading.count()) === 0) {
      test.skip(true, 'Nash data not available for default scenario');
    }
    await expect(heading).toBeVisible();
    await expect(main.locator('text=Equilibrium Type').first()).toBeVisible();
    await expect(main.locator('text=Expected Payoff').first()).toBeVisible();
    await expect(main.locator('text=Cooperation Rate').first()).toBeVisible();
  });

  test('should display Nash equilibrium strategy details when data is available', async ({
    page,
  }) => {
    const main = page.locator('main');
    const heading = main.locator('h3:has-text("Equilibrium Strategies")');
    if ((await heading.count()) === 0) {
      test.skip(true, 'Nash data not available for default scenario');
    }
    await expect(heading).toBeVisible();

    // Check for at least some agent parameter labels (rendered with underscores
    // replaced by spaces).
    const parameterNames = ['honesty', 'work tendency', 'coordination'];
    for (const param of parameterNames) {
      await expect(main.locator(`text=${param}`).first()).toBeVisible();
    }
  });

  test('should display Evolution section with fitness chart when data is available', async ({
    page,
  }) => {
    const main = page.locator('main');
    const heading = main.locator('h2:has-text("Evolutionary Optimization")');
    if ((await heading.count()) === 0) {
      test.skip(true, 'Evolution data not available for default scenario');
    }
    await expect(heading).toBeVisible();
    await expect(main.locator('text=Best Fitness').first()).toBeVisible();
    await expect(main.locator('h4:has-text("Fitness Over Generations")')).toBeVisible();
    await expect(main.locator('svg').first()).toBeVisible();
  });

  test('should switch between scenarios', async ({ page }) => {
    const selector = page.locator('select');

    // Switch to a different scenario
    await selector.selectOption('deceptive_calm');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);

    // After switching, the scenario overview heading should still be present
    await expect(page.locator('main h2').first()).toBeVisible();

    // The h1 title remains
    await expect(page.locator('main').locator('h1')).toContainText('Scenario Research');
  });
});

test.describe('Research Page - Performance', () => {
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

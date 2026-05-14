import { test, expect } from '@playwright/test';

/**
 * Comprehensive tests for the Scenario Research page.
 *
 * As of PR #140 / commit fe73a20a, the research page renders all sections
 * vertically (ComparisonSection, HeuristicsSection, EvolutionSection,
 * NashSection, ResearchInsightsSection) without tabs.
 *
 * Sections render conditionally based on whether the selected scenario
 * has data for that method (Nash, Evolution, Comparison, Heuristics).
 * In environments where only config.json is present (no generated research
 * artifacts), per-section tests are skipped gracefully.
 */
test.describe('Research Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/research');
    await page.waitForLoadState('networkidle');
  });

  test('should load the research page', async ({ page }) => {
    // Check page title and header
    await expect(page).toHaveTitle(/Bucket Brigade/);
    await expect(page.locator('main h1')).toContainText('Scenario Research');

    // Check scenario selector is present
    await expect(page.locator('select')).toBeVisible();
    await expect(page.locator('label:has-text("Select Scenario")')).toBeVisible();
  });

  test('should display scenario overview', async ({ page }) => {
    // Wait for config to load
    await page.waitForTimeout(1000);

    // Check for scenario parameters (use more specific selectors)
    await expect(page.locator('div:has-text("Fire Spread (β)")').first()).toBeVisible();
    await expect(page.locator('div:has-text("Extinguish (κ)")').first()).toBeVisible();
    await expect(page.locator('div:has-text("Work Cost (c)")').first()).toBeVisible();
    await expect(page.locator('div:has-text("Agents")').first()).toBeVisible();
  });

  test('should switch scenarios', async ({ page }) => {
    const selector = page.locator('select');

    // Change to a different scenario
    await selector.selectOption('chain_reaction');
    await page.waitForLoadState('networkidle');

    // Verify the scenario changed by checking the overview heading is visible
    await expect(page.locator('main h2').first()).toBeVisible();

    // Change to another scenario
    await selector.selectOption('deceptive_calm');
    await page.waitForLoadState('networkidle');
    await expect(page.locator('main h2').first()).toBeVisible();
  });

  test.describe('Nash Equilibrium Section', () => {
    test('should display Nash equilibrium metrics when data is available', async ({
      page,
    }) => {
      const main = page.locator('main');
      const heading = main.locator('h2:has-text("Nash Equilibrium Analysis")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Nash data not available');
      }
      await expect(heading).toBeVisible();

      // Check for key metrics
      await expect(main.locator('text=Equilibrium Type')).toBeVisible();
      await expect(main.locator('text=Expected Payoff')).toBeVisible();
      await expect(main.locator('text=Cooperation Rate')).toBeVisible();
      await expect(main.locator('text=Convergence Time')).toBeVisible();
    });

    test('should display equilibrium strategies when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h3:has-text("Equilibrium Strategies")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Nash data not available');
      }
      await expect(heading).toBeVisible();

      // Check for strategy classification heading inside the Nash section
      await expect(main.locator('h4').first()).toBeVisible();

      // Check for agent parameter labels
      await expect(main.locator('text=honesty').first()).toBeVisible();
      await expect(main.locator('text=work tendency').first()).toBeVisible();
      await expect(main.locator('text=coordination').first()).toBeVisible();
    });

    test('should display computation details when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h3:has-text("Computation Details")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Nash data not available');
      }
      await expect(heading).toBeVisible();
      await expect(main.locator('text=Algorithm')).toBeVisible();
      await expect(main.locator('text=Simulations')).toBeVisible();
      await expect(main.locator('text=Iterations')).toBeVisible();
      await expect(main.locator('text=Converged')).toBeVisible();
    });

    test('should show cooperation rate percentage when data is available', async ({ page }) => {
      const main = page.locator('main');
      const nashHeading = main.locator('h2:has-text("Nash Equilibrium Analysis")');
      if ((await nashHeading.count()) === 0) {
        test.skip(true, 'Nash data not available');
      }
      await expect(nashHeading).toBeVisible();

      // Look for cooperation rate metric card containing a percentage
      const cooperationCard = main.locator('text=Cooperation Rate').locator('..').first();
      await expect(cooperationCard).toContainText('%');
    });

    test('should display strategy parameters with visual bars when data is available', async ({
      page,
    }) => {
      const main = page.locator('main');
      const nashHeading = main.locator('h2:has-text("Nash Equilibrium Analysis")');
      if ((await nashHeading.count()) === 0) {
        test.skip(true, 'Nash data not available');
      }
      await expect(nashHeading).toBeVisible();

      // Check that parameter bars are rendered
      const parameterBars = main.locator('.rounded-full.h-2');
      await expect(parameterBars.first()).toBeVisible();

      // Should be many parameter bars across all strategies
      const count = await parameterBars.count();
      expect(count).toBeGreaterThan(5);
    });
  });

  test.describe('Evolution Section', () => {
    test('should display evolution metrics when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h2:has-text("Evolutionary Optimization")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Evolution data not available');
      }
      await expect(heading).toBeVisible();
      await expect(main.locator('text=Best Fitness').first()).toBeVisible();
      await expect(main.locator('text=Generation').first()).toBeVisible();
      await expect(main.locator('text=Time').first()).toBeVisible();
    });

    test('should display fitness chart when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h4:has-text("Fitness Over Generations")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Evolution data not available');
      }
      await expect(heading).toBeVisible();

      // Check for SVG chart with polyline data
      await expect(main.locator('svg polyline').first()).toBeVisible();

      // Check for legend labels (both Best Fitness and Mean Fitness)
      await expect(main.locator('text=Best Fitness').first()).toBeVisible();
      await expect(main.locator('text=Mean Fitness').first()).toBeVisible();
    });

    test('should display best agent parameters when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h4:has-text("Best Agent Parameters")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Evolution data not available');
      }
      await expect(heading).toBeVisible();

      // Check for parameter bars
      const parameterBars = main.locator('.rounded-full.h-2');
      await expect(parameterBars.first()).toBeVisible();
    });
  });

  test.describe('Comparison Section', () => {
    test('should show strategy comparison and tournament results when data is available', async ({
      page,
    }) => {
      const main = page.locator('main');
      const heading = main.locator('h2:has-text("Strategy Comparison")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Comparison data not available');
      }
      await expect(heading).toBeVisible();
      await expect(main.locator('h3:has-text("Tournament Results")')).toBeVisible();
      await expect(main.locator('h3:has-text("Strategy Profiles")')).toBeVisible();

      // Check for ranking numbers
      await expect(main.locator('text=#1').first()).toBeVisible();
    });
  });

  test.describe('Heuristics Section', () => {
    test('should display heuristic archetypes when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h2:has-text("Heuristic Archetypes")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Heuristics data not available');
      }
      await expect(heading).toBeVisible();

      // At least one of these subsection headings should be present
      const homogeneous = main.locator('h3:has-text("Homogeneous Teams")');
      const mixed = main.locator('h3:has-text("Mixed Teams")');
      const hasHomogeneous = (await homogeneous.count()) > 0;
      const hasMixed = (await mixed.count()) > 0;
      expect(hasHomogeneous || hasMixed).toBeTruthy();
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page, isMobile }) => {
      if (isMobile) {
        // Check that the page is accessible
        await expect(page.locator('main h1')).toContainText('Scenario Research');

        // Check that scenario selector is visible
        await expect(page.locator('select')).toBeVisible();

        // Check that at least one section heading renders (ScenarioOverview h2)
        await expect(page.locator('main h2').first()).toBeVisible();
      }
    });

    test('should handle landscape mobile', async ({ page, isMobile }) => {
      if (isMobile) {
        // Rotate to landscape
        await page.setViewportSize({ width: 812, height: 375 });

        // Check layout still works
        await expect(page.locator('main h1')).toContainText('Scenario Research');
        await expect(page.locator('select')).toBeVisible();
      }
    });
  });

  test.describe('Navigation', () => {
    test('should navigate from research to other tabs', async ({ page }) => {
      // Check we're on research nav
      const researchNav = page.locator('[data-testid="nav-research"]');
      await expect(researchNav).toHaveClass(/bg-blue-100|bg-interactive-active/);

      // Navigate to dashboard
      await page.locator('[data-testid="nav-dashboard"]').click();
      await page.waitForLoadState('networkidle');
      await expect(page.locator('main')).toContainText('Bucket Brigade');

      // Navigate back to research
      await page.locator('[data-testid="nav-research"]').click();
      await page.waitForLoadState('networkidle');
      await expect(page.locator('main h1')).toContainText('Scenario Research');
    });

    test('should preserve state when navigating away and back', async ({ page }) => {
      // Select a specific scenario
      await page.locator('select').selectOption('rest_trap');
      await page.waitForTimeout(500);

      // Navigate away
      await page.locator('[data-testid="nav-dashboard"]').click();
      await page.waitForLoadState('networkidle');

      // Navigate back
      await page.locator('[data-testid="nav-research"]').click();
      await page.waitForLoadState('networkidle');

      // Check that page state is sane (header + selector)
      await expect(page.locator('main h1')).toContainText('Scenario Research');
      await expect(page.locator('select')).toBeVisible();
    });
  });

  test.describe('Data Loading', () => {
    test('should handle loading states gracefully', async ({ page }) => {
      // Clear cache and reload
      await page.reload({ waitUntil: 'domcontentloaded' });

      // Should show loading or content (not error)
      await page.waitForTimeout(2000);

      // Should eventually show content
      await expect(page.locator('main h1')).toContainText('Scenario Research');
    });

    test('should load data for multiple scenarios', async ({ page }) => {
      const scenarios = ['greedy_neighbor', 'trivial_cooperation', 'sparse_heroics'];

      for (const scenario of scenarios) {
        await page.locator('select').selectOption(scenario);
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(500);

        // Should show scenario overview heading
        await expect(page.locator('main h2').first()).toBeVisible();
      }
    });
  });

  test.describe('Visual Regression', () => {
    test('should render Nash equilibrium section correctly when data is available', async ({
      page,
    }) => {
      const main = page.locator('main');
      const heading = main.locator('h2:has-text("Nash Equilibrium Analysis")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Nash data not available');
      }
      await expect(heading).toBeVisible();

      // Check key visual elements
      await expect(main.locator('text=Equilibrium Type')).toBeVisible();
      await expect(main.locator('.rounded-full.h-2').first()).toBeVisible();
    });

    test('should render evolution charts when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h2:has-text("Evolutionary Optimization")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Evolution data not available');
      }
      await expect(heading).toBeVisible();

      // Check SVG chart exists and is visible
      const svg = main.locator('svg').first();
      await expect(svg).toBeVisible();

      // Check chart has polylines (actual data)
      await expect(main.locator('svg polyline').first()).toBeVisible();
    });

    test('should render comparison ranking bars when data is available', async ({ page }) => {
      const main = page.locator('main');
      const heading = main.locator('h2:has-text("Strategy Comparison")');
      if ((await heading.count()) === 0) {
        test.skip(true, 'Comparison data not available');
      }
      await expect(heading).toBeVisible();

      // Check for visual ranking bars (h-4 used by ComparisonSection tournament bars)
      const bars = main.locator('.rounded-full.h-4');
      await expect(bars.first()).toBeVisible();
    });
  });

  test.describe('Performance', () => {
    test('should load research page in reasonable time', async ({ page }) => {
      const startTime = Date.now();

      await page.goto('/research');
      await page.waitForLoadState('networkidle');

      const loadTime = Date.now() - startTime;

      // Should load in under 10 seconds (generous for CI)
      expect(loadTime).toBeLessThan(10000);
    });

    test('should switch scenarios quickly', async ({ page }) => {
      const scenarios = ['chain_reaction', 'deceptive_calm', 'rest_trap'];

      for (const scenario of scenarios) {
        const startTime = Date.now();

        await page.locator('select').selectOption(scenario);
        await page.waitForLoadState('networkidle');

        const switchTime = Date.now() - startTime;

        // Should switch in under 5 seconds (generous for CI)
        expect(switchTime).toBeLessThan(5000);
      }
    });
  });
});

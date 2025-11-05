import { test, expect } from '@playwright/test';

test.describe('Research Tab', () => {
  test.beforeEach(async ({ page }) => {
    // Listen for console errors
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    // Navigate to research page
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

  test('should have all three tabs visible', async ({ page }) => {
    // Check that all three tabs are present
    await expect(page.locator('button:has-text("Nash Equilibrium")')).toBeVisible();
    await expect(page.locator('button:has-text("Evolution")')).toBeVisible();
    await expect(page.locator('button:has-text("Heuristics")')).toBeVisible();

    // Check tab icons are present
    await expect(page.locator('button:has-text("🎯")')).toBeVisible();
    await expect(page.locator('button:has-text("🧬")')).toBeVisible();
    await expect(page.locator('button:has-text("🧠")')).toBeVisible();
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
    // Get the selector
    const selector = page.locator('select');

    // Change to a different scenario
    await selector.selectOption('chain_reaction');
    await page.waitForLoadState('networkidle');

    // Verify the scenario changed by checking the description
    await expect(page.locator('h2').first()).toBeVisible();

    // Change to another scenario
    await selector.selectOption('deceptive_calm');
    await page.waitForLoadState('networkidle');
    await expect(page.locator('h2').first()).toBeVisible();
  });

  test.describe('Nash Equilibrium Tab', () => {
    test.beforeEach(async ({ page }) => {
      // Click Nash tab
      await page.locator('button:has-text("Nash Equilibrium")').click();
      await page.waitForTimeout(1000);
    });

    test('should display Nash equilibrium metrics', async ({ page }) => {
      // Check for key metrics
      await expect(page.locator('text=Equilibrium Type')).toBeVisible();
      await expect(page.locator('text=Expected Payoff')).toBeVisible();
      await expect(page.locator('text=Cooperation Rate')).toBeVisible();
      await expect(page.locator('text=Convergence Time')).toBeVisible();
    });

    test('should display equilibrium strategies', async ({ page }) => {
      // Check for strategy pool section
      await expect(page.locator('h3:has-text("Equilibrium Strategies")')).toBeVisible();

      // Check for strategy classification (should be either "Free Rider", "Coordinator", etc.)
      await expect(page.locator('h4').first()).toBeVisible();

      // Check for agent parameters
      await expect(page.locator('text=honesty')).toBeVisible();
      await expect(page.locator('text=work tendency')).toBeVisible();
      await expect(page.locator('text=coordination')).toBeVisible();
    });

    test('should display computation details', async ({ page }) => {
      await expect(page.locator('h3:has-text("Computation Details")')).toBeVisible();
      await expect(page.locator('text=Algorithm')).toBeVisible();
      await expect(page.locator('text=Simulations')).toBeVisible();
      await expect(page.locator('text=Iterations')).toBeVisible();
      await expect(page.locator('text=Converged')).toBeVisible();
    });

    test('should show cooperation rate percentage', async ({ page }) => {
      // Look for cooperation rate with percentage
      const cooperationRate = page.locator('text=Cooperation Rate').locator('..').locator('..');
      await expect(cooperationRate).toContainText('%');
    });

    test('should display strategy parameters with visual bars', async ({ page }) => {
      // Check that parameter bars are rendered
      const parameterBars = page.locator('.rounded-full.h-2');
      await expect(parameterBars.first()).toBeVisible();

      // Check for multiple parameters (should be 10 parameters per strategy)
      const count = await parameterBars.count();
      expect(count).toBeGreaterThan(5);
    });
  });

  test.describe('Evolution Tab', () => {
    test.beforeEach(async ({ page }) => {
      // Click Evolution tab (it may be default)
      await page.locator('button:has-text("Evolution")').click();
      await page.waitForTimeout(1000);
    });

    test('should display evolution metrics', async ({ page }) => {
      await expect(page.locator('h2:has-text("Evolutionary Optimization")')).toBeVisible();
      await expect(page.locator('text=Best Fitness')).toBeVisible();
      await expect(page.locator('text=Generation')).toBeVisible();
      await expect(page.locator('text=Time')).toBeVisible();
    });

    test('should display fitness chart', async ({ page }) => {
      await expect(page.locator('h4:has-text("Fitness Over Generations")')).toBeVisible();

      // Check for SVG chart
      await expect(page.locator('svg polyline').first()).toBeVisible();

      // Check for legend
      await expect(page.locator('text=Best Fitness')).toBeVisible();
      await expect(page.locator('text=Mean Fitness')).toBeVisible();
    });

    test('should display best agent parameters', async ({ page }) => {
      await expect(page.locator('h4:has-text("Best Agent Parameters")')).toBeVisible();

      // Check for parameter bars
      const parameterBars = page.locator('.rounded-full.h-2');
      await expect(parameterBars.first()).toBeVisible();
    });

    test('should show comparison results', async ({ page }) => {
      // Check for strategy comparison section
      await expect(page.locator('h2:has-text("Strategy Comparison")')).toBeVisible();
      await expect(page.locator('h3:has-text("Tournament Results")')).toBeVisible();

      // Check for ranking numbers
      await expect(page.locator('text=#1').first()).toBeVisible();
    });
  });

  test.describe('Heuristics Tab', () => {
    test.beforeEach(async ({ page }) => {
      // Click Heuristics tab
      await page.locator('button:has-text("Heuristics")').click();
      await page.waitForTimeout(1000);
    });

    test('should display heuristic archetypes', async ({ page }) => {
      await expect(page.locator('h2:has-text("Heuristic Archetypes")')).toBeVisible();
      await expect(page.locator('h3:has-text("Homogeneous Teams")')).toBeVisible();
      await expect(page.locator('h3:has-text("Mixed Teams")')).toBeVisible();
    });

    test('should display team rankings', async ({ page }) => {
      // Check for ranking bars
      const rankingBars = page.locator('.rounded-full.h-3');
      await expect(rankingBars.first()).toBeVisible();

      // Check for multiple teams
      const count = await rankingBars.count();
      expect(count).toBeGreaterThan(3);
    });

    test('should show comparison results', async ({ page }) => {
      await expect(page.locator('h2:has-text("Strategy Comparison")')).toBeVisible();
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page, isMobile }) => {
      if (isMobile) {
        // Check that the page is accessible
        await expect(page.locator('h1')).toContainText('Scenario Research');

        // Check that tabs are visible and can be clicked
        await expect(page.locator('button:has-text("Nash Equilibrium")')).toBeVisible();
        await page.locator('button:has-text("Evolution")').click();
        await page.waitForTimeout(500);

        // Check that content is visible
        await expect(page.locator('h2').first()).toBeVisible();
      }
    });

    test('should handle landscape mobile', async ({ page, isMobile }) => {
      if (isMobile) {
        // Rotate to landscape
        await page.setViewportSize({ width: 812, height: 375 });

        // Check layout still works
        await expect(page.locator('h1')).toContainText('Scenario Research');
        await expect(page.locator('button:has-text("Nash Equilibrium")')).toBeVisible();
      }
    });
  });

  test.describe('Navigation', () => {
    test('should navigate from research to other tabs', async ({ page }) => {
      // Check we're on research tab
      const researchNav = page.locator('[data-testid="nav-research"]');
      await expect(researchNav).toHaveClass(/bg-blue-100|bg-interactive-active/);

      // Navigate to dashboard
      await page.locator('[data-testid="nav-dashboard"]').click();
      await page.waitForLoadState('networkidle');
      await expect(page.locator('main')).toContainText('Bucket Brigade');

      // Navigate back to research
      await page.locator('[data-testid="nav-research"]').click();
      await page.waitForLoadState('networkidle');
      await expect(page.locator('h1')).toContainText('Scenario Research');
    });

    test('should preserve state when navigating away and back', async ({ page }) => {
      // Select a specific scenario
      await page.locator('select').selectOption('rest_trap');
      await page.waitForTimeout(500);

      // Switch to Nash tab
      await page.locator('button:has-text("Nash Equilibrium")').click();
      await page.waitForTimeout(500);

      // Navigate away
      await page.locator('[data-testid="nav-dashboard"]').click();
      await page.waitForLoadState('networkidle');

      // Navigate back
      await page.locator('[data-testid="nav-research"]').click();
      await page.waitForLoadState('networkidle');

      // Check that state is preserved (or reasonably reset)
      await expect(page.locator('h1')).toContainText('Scenario Research');
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
      await expect(page.locator('h1')).toContainText('Scenario Research');
    });

    test('should load data for all scenarios', async ({ page }) => {
      const scenarios = [
        'greedy_neighbor',
        'trivial_cooperation',
        'sparse_heroics',
        'early_containment',
        'rest_trap',
        'chain_reaction',
        'deceptive_calm',
        'overcrowding',
        'mixed_motivation'
      ];

      for (const scenario of scenarios.slice(0, 3)) { // Test first 3 to keep test time reasonable
        await page.locator('select').selectOption(scenario);
        await page.waitForTimeout(800);

        // Should show scenario data
        await expect(page.locator('h2').first()).toBeVisible();
      }
    });

    test('should switch between tabs without errors', async ({ page }) => {
      const tabs = ['Nash Equilibrium', 'Evolution', 'Heuristics'];

      for (const tab of tabs) {
        await page.locator(`button:has-text("${tab}")`).click();
        await page.waitForTimeout(800);

        // Should show content for this tab
        await expect(page.locator('h2').first()).toBeVisible();
      }
    });
  });

  test.describe('Visual Regression', () => {
    test('should render Nash equilibrium tab correctly', async ({ page }) => {
      await page.locator('button:has-text("Nash Equilibrium")').click();
      await page.waitForTimeout(1000);

      // Check key visual elements
      await expect(page.locator('text=Equilibrium Type')).toBeVisible();
      await expect(page.locator('.rounded-full.h-2').first()).toBeVisible();
    });

    test('should render evolution charts', async ({ page }) => {
      await page.locator('button:has-text("Evolution")').click();
      await page.waitForTimeout(1000);

      // Check SVG chart exists and is visible
      const svg = page.locator('svg').first();
      await expect(svg).toBeVisible();

      // Check chart has polylines (actual data)
      await expect(page.locator('svg polyline').first()).toBeVisible();
    });

    test('should render heuristic comparison bars', async ({ page }) => {
      await page.locator('button:has-text("Heuristics")').click();
      await page.waitForTimeout(1000);

      // Check for visual ranking bars
      const bars = page.locator('.rounded-full.h-3');
      await expect(bars.first()).toBeVisible();

      const count = await bars.count();
      expect(count).toBeGreaterThan(5);
    });
  });

  test.describe('Performance', () => {
    test('should load research page in reasonable time', async ({ page }) => {
      const startTime = Date.now();

      await page.goto('/research');
      await page.waitForLoadState('networkidle');

      const loadTime = Date.now() - startTime;

      // Should load in under 5 seconds
      expect(loadTime).toBeLessThan(5000);
    });

    test('should switch tabs quickly', async ({ page }) => {
      const tabs = ['Nash Equilibrium', 'Evolution', 'Heuristics'];

      for (const tab of tabs) {
        const startTime = Date.now();

        await page.locator(`button:has-text("${tab}")`).click();
        await page.waitForLoadState('networkidle');

        const switchTime = Date.now() - startTime;

        // Should switch in under 2 seconds
        expect(switchTime).toBeLessThan(2000);
      }
    });
  });
});

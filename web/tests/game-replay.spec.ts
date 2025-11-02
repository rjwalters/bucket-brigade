import { test, expect } from '@playwright/test';

test.describe('Game Replay Functionality', () => {
  test('should load and replay game data', async ({ page }) => {
    // This test would require setting up test data
    // For now, it tests the UI structure

    await page.goto('/replay');

    // Should show empty state initially
    await expect(page.locator('text=No Games Available')).toBeVisible();

    // Check navigation back to dashboard
    await page.locator('text=Dashboard').click();
    await expect(page.locator('h1')).toContainText('Welcome to Bucket Brigade');
  });

  test('should handle replay controls', async ({ page }) => {
    // Mock having game data available
    await page.addScriptTag({
      content: `
        sessionStorage.setItem('bucket_brigade_replays', JSON.stringify([{
          scenario: {
            beta: 0.25, kappa: 0.5, A: 100, L: 100, c: 0.5,
            rho_ignite: 0.2, N_min: 12, p_spark: 0.02, N_spark: 12, num_agents: 4
          },
          nights: [
            { night: 0, houses: [0,1,0,0,0,0,0,1,0], signals: [0,1,0,1], locations: [0,1,2,3], actions: [[0,0],[1,1],[2,0],[3,1]], rewards: [2.5, -0.5, 4.5, 3.5] }
          ]
        }]));
      `
    });

    await page.reload();

    // Should now show game available
    await expect(page.locator('text=Game #0')).toBeVisible();

    // Click on game to select it
    await page.locator('text=Game #0').click();

    // Should show game board and controls
    await expect(page.locator('text=Game Board')).toBeVisible();
    await expect(page.locator('text=Scenario Parameters')).toBeVisible();

    // Test replay controls
    await expect(page.locator('button[title="Play"]')).toBeVisible();
    await expect(page.locator('button[title="Reset to beginning"]')).toBeVisible();
  });

  test('should validate game data integrity', async ({ page }) => {
    // Test with invalid data
    await page.addScriptTag({
      content: `
        sessionStorage.setItem('bucket_brigade_replays', JSON.stringify([{
          invalid: 'data'
        }]));
      `
    });

    await page.reload();
    await page.goto('/replay');

    // Should filter out invalid data and show empty state
    await expect(page.locator('text=No Games Available')).toBeVisible();
  });
});

test.describe('Agent Data Validation', () => {
  test('should validate batch result uploads', async ({ page }) => {
    await page.goto('/settings');

    // Create a test CSV file content
    const csvContent = `game_id,scenario_id,team,agent_params,team_reward,agent_rewards,nights_played,saved_houses,ruined_houses,replay_path
0,0,"[0,1,2,3]","[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1],[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2],[0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3]]",150.0,"[2.5,-0.5,4.5,3.5]",1,7,2,"replays/game_0.json"`;

    // Create a blob and simulate file upload
    await page.setInputFiles('input[type="file"][accept=".csv"]', {
      name: 'test_batch.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    // Should show success message or update data count
    await expect(page.locator('text=Successfully loaded')).toBeVisible();
  });

  test('should reject invalid CSV format', async ({ page }) => {
    await page.goto('/settings');

    const invalidCsvContent = `invalid,csv,format
this,is,not,a,valid,batch,result`;

    await page.setInputFiles('input[type="file"][accept=".csv"]', {
      name: 'invalid.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(invalidCsvContent)
    });

    // Should show error message
    await expect(page.locator('text=Error loading results file')).toBeVisible();
  });
});

test.describe('End-to-End Workflows', () => {
  test('should complete full data import workflow', async ({ page }) => {
    // This would test the complete workflow from data upload to visualization
    // For now, just test navigation and UI structure

    await page.goto('/');

    // Navigate through all sections
    await page.locator('text=Rankings').click();
    await expect(page.locator('text=Agent Rankings')).toBeVisible();

    await page.locator('text=Game Replay').click();
    await expect(page.locator('text=Select Game')).toBeVisible();

    await page.locator('text=Settings').click();
    await expect(page.locator('text=Data Management')).toBeVisible();

    // Back to dashboard
    await page.locator('text=Dashboard').click();
    await expect(page.locator('h1')).toContainText('Welcome to Bucket Brigade');
  });
});

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (err) => errors.push(err.message));
  return errors;
}

test('pick films, compare models and read the explanation', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('./');

  const search = page.getByRole('combobox', { name: 'Search for a film you like' });
  await search.fill('pulp fic');
  await expect(page.getByRole('option', { name: /Pulp Fiction \(1994\)/ })).toBeVisible();
  await search.press('ArrowDown');
  await search.press('Enter');

  const liked = page.getByRole('list', { name: 'Films you like' });
  await expect(liked.getByText('Pulp Fiction (1994)')).toBeVisible();
  await expect(page).toHaveURL(/#liked=296$/);

  const results = page
    .getByRole('region', { name: 'Recommendations' })
    .getByRole('listitem')
    .filter({
      has: page.getByRole('heading', { level: 3 }),
    });
  await expect(results).toHaveCount(10);
  await expect(results.first()).toContainText('Because you liked Pulp Fiction (1994)');

  await page.getByRole('radio', { name: 'Content' }).check();
  await expect(page).toHaveURL(/model=content/);
  await expect(results.first()).toContainText('0% co-ratings, 100% content');

  await search.fill('toy story');
  await page.getByRole('option', { name: /^Toy Story \(1995\)/ }).click();
  await expect(liked.getByRole('listitem')).toHaveCount(2);

  await page.getByRole('button', { name: 'Remove Pulp Fiction (1994)' }).click();
  await expect(liked.getByRole('listitem')).toHaveCount(1);
  await expect(results.first()).toContainText('Because you liked Toy Story (1995)');

  await page.getByText('About the model').click();
  await expect(page.getByRole('table').first()).toContainText('Hybrid');

  const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
  expect(axe.violations).toEqual([]);
  expect(errors).toEqual([]);
});

test('a film with too few ratings falls back to content in the hybrid', async ({ page }) => {
  await page.goto('./#liked=40&model=collab');
  await expect(page.getByRole('note')).toContainText('too few ratings');
  await page.getByRole('button', { name: 'Switch to hybrid' }).click();
  await expect(page.getByRole('radio', { name: 'Hybrid' })).toBeChecked();
  await expect(page.getByRole('heading', { level: 3 }).first()).toBeVisible();
});

test('ignores a malformed URL fragment', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('./#liked=<img src=x>,99999999&model=../../etc');
  await expect(page.getByRole('heading', { name: 'Films you like' })).toBeVisible();
  await expect(page.getByRole('list', { name: 'Films you like' })).toHaveCount(0);
  await expect(page.getByRole('radio', { name: 'Hybrid' })).toBeChecked();
  expect(errors).toEqual([]);
});

test('fits a 375px wide screen without horizontal scrolling', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await page.goto('./#liked=1,6377');
  await expect(page.getByRole('heading', { level: 3 }).first()).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

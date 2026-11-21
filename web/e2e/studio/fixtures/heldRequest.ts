import type { Page, Route } from '@playwright/test';

/** Holds the first request to `path` whose body `matches`, until `release` is called. */
export async function hold(page: Page, path: string, matches: (body: unknown) => boolean) {
  let release!: () => void;
  const released = new Promise<void>((resolve) => { release = resolve; });
  let arrived!: () => void;
  const sent = new Promise<void>((resolve) => { arrived = resolve; });
  let held = false;
  await page.route((url) => url.pathname === path, async (route: Route) => {
    if (!held && matches(route.request().postDataJSON())) {
      held = true;
      arrived();
      await released;
    }
    await route.fallback();
  });
  return { sent, release };
}

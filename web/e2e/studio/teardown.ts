import type { FullConfig } from '@playwright/test';
import { rm } from 'node:fs/promises';

export default async function teardown({ metadata }: FullConfig) {
  if (metadata.ownsHome === true) {
    await rm(metadata.home, { recursive: true, force: true });
  }
}

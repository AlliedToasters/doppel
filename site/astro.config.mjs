// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import queue from './queue/integration.mjs';

// https://astro.build/config
export default defineConfig({
  integrations: [mdx(), queue()]
});
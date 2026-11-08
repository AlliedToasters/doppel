import { getCollection } from 'astro:content';

/** Published posts, newest first. Drafts show only in `npm run dev`. */
export async function posts() {
  const all = await getCollection('blog', (p) => import.meta.env.DEV || !p.data.draft);
  return all.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export const fmtDate = (d: Date) =>
  d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });

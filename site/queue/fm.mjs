// Frontmatter helpers for the post queue (dev server and scripts/publish-next.mjs). Posts keep flat `key: value`
// frontmatter, so line edits are enough and the rest of the file is left byte-for-byte alone.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const BLOG = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/content/blog');
export const SLUG = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const FM = /^---\n([\s\S]*?)\n---\n/;

/** The post file for a slug, or null. Slugs are checked, so the path never leaves BLOG. */
export function file(slug) {
  if (!SLUG.test(slug)) return null;
  for (const ext of ['.mdx', '.md']) {
    const f = path.join(BLOG, slug + ext);
    if (fs.existsSync(f)) return f;
  }
  return null;
}

export function read(text) {
  const m = text.match(FM);
  if (!m) return {};
  const out = {};
  for (const line of m[1].split('\n')) {
    const kv = line.match(/^([a-zA-Z_]+):\s*(.*)$/);
    if (kv) out[kv[1]] = kv[2].replace(/^(['"])(.*)\1$/, '$2');
  }
  return out;
}

/** Set (or with null, remove) top-level keys. */
export function set(text, values) {
  const m = text.match(FM);
  if (!m) throw new Error('no frontmatter');
  let lines = m[1].split('\n');
  for (const [k, v] of Object.entries(values)) {
    const i = lines.findIndex((l) => l.startsWith(k + ':'));
    if (v === null) { if (i >= 0) lines.splice(i, 1); continue; }
    if (i >= 0) lines[i] = `${k}: ${v}`; else lines.push(`${k}: ${v}`);
  }
  return `---\n${lines.join('\n')}\n---\n` + text.slice(m[0].length);
}

/** Every post: slug, title, status, queue position (approved only), date. */
export function list() {
  return fs.readdirSync(BLOG)
    .filter((f) => /\.mdx?$/.test(f))
    .map((f) => {
      const d = read(fs.readFileSync(path.join(BLOG, f), 'utf8'));
      return {
        slug: f.replace(/\.mdx?$/, ''), title: d.title ?? f, description: d.description ?? '',
        status: d.status ?? 'draft', queue: d.queue ? Number(d.queue) : null, date: d.date ?? '',
      };
    });
}

/** Approved posts in publishing order. */
export const approved = () =>
  list().filter((p) => p.status === 'approved').sort((a, b) => (a.queue ?? 1e9) - (b.queue ?? 1e9));

export function update(slug, values) {
  const f = file(slug);
  fs.writeFileSync(f, set(fs.readFileSync(f, 'utf8'), values));
}

/** Renumber the approved queue 1..n in the given order. */
export function reorder(slugs) {
  slugs.forEach((s, i) => update(s, { queue: i + 1 }));
}

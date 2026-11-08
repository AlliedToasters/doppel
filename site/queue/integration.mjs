// Post queue: a page at /queue/ and a small JSON API at /__queue/, both in `astro dev` only. A build never
// contains them. The API writes post files, and MDX runs code when it renders, so every write must carry the
// X-Queue header (cross-site pages can't send it without a CORS preflight, which Vite refuses) and, if the
// browser sends an Origin, it must match the Host.
import fs from 'node:fs';
import * as fm from './fm.mjs';

const MAX = 512 * 1024;

function body(req) {
  return new Promise((ok, fail) => {
    let s = '';
    req.on('data', (c) => { s += c; if (s.length > MAX) { fail(new Error('too large')); req.destroy(); } });
    req.on('end', () => ok(s));
    req.on('error', fail);
  });
}

function sameOrigin(req) {
  const o = req.headers.origin;
  if (!o) return true;
  try { return new URL(o).host === req.headers.host; } catch { return false; }
}

async function handle(req, res) {
  const url = new URL(req.url, 'http://x');
  const send = (code, obj) => { res.statusCode = code; res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(obj)); };
  if (req.method !== 'GET' && (req.headers['x-queue'] !== '1' || !sameOrigin(req))) return send(403, { error: 'forbidden' });
  const slug = url.searchParams.get('slug') ?? '';
  const route = `${req.method} ${url.pathname}`;

  if (route === 'GET /__queue/posts') return send(200, { posts: fm.list() });
  if (route === 'GET /__queue/source') {
    const f = fm.file(slug);
    return f ? send(200, { text: fs.readFileSync(f, 'utf8') }) : send(404, { error: 'no such post' });
  }
  if (route === 'PUT /__queue/source') {
    const f = fm.file(slug);
    if (!f) return send(404, { error: 'no such post' });
    const text = await body(req);
    if (!/^---\n[\s\S]*?\n---\n/.test(text)) return send(400, { error: 'the file must start with frontmatter' });
    fs.writeFileSync(f, text);
    return send(200, { ok: true });
  }
  if (route === 'POST /__queue/status') {
    const { status } = JSON.parse(await body(req));
    if (!fm.file(slug) || !['draft', 'approved'].includes(status)) return send(400, { error: 'bad request' });
    const q = fm.approved().filter((p) => p.slug !== slug).map((p) => p.slug);
    fm.update(slug, { status, queue: null });
    fm.reorder(status === 'approved' ? [...q, slug] : q);   // approving puts it at the back of the queue
    return send(200, { ok: true });
  }
  if (route === 'POST /__queue/order') {
    const { slugs } = JSON.parse(await body(req));
    const now = new Set(fm.approved().map((p) => p.slug));
    if (!Array.isArray(slugs) || slugs.length !== now.size || !slugs.every((s) => now.has(s))) return send(400, { error: 'stale queue, reload' });
    fm.reorder(slugs);
    return send(200, { ok: true });
  }
  if (route === 'POST /__queue/new') {
    const { slug: s, title } = JSON.parse(await body(req));
    if (!fm.SLUG.test(s ?? '') || fm.file(s) || !title) return send(400, { error: 'need a new lowercase-hyphen slug and a title' });
    const today = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(`${fm.BLOG}/${s}.mdx`,
      `---\ntitle: ${JSON.stringify(title)}\ndescription: One or two sentences for the post list.\ndate: ${today}\nstatus: draft\n---\n\nStart writing here.\n`);
    return send(200, { ok: true });
  }
  return send(404, { error: 'not found' });
}

export default function queue() {
  return {
    name: 'doppel-post-queue',
    hooks: {
      'astro:config:setup': ({ command, injectRoute, updateConfig }) => {
        if (command !== 'dev') return;
        injectRoute({ pattern: '/queue', entrypoint: new URL('./Queue.astro', import.meta.url) });
        updateConfig({
          vite: {
            plugins: [{
              name: 'doppel-post-queue-api',
              configureServer(server) {
                server.middlewares.use((req, res, next) => {
                  if (!req.url?.startsWith('/__queue/')) return next();
                  handle(req, res).catch((e) => { res.statusCode = 500; res.end(JSON.stringify({ error: String(e.message ?? e) })); });
                });
              },
            }],
          },
        });
      },
    },
  };
}

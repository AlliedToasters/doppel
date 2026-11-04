// App-shell cache so the UI opens instantly; API calls always go to the network.
const CACHE = 'doppel-v1';
const SHELL = ['./', 'app.js', 'app.css', 'manifest.webmanifest', 'icon.svg', 'icon-192.png',
  'vendor/chessground.js', 'vendor/chess.js', 'vendor/chessground.base.css', 'vendor/chessground.cburnett.css'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.includes('/api/') || e.request.method !== 'GET') return;
  // network first, cache fallback: deploys show up on the next open, and it still opens offline
  e.respondWith(fetch(e.request).then((r) => {
    const copy = r.clone();
    caches.open(CACHE).then((c) => c.put(e.request, copy));
    return r;
  }).catch(() => caches.match(e.request)));
});

const CACHE_NAME = 'python-saas-cache-v2';
const urlsToCache = [
  '/',
  'index.html',
  'k1.py',
  'manifest.json',
  'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
          .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
          .then(response => response || fetch(event.request))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(name => {
          if (name !== CACHE_NAME) {
            return caches.delete(name);
          }
        })
      );
    })
  );
  // ─── 수정된 부분: 업데이트된 SW가 즉시 페이지를 제어하도록 ───
  event.waitUntil(self.clients.claim());
  // ───────────────────────────────────────────────────────────
});

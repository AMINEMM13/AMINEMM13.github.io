const CACHE_NAME = 'python-saas-cache-v4';
const urlsToCache = [
  '/',
  'index.html',
  'k1.py',
  'manifest.json',
  'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js'
];

self.addEventListener('install', event => {
  // 최신 SW를 즉시 활성화
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME)
          .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    Promise.all([
      // 옛 캐시 삭제
      caches.keys().then(keys => 
        Promise.all(keys.map(key => {
          if (key !== CACHE_NAME) return caches.delete(key);
        }))
      ),
      // 활성화 후 제어권 즉시 획득
      self.clients.claim()
    ])
  );
});

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // index.html 또는 루트는 네트워크 우선
  if (url.pathname === '/' || url.pathname.endsWith('index.html')) {
    event.respondWith(
      fetch(event.request)
        .then(resp => resp)
        .catch(() => caches.match('index.html'))
    );
    return;
  }

  // 그 외 리소스는 캐시 우선
  event.respondWith(
    caches.match(event.request)
          .then(resp => resp || fetch(event.request))
  );
});

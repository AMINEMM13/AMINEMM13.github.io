const CACHE_NAME = 'python-saas-cache-v9';
const urlsToCache = [
  '/',
  'index.html',
  'k1.py?v=9',
  'manifest.json',
  'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js'
];

self.addEventListener('install', event => {
  self.skipWaiting();  // 새 SW 즉시 활성화
  event.waitUntil(
    caches.open(CACHE_NAME)
          .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    Promise.all([
      // 구버전 캐시 삭제
      caches.keys().then(keys =>
        Promise.all(keys.map(key => {
          if (key !== CACHE_NAME) return caches.delete(key);
        }))
      ),
      self.clients.claim()  // 활성화 후 바로 페이지 제어
    ])
  );
});

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // index.html 은 네트워크 우선
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

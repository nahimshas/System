// Picks — Service Worker
// Strategy: network-first with cache fallback.
// Each new report overwrites the cache so the phone always shows
// today's picks when online, and yesterday's when offline.

const CACHE = 'picks-2026-05-28';
const PRECACHE = ['/'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE))
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  // Delete any old cache versions
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Push notification handler ─────────────────────────────────────────────────
self.addEventListener('push', e => {
  if (!e.data) return;

  let payload;
  try { payload = e.data.json(); }
  catch { payload = { title: 'Picks', body: e.data.text() }; }

  e.waitUntil(
    self.registration.showNotification(payload.title || 'Picks', {
      body:      payload.body  || '',
      icon:      '/icon-192.png',
      badge:     '/icon-192.png',
      tag:       payload.tag   || 'picks',
      renotify:  true,
      data:      { url: payload.url || '/index_spa.html' },
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  const target = e.notification.data?.url || '/index_spa.html';
  e.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(wcs => {
      for (const wc of wcs) {
        if (wc.url.includes(self.location.origin) && 'focus' in wc) {
          wc.navigate(target);
          return wc.focus();
        }
      }
      return clients.openWindow(target);
    })
  );
});

self.addEventListener('fetch', e => {
  // Only handle same-origin GET requests
  if (e.request.method !== 'GET') return;
  if (!e.request.url.startsWith(self.location.origin)) return;

  e.respondWith(
    fetch(e.request)
      .then(response => {
        // Cache a copy of the fresh response
        const clone = response.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return response;
      })
      .catch(() =>
        // Offline: serve from cache (yesterday's report)
        caches.match(e.request)
      )
  );
});

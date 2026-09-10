// Songgot Pocket service worker: caches the app shell and the model so the app works with no network.
// Model files are large, so they are cached on first successful fetch (not at install) to keep install fast.
const VERSION = "songgot-pocket-v1";
const SHELL = ["./", "./index.html", "./app.js", "./tools.json", "./manifest.webmanifest",
  "./vendor/wllama/dist/index.js", "./vendor/wllama/dist/wllama.wasm", "./icons/icon-192.png", "./icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;
  // cache-first for everything under the app scope; the model file is stored after its first full download
  e.respondWith(caches.match(req, { ignoreSearch: true }).then((hit) => hit || fetch(req).then((res) => {
    if (res.ok && (res.type === "basic")) {
      const copy = res.clone();
      caches.open(VERSION).then((c) => c.put(req, copy));
    }
    return res;
  })));
});

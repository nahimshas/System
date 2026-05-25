#!/usr/bin/env node
// Generate VAPID keys for Web Push notifications.
// Run once with:  node generate-keys.js
// Requires Node.js 18+

const { webcrypto } = require('crypto');
const { subtle } = webcrypto;

async function main() {
  const kp = await subtle.generateKey(
    { name: 'ECDSA', namedCurve: 'P-256' },
    true, ['sign', 'verify']
  );

  // Public key: 65-byte uncompressed point, base64url
  const pubRaw  = new Uint8Array(await subtle.exportKey('raw', kp.publicKey));
  const privJwk = await subtle.exportKey('jwk', kp.privateKey);
  // privJwk.d is the raw 32-byte private scalar in base64url already

  const toB64url = buf =>
    Buffer.from(buf).toString('base64').replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,'');

  const publicKey  = toB64url(pubRaw);
  const privateKey = privJwk.d;  // already base64url from Web Crypto JWK export

  console.log('\n====  VAPID Keys  ====\n');
  console.log('Public key  (embed in frontend + Cloudflare secret):');
  console.log(publicKey);
  console.log('\nPrivate key (Cloudflare secret ONLY — never commit):');
  console.log(privateKey);
  console.log('\n====  Next steps  ====\n');
  console.log('1. wrangler kv namespace create SUBSCRIPTIONS');
  console.log('   wrangler kv namespace create GAME_STATE');
  console.log('   wrangler kv namespace create PICKS_STORE');
  console.log('   → Paste the three IDs into wrangler.toml\n');
  console.log('2. wrangler secret put VAPID_PUBLIC_KEY   # paste public key above');
  console.log('   wrangler secret put VAPID_PRIVATE_KEY  # paste private key above');
  console.log('   wrangler secret put NOTIFY_SECRET      # any random string');
  console.log('   wrangler secret put VAPID_SUBJECT      # mailto:you@example.com\n');
  console.log('3. wrangler deploy\n');
  console.log('4. In report_spa.html, replace the two placeholders:');
  console.log('   REPLACE_WITH_VAPID_PUBLIC_KEY  →', publicKey.slice(0, 20) + '…');
  console.log('   REPLACE_WITH_WORKER_URL        → https://picks-notifications.<account>.workers.dev\n');
  console.log('5. In GitHub Actions secrets, add:');
  console.log('   NOTIFY_SECRET           (same value as step 2)');
  console.log('   Add variable NOTIFICATIONS_WORKER_URL (same URL as step 4)\n');
}

main().catch(console.error);

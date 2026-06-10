/**
 * Picks — Push Notification Worker  (v2: per-tab preferences)
 *
 * Routes
 *   POST   /subscribe            save a push subscription (body: PushSubscription JSON)
 *   PUT    /update-prefs         update notification prefs for existing subscription
 *                                body: { endpoint, prefs: { today, MLB, NBA, … } }
 *   DELETE /unsubscribe          remove a subscription  body: { endpoint }
 *   POST   /notify-picks-ready   (Bearer NOTIFY_SECRET) store picks + send "ready" blast
 *   GET    /health               liveness probe
 *
 * Cron  * /3 * * * *
 *   Poll ESPN, fire filtered notifications per subscriber prefs.
 *
 * Notification routing:
 *   picks-ready notification  → prefs.today === true
 *   debrief notification      → prefs.today === true  (Today's Card bell)
 *   parlay notification       → prefs.today === true
 *   single pick notification  → prefs[sport] === true
 *                               OR (prefs.today === true AND pick.inTodaysCard)
 *   Dedup: if both conditions match, one notification is sent (never two).
 */

// ─── Base64url helpers ────────────────────────────────────────────────────────

function b64url(buf) {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function fromB64url(s) {
  s = s.replace(/-/g, '+').replace(/_/g, '/');
  while (s.length % 4) s += '=';
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function concat(...arrays) {
  const total = arrays.reduce((n, a) => n + a.length, 0);
  const out = new Uint8Array(total);
  let pos = 0;
  for (const a of arrays) { out.set(a, pos); pos += a.length; }
  return out;
}

// ─── VAPID JWT (ES256) ────────────────────────────────────────────────────────

async function vapidAuthHeader(endpoint, env) {
  const origin = new URL(endpoint).origin;
  const exp    = Math.floor(Date.now() / 1000) + 43200;
  const sub    = env.VAPID_SUBJECT || 'mailto:admin@example.com';

  const hdr = b64url(new TextEncoder().encode(JSON.stringify({ typ: 'JWT', alg: 'ES256' })));
  const pay = b64url(new TextEncoder().encode(JSON.stringify({ aud: origin, exp, sub })));
  const unsigned = `${hdr}.${pay}`;

  const pubBytes = fromB64url(env.VAPID_PUBLIC_KEY);
  const x = b64url(pubBytes.slice(1, 33));
  const y = b64url(pubBytes.slice(33, 65));

  const privKey = await crypto.subtle.importKey(
    'jwk',
    { kty: 'EC', crv: 'P-256', x, y, d: env.VAPID_PRIVATE_KEY, key_ops: ['sign'] },
    { name: 'ECDSA', namedCurve: 'P-256' },
    false, ['sign']
  );

  const sig = await crypto.subtle.sign(
    { name: 'ECDSA', hash: 'SHA-256' },
    privKey,
    new TextEncoder().encode(unsigned)
  );

  return `vapid t=${unsigned}.${b64url(sig)},k=${env.VAPID_PUBLIC_KEY}`;
}

// ─── RFC 8291 Web Push encryption ────────────────────────────────────────────

async function encryptPayload(sub, payloadObj) {
  const plaintext  = new TextEncoder().encode(JSON.stringify(payloadObj));
  const recipPub   = fromB64url(sub.keys.p256dh);
  const authSecret = fromB64url(sub.keys.auth);

  const senderKP = await crypto.subtle.generateKey(
    { name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits']
  );
  const senderPub = new Uint8Array(
    await crypto.subtle.exportKey('raw', senderKP.publicKey)
  );

  const recipKey = await crypto.subtle.importKey(
    'raw', recipPub, { name: 'ECDH', namedCurve: 'P-256' }, false, []
  );
  const sharedBits = await crypto.subtle.deriveBits(
    { name: 'ECDH', public: recipKey }, senderKP.privateKey, 256
  );

  const salt = crypto.getRandomValues(new Uint8Array(16));

  const ikm = await crypto.subtle.importKey('raw', sharedBits, 'HKDF', false, ['deriveBits']);
  const prkInfo = concat(
    new TextEncoder().encode('WebPush: info\x00'),
    recipPub, senderPub
  );
  const prkBits = await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt: authSecret, info: prkInfo }, ikm, 256
  );

  const prk = await crypto.subtle.importKey('raw', prkBits, 'HKDF', false, ['deriveBits']);

  const cekBits = await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt,
      info: new TextEncoder().encode('Content-Encoding: aes128gcm\x00') },
    prk, 128
  );
  const nonceBits = await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt,
      info: new TextEncoder().encode('Content-Encoding: nonce\x00') },
    prk, 96
  );

  const cek   = await crypto.subtle.importKey('raw', cekBits, 'AES-GCM', false, ['encrypt']);
  const nonce = new Uint8Array(nonceBits);

  const padded = new Uint8Array(plaintext.length + 1);
  padded.set(plaintext);
  padded[plaintext.length] = 0x02;

  const ciphertext = new Uint8Array(
    await crypto.subtle.encrypt({ name: 'AES-GCM', iv: nonce }, cek, padded)
  );

  const rs  = 4096;
  const hdr = new Uint8Array(21);
  hdr.set(salt, 0);
  hdr[16] = (rs >> 24) & 0xff; hdr[17] = (rs >> 16) & 0xff;
  hdr[18] = (rs >>  8) & 0xff; hdr[19] =  rs        & 0xff;
  hdr[20] = 65;

  return concat(hdr, senderPub, ciphertext);
}

// ─── Send one push notification ───────────────────────────────────────────────

async function sendPush(sub, payload, env) {
  let body;
  try { body = await encryptPayload(sub, payload); }
  catch (e) { console.error('encryptPayload:', e.message); return false; }

  const authorization = await vapidAuthHeader(sub.endpoint, env);
  const resp = await fetch(sub.endpoint, {
    method: 'POST',
    headers: {
      'Authorization':    authorization,
      'Content-Type':     'application/octet-stream',
      'Content-Encoding': 'aes128gcm',
      'TTL':              '86400',
    },
    body,
  });

  if (resp.status === 410 || resp.status === 404) {
    await env.SUBSCRIPTIONS.delete(await subKey(sub.endpoint));
  } else if (!resp.ok) {
    console.error(`Push HTTP ${resp.status}: ${sub.endpoint.slice(-40)}`);
  }
  return resp.ok || resp.status === 201;
}

// ─── Preference-filtered broadcast ───────────────────────────────────────────
//
// context shape:
//   { isPicksReady: true }
//   { isParlay: true }
//   { sport: 'MLB', inTodaysCard: true/false }

function shouldSend(prefs, context) {
  if (context.isPicksReady) return !!prefs.today;
  if (context.isParlay)     return !!prefs.today;
  // Single pick: sport tab OR today's card (if in today's card)
  if (prefs[context.sport]) return true;
  if (prefs.today && context.inTodaysCard) return true;
  return false;
}

async function broadcastFiltered(payload, context, env) {
  const list = await env.SUBSCRIPTIONS.list({ limit: 1000 });
  if (!list.keys.length) return;

  const results = await Promise.allSettled(
    list.keys.map(async ({ name }) => {
      const raw = await env.SUBSCRIPTIONS.get(name);
      if (!raw) return;
      const sub = JSON.parse(raw);
      if (shouldSend(sub.prefs || {}, context)) {
        return sendPush(sub, payload, env);
      }
    })
  );

  const sent = results.filter(r => r.status === 'fulfilled' && r.value).length;
  console.log(`broadcast[${JSON.stringify(context)}] "${payload.title}": ${sent}/${list.keys.length}`);
}

// ─── KV key for a subscription ────────────────────────────────────────────────

async function subKey(endpoint) {
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(endpoint));
  return 'sub:' + b64url(hash).slice(0, 32);
}

// ─── Broadcast to ALL subscribers (no pref filtering) ────────────────────────

async function broadcastAll(payload, env) {
  const list = await env.SUBSCRIPTIONS.list({ limit: 1000 });
  if (!list.keys.length) return 0;

  const results = await Promise.allSettled(
    list.keys.map(async ({ name }) => {
      const raw = await env.SUBSCRIPTIONS.get(name);
      if (!raw) return;
      return sendPush(JSON.parse(raw), payload, env);
    })
  );

  const sent = results.filter(r => r.status === 'fulfilled' && r.value).length;
  console.log(`broadcastAll "${payload.title}": ${sent}/${list.keys.length}`);
  return sent;
}

// ─── ESPN helpers ─────────────────────────────────────────────────────────────

const ESPN_SPORT_PATH = {
  MLB:  'baseball/mlb',
  NBA:  'basketball/nba',
  NHL:  'hockey/nhl',
  WNBA: 'basketball/wnba',
  MLS:  'soccer/usa.1',
  NFL:  'football/nfl',
  // cricket/ipl returns 404 — use the IPL series ID endpoint instead
  // (same fix as outcome_checker.py which switched to cricket/8048)
  IPL:  'cricket/8048',
};

const SPORT_EMOJI = {
  MLB: '⚾', NBA: '🏀', NHL: '🏒', WNBA: '🏀', MLS: '⚽', NFL: '🏈', IPL: '🏏',
};

// Use Pacific time for all date keys — picks are stored under the Pacific date
// by the daily report, and ESPN returns games under local (ET) dates which
// match Pacific for any game starting before midnight Pacific.
// PDT = UTC-7 (active Mar–Nov, covers the entire sports season).
// Pure arithmetic avoids Intl/locale/timezone-database dependencies in V8.
function todayPacific() {
  const d = new Date(Date.now() - 7 * 60 * 60 * 1000);
  return d.toISOString().slice(0, 10); // always YYYY-MM-DD
}
function todayDateStr() {
  return todayPacific().replace(/-/g, '');
}
function todayIso() {
  return todayPacific();
}

async function espnScoreboard(sport, dateStr) {
  const path = ESPN_SPORT_PATH[sport];
  if (!path) return null;
  const url = `https://site.api.espn.com/apis/site/v2/sports/${path}/scoreboard?dates=${dateStr}`;
  try {
    const r = await fetch(url, { cf: { cacheTtl: 25 } });
    return r.ok ? r.json() : null;
  } catch { return null; }
}

function normTeam(name) {
  return (name || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

function gameMatchesPick(event, homeTeam, awayTeam) {
  const comps = event.competitions?.[0]?.competitors || [];
  const names = comps.map(c => normTeam(c.team?.displayName || c.team?.name || ''));
  const hn = normTeam(homeTeam);
  const an = normTeam(awayTeam);
  const hits = (n, t) => n.includes(t) || t.includes(n);
  return names.some(n => hits(n, hn)) && names.some(n => hits(n, an));
}

function gameScores(event) {
  let home = 0, away = 0;
  for (const c of (event.competitions?.[0]?.competitors || [])) {
    const s = parseInt(c.score ?? '0', 10);
    if (c.homeAway === 'home') home = s; else away = s;
  }
  return { home, away };
}

function gameStatus(event) {
  const state = event.status?.type?.state;
  if (state === 'post') return 'final';
  if (state === 'in')   return 'live';
  return 'pre';
}

function gameDetail(event) {
  return event.status?.type?.shortDetail || event.status?.type?.detail || '';
}

function getCompAbbr(event, side) {
  const comps = event.competitions?.[0]?.competitors || [];
  const c = comps.find(x => x.homeAway === side);
  return c?.team?.abbreviation || c?.team?.shortDisplayName || '';
}

// ─── Notification label helpers ──────────────────────────────────────────────

// Returns the pick name with " ML" appended for moneyline picks that don't
// already end in "ML" (e.g. "New York Yankees" → "New York Yankees ML").
function pickLabel(pick) {
  const p  = (pick.pick || '').trim();
  const bt = (pick.betType || '').toLowerCase();
  if (bt === 'moneyline' && !/\bML$/i.test(p)) return p + ' ML';
  return p;
}

// ─── Outcome determination ────────────────────────────────────────────────────

function determineOutcome(pick, homeTeamFull, homeScore, awayScore) {
  const p  = (pick.pick || '').trim();
  const bt = (pick.betType || '').toLowerCase();
  const normPick = normTeam(p.replace(/ ML$/i, '').trim());
  const normHome = normTeam(homeTeamFull);

  if (bt === 'moneyline' || /\bML$/i.test(p)) {
    const homeWon    = homeScore > awayScore;
    const isHomePick = normPick.includes(normHome) || normHome.includes(normPick);
    return homeWon === isHomePick ? 'WON' : 'LOST';
  }
  if (bt === 'total' || /^(over|under)\s/i.test(p)) {
    const isOver = /^over/i.test(p);
    const line   = parseFloat(p.replace(/^(over|under)\s*/i, ''));
    if (isNaN(line)) return null;
    const total = homeScore + awayScore;
    if (total === line) return null;
    return (isOver ? total > line : total < line) ? 'WON' : 'LOST';
  }
  if (bt === 'spread') {
    const m = p.match(/^(.+?)\s+([-+]?\d+\.?\d*)\s*$/);
    if (!m) return null;
    const spread     = parseFloat(m[2]);
    const normPickTm = normTeam(m[1].trim());
    const isHomePick = normPickTm.includes(normHome) || normHome.includes(normPickTm);
    const diff       = isHomePick ? homeScore - awayScore : awayScore - homeScore;
    if (diff + spread === 0) return null;
    return diff + spread > 0 ? 'WON' : 'LOST';
  }
  return null;
}

// ─── ESPN cron polling ────────────────────────────────────────────────────────

async function runCron(env) {
  const dateStr = todayDateStr();
  const isoDate = todayIso();

  const raw = await env.PICKS_STORE.get(`picks:${isoDate}`);
  if (!raw) return;

  const store = JSON.parse(raw);
  if (!store.date || store.date !== isoDate) return;

  const singles    = store.singles    || [];
  const parlayLegs = store.parlay_legs || [];
  const parlays    = store.parlays    || [];

  const bySport = {};
  for (const p of [...singles, ...parlayLegs]) {
    (bySport[p.sport] = bySport[p.sport] || []).push(p);
  }

  // ── Picks-ready broadcast (fires once, on first cron after KV write) ─────────
  // The workflow writes picks directly to KV via the Cloudflare REST API to
  // bypass workers.dev Bot Fight Mode.  A broadcast_done flag prevents re-firing.
  const broadcastKey  = `broadcast_done:${isoDate}`;
  const broadcastDone = await env.PICKS_STORE.get(broadcastKey);
  if (!broadcastDone) {
    const pickCount  = singles.length;
    const propCount  = 0;   // props not stored in KV (display-only)
    const pickWord   = pickCount === 1 ? '1 pick' : `${pickCount} picks`;
    const propPart   = propCount > 0 ? ` · ${propCount} props` : '';
    // Earliest start across Today's Card: budget singles + every parlay leg
    const cardTimes = [...singles.filter(s => s.inTodaysCard), ...parlayLegs]
      .map(p => p.commenceTime).filter(Boolean)
      .map(t => new Date(t)).filter(d => !isNaN(d));
    let firstGamePart = '';
    if (cardTimes.length) {
      const first = new Date(Math.min(...cardTimes));
      const timeStr = first.toLocaleTimeString('en-US', {
        timeZone: 'America/Los_Angeles', hour: 'numeric', minute: '2-digit',
      });
      firstGamePart = ` · ⏰ First game ${timeStr}`;
    }
    await broadcastFiltered({
      title: "🎯 Today's picks are ready",
      body:  `${pickWord}${propPart}${firstGamePart} · Tap to view`,
      tag:   `picks-ready-${isoDate}`, url: '/',
    }, { isPicksReady: true }, env);
    await env.PICKS_STORE.put(broadcastKey, '1', { expirationTtl: 86400 });
    console.log(`picks-ready broadcast fired: ${pickCount} picks`);
  }

  // ── Debrief notification (fires once after nightly_debrief.yml writes KV) ───
  const debriefKey = `debrief_notify:${isoDate}`;
  const debriefRaw = await env.PICKS_STORE.get(debriefKey);
  if (debriefRaw) {
    const debrief = JSON.parse(debriefRaw);
    if (!debrief.notified) {
      await broadcastFiltered({
        title: debrief.title || '📊 Nightly Debrief ready',
        body:  debrief.body  || "Today's picks analyzed — tap to review",
        tag:   `debrief-${isoDate}`,
        url:   debrief.url  || '/debrief_latest.html',
      }, { isPicksReady: true }, env);
      debrief.notified = true;
      await env.PICKS_STORE.put(debriefKey, JSON.stringify(debrief), { expirationTtl: 86400 });
      console.log(`Debrief notification fired for ${isoDate}`);
    }
  }

  let storeModified = false;

  for (const [sport, picks] of Object.entries(bySport)) {
    const data = await espnScoreboard(sport, dateStr);
    if (!data?.events?.length) continue;

    for (const event of data.events) {
      const espnId = event.id;
      const status = gameStatus(event);
      const scores = gameScores(event);
      const detail = gameDetail(event);

      const gamePicks = picks.filter(p =>
        p.espnEventId === espnId ||
        (!p.espnEventId && gameMatchesPick(event, p.homeTeam, p.awayTeam))
      );
      if (!gamePicks.length) continue;

      for (const p of gamePicks) {
        if (!p.espnEventId) {
          p.espnEventId = espnId;
          storeModified = true;
          // Sync espnEventId to the nested copy in store.parlays[i].legs so that
          // updateParlayOnLegFinal (which reads parlay.legs) can match the leg.
          // The flat parlay_legs and the nested parlay.legs are separate objects
          // after JSON.parse, so we must update both.
          if (p.isParlay && p.parlayId) {
            const pp = parlays.find(x => x.id === p.parlayId);
            const nl = pp?.legs?.find(l => l.id === p.id);
            if (nl && !nl.espnEventId) nl.espnEventId = espnId;
          }
        }
      }

      const gsKey  = `gs:${sport}:${espnId}`;
      const prevRaw = await env.GAME_STATE.get(gsKey);
      const prev   = prevRaw ? JSON.parse(prevRaw) : {
        status: 'pre', homeScore: 0, awayScore: 0, seenLive: false, seenFinal: false,
      };

      const homeAbbr = getCompAbbr(event, 'home');
      const awayAbbr = getCompAbbr(event, 'away');
      const homeFull = (event.competitions?.[0]?.competitors || [])
        .find(c => c.homeAway === 'home')?.team?.displayName || gamePicks[0]?.homeTeam || '';

      // ── Skip entirely if already fully settled (no more state changes possible) ──
      // This avoids a GAME_STATE write every cron tick after a game is final.
      if (prev.seenFinal) continue;

      // ── Game started ──────────────────────────────────────────────────────
      if (status === 'live' && !prev.seenLive) {
        for (const pick of gamePicks) {
          const emoji = SPORT_EMOJI[sport] || '🏆';
          if (pick.isParlay) {
            // Skip if the parent parlay is already resolved (e.g. leg 1 already lost)
            const parentParlay = parlays.find(p => p.id === pick.parlayId);
            if (parentParlay?.notifiedFinal) continue;
            await broadcastFiltered({
              title: `🎰 ${pick.parlayLabel} · Leg ${pick.legNum}/${pick.legTotal} LIVE`,
              body:  `${pickLabel(pick)} · ${awayAbbr} 0 – ${homeAbbr} 0`,
              tag:   `live-${pick.id}`, url: '/',
            }, { isParlay: true }, env);
          } else {
            await broadcastFiltered({
              title: `${emoji} ${pickLabel(pick)} — LIVE`,
              body:  `${awayAbbr} 0 · ${homeAbbr} 0 · Game started`,
              tag:   `live-${pick.id}`, url: '/',
            }, { sport, inTodaysCard: !!pick.inTodaysCard }, env);
          }
        }
        prev.seenLive = true;
      }

      // ── Game ended ────────────────────────────────────────────────────────
      if (status === 'final' && !prev.seenFinal) {
        const scoreStr = `Final: ${awayAbbr} ${scores.away} · ${homeAbbr} ${scores.home}`;

        for (const pick of gamePicks) {
          if (pick.isParlay) continue; // parlays handled by updateParlayOnLegFinal below
          const emoji   = SPORT_EMOJI[sport] || '🏆';
          const outcome = determineOutcome(pick, homeFull, scores.home, scores.away);
          const ctx     = { sport, inTodaysCard: !!pick.inTodaysCard };
          if (outcome === 'WON') {
            const pnl = (pick.inTodaysCard && pick.profitIfWin != null)
              ? ` · +$${Number(pick.profitIfWin).toFixed(2)}` : '';
            await broadcastFiltered({
              title: `✅ ${pickLabel(pick)} — WON`,
              body:  `${scoreStr}${pnl}`,
              tag:   `end-${pick.id}`, url: '/',
            }, ctx, env);
          } else if (outcome === 'LOST') {
            const pnl = (pick.inTodaysCard && pick.cost != null)
              ? ` · -$${Number(pick.cost).toFixed(2)}` : '';
            await broadcastFiltered({
              title: `❌ ${pickLabel(pick)} — LOST`,
              body:  `${scoreStr}${pnl}`,
              tag:   `end-${pick.id}`, url: '/',
            }, ctx, env);
          } else {
            await broadcastFiltered({
              title: `${emoji} ${pickLabel(pick)} — FINAL`,
              body:  scoreStr,
              tag:   `end-${pick.id}`, url: '/',
            }, ctx, env);
          }
        }

        if (await updateParlayOnLegFinal(parlays, espnId, scores, homeFull, env)) {
          storeModified = true;
        }
        prev.seenFinal = true;
      }

      // Only write GAME_STATE if something meaningful changed — avoids burning the
      // free-tier 1,000 writes/day limit with 480 unconditional writes per game.
      const nextState = {
        status,
        homeScore: scores.home, awayScore: scores.away,
        seenLive:  prev.seenLive,
        seenFinal: prev.seenFinal,
        detail,
      };
      const stateChanged = (
        nextState.status    !== prev.status    ||
        nextState.homeScore !== prev.homeScore ||
        nextState.awayScore !== prev.awayScore ||
        nextState.seenLive  !== prev.seenLive  ||
        nextState.seenFinal !== prev.seenFinal
      );
      if (stateChanged) {
        await env.GAME_STATE.put(gsKey, JSON.stringify(nextState), { expirationTtl: 172800 });
      }
    }
  }

  if (storeModified) {
    await env.PICKS_STORE.put(`picks:${isoDate}`, JSON.stringify(store), {
      expirationTtl: 604800,
    });
  }
}

// ─── Smart parlay leg resolution ─────────────────────────────────────────────
//
// Called each time a game goes final. Handles three cases:
//   1. Any leg LOST  → notify "Parlay LOST" immediately, ignore remaining legs
//   2. All legs from SAME game → resolve entire parlay in one notification
//   3. Sequential legs (different games) → "Leg X/N hit · N remaining" until all done
//
// Leg outcomes are persisted on the leg object itself (leg.outcome = 'WON'/'LOST'/null)
// so state survives across cron ticks. Returns true if the store was modified.

async function updateParlayOnLegFinal(parlays, resolvedEspnId, resolvedScores, resolvedHomeFull, env) {
  let modified = false;

  for (const parlay of parlays) {
    if (parlay.notifiedFinal) continue;
    const legs = parlay.legs || [];
    if (!legs.length) continue;

    // Only process parlays that have at least one leg in this game
    const thisGameLegs = legs.filter(leg => leg.espnEventId === resolvedEspnId);
    if (!thisGameLegs.length) continue;

    // Resolve outcomes for all legs from this game
    for (const leg of thisGameLegs) {
      if (leg.outcome != null) continue; // already resolved in a previous cron tick
      leg.outcome = determineOutcome(leg, resolvedHomeFull, resolvedScores.home, resolvedScores.away);
      // null outcome = push/tie → treat as LOST (conservative)
      if (leg.outcome === null) leg.outcome = 'LOST';
      modified = true;
    }

    // Are ALL legs of this parlay from the same game?
    const allSameGame = legs.every(leg => leg.espnEventId === resolvedEspnId);

    const anyLost    = legs.some(leg => leg.outcome === 'LOST');
    const wonLegs    = legs.filter(leg => leg.outcome === 'WON');
    const pendingLegs = legs.filter(leg => leg.outcome == null);

    if (anyLost) {
      // ── Parlay dead — fire immediately, no more notifications for this parlay ──
      const killerLeg = legs.find(leg => leg.outcome === 'LOST');
      await broadcastFiltered({
        title: `❌ ${parlay.label} — LOST`,
        body:  `${pickLabel(killerLeg)} didn't hit · -$${Number(parlay.cost || 0).toFixed(2)}`,
        tag:   `parlay-final-${parlay.id}`, url: '/',
      }, { isParlay: true }, env);
      parlay.notifiedFinal = true;
      modified = true;

    } else if (pendingLegs.length === 0) {
      // ── All legs resolved and all WON ────────────────────────────────────────
      await broadcastFiltered({
        title: `🏆 ${parlay.label} — WON`,
        body:  `All ${legs.length} legs hit · +$${Number(parlay.profitIfWin || 0).toFixed(2)}`,
        tag:   `parlay-final-${parlay.id}`, url: '/',
      }, { isParlay: true }, env);
      parlay.notifiedFinal = true;
      modified = true;

    } else if (!allSameGame && wonLegs.length > 0) {
      // ── Intermediate: some legs WON, more games still pending ────────────────
      const remaining  = pendingLegs.length;
      const legNames   = thisGameLegs.filter(l => l.outcome === 'WON').map(l => pickLabel(l)).join(' + ');
      await broadcastFiltered({
        title: `✅ ${parlay.label} — Leg ${wonLegs.length}/${legs.length} hit`,
        body:  `${legNames} · ${remaining} leg${remaining !== 1 ? 's' : ''} remaining`,
        tag:   `parlay-progress-${parlay.id}-${wonLegs.length}`, url: '/',
      }, { isParlay: true }, env);
      // notifiedFinal stays false — waiting for remaining legs
    }
    // allSameGame + pending: all same-game legs resolve together, no intermediate needed
  }

  return modified;
}

// ─── HTTP handlers ────────────────────────────────────────────────────────────

function cors(origin) {
  return {
    'Access-Control-Allow-Origin':  origin || '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age':       '86400',
  };
}

function jsonResp(body, status = 200) {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), {
    status, headers: { 'Content-Type': 'application/json' },
  });
}

async function handleSubscribe(req, env) {
  let sub;
  try { sub = await req.json(); } catch { return jsonResp('Bad JSON', 400); }

  if (!sub?.endpoint || !sub?.keys?.p256dh || !sub?.keys?.auth) {
    return jsonResp('Missing required fields', 400);
  }

  const key      = await subKey(sub.endpoint);
  const existing = await env.SUBSCRIPTIONS.get(key);
  const oldPrefs = existing ? (JSON.parse(existing).prefs || {}) : {};

  await env.SUBSCRIPTIONS.put(key, JSON.stringify({
    endpoint: sub.endpoint,
    keys:     { p256dh: sub.keys.p256dh, auth: sub.keys.auth },
    prefs:    oldPrefs,   // preserve existing prefs on re-subscribe
  }));
  return jsonResp({ ok: true }, 201);
}

async function handleUpdatePrefs(req, env) {
  let body;
  try { body = await req.json(); } catch { return jsonResp('Bad JSON', 400); }

  const { endpoint, prefs } = body;
  if (!endpoint || !prefs) return jsonResp('Missing endpoint or prefs', 400);

  const key = await subKey(endpoint);
  const raw = await env.SUBSCRIPTIONS.get(key);
  if (!raw) return jsonResp('Subscription not found', 404);

  const stored = JSON.parse(raw);
  stored.prefs = { ...(stored.prefs || {}), ...prefs };
  await env.SUBSCRIPTIONS.put(key, JSON.stringify(stored));
  return jsonResp({ ok: true });
}

async function handleUnsubscribe(req, env) {
  let body;
  try { body = await req.json(); } catch { return jsonResp('Bad JSON', 400); }
  if (!body?.endpoint) return jsonResp('Missing endpoint', 400);
  await env.SUBSCRIPTIONS.delete(await subKey(body.endpoint));
  return jsonResp({ ok: true });
}

async function handleNotifyPicksReady(req, env) {
  const auth = req.headers.get('Authorization') || '';
  if (auth !== `Bearer ${env.NOTIFY_SECRET}`) return jsonResp('Unauthorized', 401);

  let data;
  try { data = await req.json(); } catch { return jsonResp('Bad JSON', 400); }

  const { date, pickCount = 0, propCount = 0, singles = [], parlays = [] } = data;
  if (!date) return jsonResp('Missing date', 400);

  const parlayLegs = [];
  const storedParlays = parlays.map((p, pi) => {
    const id    = `parlay-${pi}`;
    const label = p.label || `Parlay ${pi + 1}`;
    const legs  = (p.legs || []).map((leg, li) => {
      const legObj = {
        ...leg,
        id: `p${pi}-l${li}`, parlayId: id, parlayLabel: label,
        legNum: li + 1, legTotal: p.legs.length,
        isParlay: true, cost: p.cost || 0, profitIfWin: p.profitIfWin || 0,
        espnEventId: null,
      };
      parlayLegs.push(legObj);
      return { ...legObj };
    });
    return { id, label, cost: p.cost || 0, profitIfWin: p.profitIfWin || 0,
             legs, notifiedFinal: false };
  });

  const storedSingles = singles.map((s, i) => ({
    ...s, id: `s${i}`, isParlay: false, espnEventId: null,
  }));

  await env.PICKS_STORE.put(`picks:${date}`, JSON.stringify({
    date, singles: storedSingles, parlay_legs: parlayLegs, parlays: storedParlays,
  }), { expirationTtl: 604800 });

  // Picks-ready: only today's-card subscribers
  const pickWord = pickCount === 1 ? '1 pick' : `${pickCount} picks`;
  const propPart = propCount > 0 ? ` · ${propCount} props` : '';
  await broadcastFiltered({
    title: "🎯 Today's picks are ready",
    body:  `${pickWord}${propPart} · Tap to view`,
    tag:   `picks-ready-${date}`, url: '/',
  }, { isPicksReady: true }, env);

  const subCount = (await env.SUBSCRIPTIONS.list()).keys.length;
  return jsonResp({ ok: true, subscribers: subCount });
}

async function handleHealth(env) {
  const isoDate = todayIso();
  const raw     = await env.PICKS_STORE.get(`picks:${isoDate}`);
  const store   = raw ? JSON.parse(raw) : null;
  const subList = await env.SUBSCRIPTIONS.list({ limit: 1000 });

  // Collect per-subscriber prefs (without exposing keys/endpoints)
  const subPrefs = await Promise.all(
    subList.keys.map(async ({ name }) => {
      const s = await env.SUBSCRIPTIONS.get(name);
      return s ? (JSON.parse(s).prefs || {}) : null;
    })
  );

  // Summarise game state for all tracked games
  const gameStates = [];
  if (store) {
    const sports = {};
    for (const p of [...(store.singles || []), ...(store.parlay_legs || [])]) {
      (sports[p.sport] = sports[p.sport] || []).push(p);
    }
    for (const [sport, picks] of Object.entries(sports)) {
      for (const p of picks) {
        if (p.espnEventId) {
          const gsRaw = await env.GAME_STATE.get(`gs:${sport}:${p.espnEventId}`);
          if (gsRaw) gameStates.push({ sport, id: p.espnEventId, ...JSON.parse(gsRaw) });
        }
      }
    }
  }

  return jsonResp({
    ok:           true,
    pacificDate:  isoDate,
    dateStr:      todayDateStr(),
    hasPicks:     !!store,
    singles:      store ? (store.singles     || []).length : 0,
    parlayLegs:   store ? (store.parlay_legs || []).length : 0,
    subscribers:  subList.keys.length,
    subPrefs,
    gameStates,
  });
}

// ─── Main export ──────────────────────────────────────────────────────────────

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);
    const origin   = request.headers.get('Origin') || '';
    const corsHdrs = cors(origin);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHdrs });
    }

    let resp;
    if      (pathname === '/subscribe'          && request.method === 'POST')   resp = await handleSubscribe(request, env);
    else if (pathname === '/update-prefs'       && request.method === 'PUT')    resp = await handleUpdatePrefs(request, env);
    else if (pathname === '/unsubscribe'        && request.method === 'DELETE') resp = await handleUnsubscribe(request, env);
    else if (pathname === '/notify-picks-ready' && request.method === 'POST')   resp = jsonResp({ ok: true, note: 'picks now written directly to KV by workflow' });
    else if (pathname === '/health')                                             resp = await handleHealth(env);
    else resp = new Response('Not Found', { status: 404 });

    const headers = new Headers(resp.headers);
    for (const [k, v] of Object.entries(corsHdrs)) headers.set(k, v);
    return new Response(resp.body, { status: resp.status, headers });
  },

  async scheduled(event, env, ctx) {
    ctx.waitUntil(runCron(env));
  },
};

/**
 * Rate limiting and origin control in front of ResearchLens.
 *
 * Everything here exists because the demo is public and the thing behind it
 * costs money and CPU to wake. Refusing at the edge costs neither.
 */

const JSON_HEADERS = { 'Content-Type': 'application/json' };

function corsHeaders(request, env) {
  const origin = request.headers.get('Origin') ?? '';
  const allowed = (env.ALLOWED_ORIGINS ?? '').split(',').map((s) => s.trim());
  // Echo the origin only when it is on the list. A wildcard would let any page
  // spend this instance's budget from a visitor's browser.
  const allow = allowed.includes(origin) ? origin : allowed[0] ?? '';
  return {
    'Access-Control-Allow-Origin': allow,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '600',
    Vary: 'Origin',
  };
}

function refuse(status, message, request, env) {
  return new Response(JSON.stringify({ detail: message }), {
    status,
    headers: { ...JSON_HEADERS, ...corsHeaders(request, env) },
  });
}

/** Increment a counter with a TTL, returning the new value. */
async function bump(kv, key, ttlSeconds) {
  const current = Number((await kv.get(key)) ?? 0);
  const next = current + 1;
  await kv.put(key, String(next), { expirationTtl: ttlSeconds });
  return next;
}

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(request, env) });
    }

    const url = new URL(request.url);

    // /health is unmetered. The page probes it on every load, and a reader who
    // opens the page twice has not used the demo at all.
    const metered = url.pathname === '/ask' || url.pathname === '/ask/stream';

    if (metered && env.RL_LIMITS) {
      const ip = request.headers.get('CF-Connecting-IP') ?? 'unknown';
      const hour = new Date().toISOString().slice(0, 13);
      const day = new Date().toISOString().slice(0, 10);

      const perIp = await bump(env.RL_LIMITS, `ip:${ip}:${hour}`, 3600);
      if (perIp > Number(env.RATE_PER_HOUR ?? 20)) {
        return refuse(
          429,
          'Rate limit reached for this hour. ResearchLens runs locally with ' +
            'no API key — see github.com/asifuddin01/researchlens.',
          request,
          env
        );
      }

      const today = await bump(env.RL_LIMITS, `day:${day}`, 86400);
      if (today > Number(env.DAILY_CEILING ?? 600)) {
        // Fails closed, and says why. A demo that silently degrades teaches a
        // reader that the system is unreliable rather than that it is capped.
        return refuse(
          503,
          "The demo's daily limit is reached. It runs locally with no API key — " +
            'see github.com/asifuddin01/researchlens.',
          request,
          env
        );
      }
    }

    const upstream = new Request(`${env.ORIGIN}${url.pathname}${url.search}`, request);
    let response;
    try {
      response = await fetch(upstream);
    } catch (e) {
      // The retrieval machine suspends between visitors; a wake occasionally
      // outlasts the first request. 503 says "try again", which is true.
      return refuse(503, `ResearchLens is waking up. Try again in a moment. (${e})`, request, env);
    }

    const headers = new Headers(response.headers);
    for (const [k, v] of Object.entries(corsHeaders(request, env))) headers.set(k, v);
    return new Response(response.body, { status: response.status, headers });
  },
};

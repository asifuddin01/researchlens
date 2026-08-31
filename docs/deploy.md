# Deploying the public demo

Everything here is prepared and tested except the steps that need your
credentials. Those are marked **you**.

The shape: two Fly machines and a Cloudflare Worker.

```
asifuddin.com/researchlens          static page, already live
        │
        ▼
api.asifuddin.com                   Worker — rate limit, daily cap, CORS
        │
        ▼
researchlens-api    (2 GB, suspends, wakes <1s)   retrieval + API
        │
        ▼
researchlens-llm    (4 GB, stops, ~60s boot)      generation
```

Two machines, not one, for a measured reason: Fly resumes a suspended machine
from a memory snapshot in well under a second, but only up to about 2 GB. A 3B
model at Q4 is ~2 GB by itself. Keeping them apart means the part every visitor
touches is instant, and only the part they explicitly choose is slow.

## What is already done

- **The image needs no PDFs.** `scripts/export_bundle.py` writes passages and
  vectors — 24 MB against ~600 MB of corpus — and the engine loads them
  directly. Verified with the PDFs absent: 101 papers, 9,540 passages, 0.5 s.
- **`deploy/fly.retrieval.toml`** and **`deploy/fly.ollama.toml`**, sized and
  annotated.
- **`deploy/worker/`** — per-IP hourly limit, whole-instance daily ceiling,
  origin allow-list, failing closed with a message that names the local
  alternative.
- **The Dockerfile** copies the bundle and runs one worker.

## Before anything: which papers ship

```bash
python scripts/export_bundle.py --open-access-only
```

A bundle is extracted full text. A private image for your own use is one thing;
a public one built from subscription journal PDFs is another. The filter matches
on filenames and is coarse — **read the list it prints before publishing.**

## you · Fly

```bash
brew install flyctl && fly auth login
```

```bash
fly launch --no-deploy --copy-config --config deploy/fly.ollama.toml
fly volumes create ollama_models --size 5 --region sin -a researchlens-llm
fly deploy --config deploy/fly.ollama.toml
fly ssh console -a researchlens-llm -C "ollama pull qwen2.5:3b-instruct"
```

```bash
fly launch --no-deploy --copy-config --config deploy/fly.retrieval.toml
fly deploy --config deploy/fly.retrieval.toml
fly apps list   # confirm both, then: curl https://researchlens-api.fly.dev/health
```

Ollama has no public route by design — an open inference endpoint is free
compute for whoever finds it. It is reachable only from the retrieval service
over `.flycast`.

## you · Cloudflare Worker

```bash
npx wrangler kv namespace create RL_LIMITS
```

Put the printed id into `deploy/worker/wrangler.toml`, then:

```bash
npx wrangler deploy -c deploy/worker/wrangler.toml
```

Add a route for `api.asifuddin.com/*` in the Cloudflare dashboard, or with
`wrangler`, pointing at the `researchlens-api` worker.

## Point the page at it

```bash
PUBLIC_RESEARCHLENS_API=https://api.asifuddin.com npm run build
```

The page probes that first and falls back to localhost, so a reader running the
container locally still gets their own instance.

## Cost

Both machines idle at zero. Fly bills a suspended machine for its snapshot
storage and a stopped one for its rootfs — pennies. Real cost is the seconds
each is awake, which the Worker's daily ceiling bounds.

## What to check after

```bash
curl https://api.asifuddin.com/health
```

Then open the page. `/health` is unmetered on purpose: the page probes it on
every load, and a reader who opens the page twice has not used the demo.

## Known limits

- **First wake after a deploy is a cold boot,** not a resume — the snapshot
  belongs to the previous version.
- **Fly does not guarantee a resume.** Host migration or capacity pressure
  gives a cold start instead. The page shows a wake state rather than an error.
- **A local-model answer takes about a minute** on first use, the machine boot
  plus generation. That is the trade the project is arguing about, so the page
  says so rather than hiding it behind a spinner.

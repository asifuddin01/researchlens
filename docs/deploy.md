# Deploying the public demo

Two routes are prepared. **Hugging Face Spaces** is the one to use: free, no
card, and 2 vCPU / 16 GB on the free CPU tier — more memory than the Fly plan
below. Fly is kept because it is better on latency and is the right answer if
the demo ever matters enough to pay for.

---

# Route A · Hugging Face Spaces

## What differs from local, and why

Generation goes to a hosted OpenAI-compatible endpoint rather than Ollama. A
free Space has no persistent volume, so a local model would re-download two
gigabytes on every cold start — a minute of waiting, repeated, for a model
weaker than the hosted one. Locally the local model stays the default, where it
is the entire point; on a Space it would be theatre.

Conveniently this needs no new code: Hugging Face's Inference Providers speak
the OpenAI protocol, so the existing hosted provider works with a base URL and
a token.

## Build the bundle first

```bash
python scripts/export_bundle.py --open-access-only
```

A bundle is extracted full text. **Read the list it prints before publishing** —
a private image built from subscription journals is one thing, a public Space
is another.

## you · create the Space

At <https://huggingface.co/new-space>: name `researchlens`, SDK **Docker**,
hardware **CPU basic (free)**, visibility public.

Then, in Settings → Variables and secrets, add a secret:

| name | value |
|---|---|
| `HOSTED_API_KEY` | a token from <https://huggingface.co/settings/tokens> with **Inference** permission |

The token is a Space secret, so it stays server-side — the page never sees it.
Without it the Space still starts and serves `/search`, and `/ask` reports that
no provider is configured.

## Push

```bash
git clone https://huggingface.co/spaces/<your-username>/researchlens ~/hf/researchlens
./deploy/hf/sync.sh ~/hf/researchlens
cd ~/hf/researchlens && git add -A && git commit -m "ResearchLens" && git push
```

The first build takes several minutes; watch it in the Space's **Logs** tab.

## Point the page at it

```bash
PUBLIC_RESEARCHLENS_API=https://<your-username>-researchlens.hf.space npm run build
```

Set that in the portfolio repo and deploy as usual. The page probes it first and
still falls back to localhost, so a reader running the container locally gets
their own instance instead.

## What to expect

- **Cold start.** A sleeping Space wakes in roughly half a minute, then loads
  two ONNX models. The first question after a quiet period is slow; the rest
  are not.
- **No rate limiting.** The Cloudflare Worker below is Fly-shaped. On a Space
  the free hardware is itself the limit, and the HF token has its own quota.
- **Sleeps after inactivity.** Expected, and the page's wake state covers it.

---



Everything here is prepared and tested except the steps that need your
credentials. Those are marked **you**.

# Route B · Fly.io

Better latency and a real volume for a local model, at a couple of dollars a
month. Requires a payment method on the Fly account, which is verification
against abuse rather than a charge.

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

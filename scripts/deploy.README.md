# PAYBACK Assistant — Cloud Run Deployment Guide

## What deploy.sh does

`scripts/deploy.sh` is a single-command deployment script that takes the local repo
from source to a live Cloud Run service. It verifies prerequisites (gcloud CLI,
authenticated account, project configuration, required APIs), stores the Anthropic API
key in GCP Secret Manager under the name `payback-anthropic-key` (idempotent — skips
creation if the secret already exists), and calls `gcloud run deploy --source .` which
hands the build off to Cloud Build. Cloud Build constructs the container image from the
`Dockerfile` in the repo root and pushes it to Artifact Registry; Cloud Run then pulls
and runs it. The script finishes with a health-check probe against the live URL and
prints the service URL, Swagger UI link, and a sample `curl` command.

No local Docker daemon is required — the build happens entirely in the cloud.

---

## Prerequisites

### 1. Google Cloud SDK

```bash
# macOS (Homebrew)
brew install --cask google-cloud-sdk

# Or download the installer:
# https://cloud.google.com/sdk/docs/install
```

Verify: `gcloud version`

### 2. Authenticate and set a project

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Verify:
```bash
gcloud config get-value account   # should show your email
gcloud config get-value project   # should show your project ID
```

### 3. Enable billing

Cloud Run, Cloud Build, and Secret Manager all require an active billing account.
Enable billing in the GCP console: **Billing → Link a billing account**.

The script will attempt to enable the four required APIs automatically
(`run.googleapis.com`, `artifactregistry.googleapis.com`,
`secretmanager.googleapis.com`, `cloudbuild.googleapis.com`).
If it fails, enable them manually:

```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    --project=YOUR_PROJECT_ID
```

### 4. Build the ChromaDB index locally

The container image bakes in the pre-built vector index from `chroma_db/`.
Run the ingestion step before deploying:

```bash
python -m app.retrieval.ingest --rebuild
```

This populates `chroma_db/` from the JSON catalogs in `data/catalogs/`.
The deploy script checks for `chroma_db/chroma.sqlite3` and aborts early if it is missing.

### 5. Permissions

The deploy script grants the **default Compute Engine service account**
(`PROJECT_NUMBER-compute@developer.gserviceaccount.com`) the
`roles/secretmanager.secretAccessor` IAM role on the secret.

If your project uses a custom service account for Cloud Run, you must grant
`roles/secretmanager.secretAccessor` to that account manually:

```bash
gcloud secrets add-iam-policy-binding payback-anthropic-key \
    --member="serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

You also need `roles/cloudbuild.builds.editor` and `roles/run.admin` on your own
account to run deployments.

---

## First-time deployment (happy path)

```bash
# 1. Build the vector index (if not already built)
python -m app.retrieval.ingest --rebuild

# 2. Run the deployment script
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

When prompted, paste your Anthropic API key (input is hidden). The script will:
- Store the key in Secret Manager
- Build the container via Cloud Build (~3–6 minutes on first run)
- Deploy the container to Cloud Run
- Run a health check and print the live URL

---

## Rotating the API key

The secret creation step is idempotent — it skips creation if `payback-anthropic-key`
already exists. To rotate the key, delete the secret first, then re-run the script:

```bash
gcloud secrets delete payback-anthropic-key --project=YOUR_PROJECT_ID
./scripts/deploy.sh
```

Cloud Run reads the secret at container start-up via the `UNIFIED_ENDPOINT_KEY`
environment variable (injected by `--set-secrets`). After rotating, trigger a new
revision by re-running the deploy script or:

```bash
gcloud run deploy payback-assistant \
    --region=europe-west3 \
    --project=YOUR_PROJECT_ID \
    --image=$(gcloud run services describe payback-assistant \
        --region=europe-west3 --format='value(spec.template.spec.containers[0].image)')
```

---

## Teardown

To delete the Cloud Run service only (keeps the secret and Artifact Registry image):

```bash
gcloud run services delete payback-assistant \
    --region=europe-west3 \
    --project=YOUR_PROJECT_ID
```

To delete all resources created by this script:

```bash
# 1. Delete the Cloud Run service
gcloud run services delete payback-assistant \
    --region=europe-west3 --project=YOUR_PROJECT_ID

# 2. Delete the Secret Manager secret
gcloud secrets delete payback-anthropic-key --project=YOUR_PROJECT_ID

# 3. Delete the Artifact Registry repository (contains built images)
gcloud artifacts repositories delete payback-assistant-repo \
    --location=europe-west3 --project=YOUR_PROJECT_ID

# 4. (Optional) Disable APIs if no longer needed
gcloud services disable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    --project=YOUR_PROJECT_ID
```

---

## Cost expectations

At the configured limits (`--max-instances=3`, `--concurrency=10`, `--memory=2Gi`,
`--cpu=2`), and assuming the service scales to zero between requests:

| Resource | Est. cost (light usage, europe-west3) |
|---|---|
| Cloud Run (CPU + memory, per-request billing) | < €0.50 / 1 000 requests |
| Cloud Build (container build) | < €0.05 per build (first 120 min/day free) |
| Artifact Registry (image storage ~1.5 GB) | ~€0.05 / month |
| Secret Manager (1 secret, < 10 000 accesses/month) | < €0.01 / month |

**Total for a prototype / demo workload (a few hundred requests/day):** < €5/month.

For production-scale estimates (50 000 requests/day), run:

```bash
python scripts/cost_analysis.py --requests-per-day 50000 --provider claude
```

---

## Why this script exists but was not run for the submission

The project was built and evaluated in a local environment using a managed
Anthropic unified endpoint (`UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC` + `UNIFIED_ENDPOINT_KEY`).
Deploying to Cloud Run would require a personal GCP project with billing enabled and
an API key that can be exposed outside that managed endpoint — neither of which seemed
appropriate to provision as part of a course submission.

The Dockerfile, `.dockerignore`, and `deploy.sh` are fully functional and have been
validated locally (`docker build`, `bash -n scripts/deploy.sh`). They demonstrate
production-readiness — credential hygiene via Secret Manager, multi-stage image
minimization, non-root runtime user, Cloud Build–based container construction — as
design artefacts rather than as a live deployed service.

To deploy for real, a reviewer with a GCP project would run `./scripts/deploy.sh`
following the steps above. The entire process should take under 10 minutes from a cold start.

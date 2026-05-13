#!/usr/bin/env bash
# PAYBACK Assistant — Cloud Run deployment script
#
# What this script does:
#   1. Verifies prerequisites (gcloud, project, APIs)
#   2. Stores the Anthropic API key in Secret Manager (once; idempotent on reruns)
#   3. Deploys the app to Cloud Run in europe-west3 via `gcloud run deploy --source .`
#      (Cloud Build builds the container — no local Docker daemon required)
#   4. Runs a health-check against the deployed URL and prints next steps
#
# Prerequisites: see scripts/deploy.README.md for the full setup guide.
#
# Usage:
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
#
# To rotate the API key:
#   gcloud secrets delete payback-anthropic-key --project=$PROJECT_ID
#   ./scripts/deploy.sh
#
# To tear everything down:
#   gcloud run services delete payback-assistant --region=europe-west3
#   gcloud secrets delete payback-anthropic-key
#   gcloud artifacts repositories delete payback-assistant-repo --location=europe-west3

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG — edit these if you want a different project/region/service name
# ---------------------------------------------------------------------------
SERVICE_NAME="payback-assistant"
REGION="europe-west3"
SECRET_NAME="payback-anthropic-key"
MEMORY="2Gi"
CPU="2"
MAX_INSTANCES="3"
CONCURRENCY="10"
TIMEOUT="60s"

# ---------------------------------------------------------------------------
# COLORS (only when writing to a real terminal)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'
else
    GREEN='' YELLOW='' RED='' NC=''
fi

ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# SANITY CHECKS
# ---------------------------------------------------------------------------
echo ""
echo "=== PAYBACK Assistant — Cloud Run Deployment ==="
echo ""

# 1. gcloud must be installed
if ! command -v gcloud &>/dev/null; then
    fail "gcloud CLI not found. Install it: https://cloud.google.com/sdk/docs/install"
fi
ok "gcloud CLI found: $(gcloud version --format='value(Google Cloud SDK)' 2>/dev/null)"

# 2. A GCP project must be configured
PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
if [ -z "$PROJECT_ID" ]; then
    fail "No GCP project configured. Run: gcloud config set project YOUR_PROJECT_ID"
fi
ok "GCP project: $PROJECT_ID"

# 3. User must be authenticated
ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
if [ -z "$ACCOUNT" ]; then
    fail "Not authenticated. Run: gcloud auth login"
fi
ok "Authenticated as: $ACCOUNT"

# 4. Required APIs must be enabled
echo ""
echo "Checking required APIs..."
REQUIRED_APIS=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "secretmanager.googleapis.com"
    "cloudbuild.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    STATUS=$(gcloud services list --enabled --filter="name:$api" --format="value(name)" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        warn "API not enabled: $api"
        echo "  Enabling $api ..."
        gcloud services enable "$api" --project="$PROJECT_ID" \
            || fail "Failed to enable $api. Check billing is enabled on the project."
        ok "Enabled $api"
    else
        ok "API enabled: $api"
    fi
done

# 5. chroma_db/ must exist (pre-built index required inside the container)
if [ ! -d "chroma_db" ] || [ ! -f "chroma_db/chroma.sqlite3" ]; then
    fail "chroma_db/ not found or empty. Run 'python -m app.retrieval.ingest --rebuild' first, then retry."
fi
ok "chroma_db/ present ($(du -sh chroma_db/ | cut -f1))"

# ---------------------------------------------------------------------------
# SECRET HANDLING — store the Anthropic key in Secret Manager (idempotent)
# ---------------------------------------------------------------------------
echo ""
echo "=== Secret Manager setup ==="

SECRET_EXISTS=$(gcloud secrets describe "$SECRET_NAME" \
    --project="$PROJECT_ID" \
    --format="value(name)" 2>/dev/null || true)

if [ -n "$SECRET_EXISTS" ]; then
    ok "Secret '$SECRET_NAME' already exists — skipping creation."
    warn "To rotate the key: delete the secret and re-run this script."
    warn "  gcloud secrets delete $SECRET_NAME --project=$PROJECT_ID"
else
    echo ""
    echo "The Anthropic API key will be stored in Secret Manager as '$SECRET_NAME'."
    echo "It will never appear in container environment variable listings or logs."
    echo ""
    # read -rs hides the input (no echo to terminal)
    read -rs -p "Paste your Anthropic API key (input hidden): " ANTHROPIC_KEY
    echo ""

    if [ -z "$ANTHROPIC_KEY" ]; then
        fail "No key entered. Aborting."
    fi

    printf '%s' "$ANTHROPIC_KEY" | gcloud secrets create "$SECRET_NAME" \
        --project="$PROJECT_ID" \
        --replication-policy="automatic" \
        --data-file=- \
        || fail "Failed to create secret '$SECRET_NAME'."

    ok "Secret '$SECRET_NAME' created in Secret Manager."

    # Allow the Cloud Run service account to access the secret.
    # Cloud Run uses the default Compute Engine service account unless overridden.
    PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
    SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

    gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
        --project="$PROJECT_ID" \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet \
        || warn "Could not set IAM binding on secret. Deployment may fail if the service account lacks access. See deploy.README.md § Permissions."

    ok "IAM binding set: $SERVICE_ACCOUNT → secretAccessor"
fi

# ---------------------------------------------------------------------------
# DEPLOY
# ---------------------------------------------------------------------------
echo ""
echo "=== Deploying to Cloud Run ($REGION) ==="
echo ""
echo "This uses 'gcloud run deploy --source .' which builds the container via"
echo "Cloud Build using the Dockerfile in this directory. No local Docker required."
echo ""

# Note: --set-secrets maps the Secret Manager secret to the env var name
# that ClaudeClient reads (UNIFIED_ENDPOINT_KEY). UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC
# is left unset so the client uses the standard api.anthropic.com endpoint.
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --memory="$MEMORY" \
    --cpu="$CPU" \
    --max-instances="$MAX_INSTANCES" \
    --concurrency="$CONCURRENCY" \
    --timeout="$TIMEOUT" \
    --allow-unauthenticated \
    --set-secrets="UNIFIED_ENDPOINT_KEY=${SECRET_NAME}:latest" \
    --set-env-vars="DEFAULT_LLM_PROVIDER=claude,EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2,VECTOR_STORE=local,CHROMA_PERSIST_DIR=/app/chroma_db" \
    || fail "Deployment failed. Check Cloud Build logs: https://console.cloud.google.com/cloud-build/builds?project=$PROJECT_ID"

# ---------------------------------------------------------------------------
# POST-DEPLOY
# ---------------------------------------------------------------------------
echo ""
echo "=== Post-deploy health check ==="

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format="value(status.url)" 2>/dev/null)

if [ -z "$SERVICE_URL" ]; then
    warn "Could not retrieve service URL automatically."
    warn "Check: https://console.cloud.google.com/run?project=$PROJECT_ID"
else
    # Wait a moment for the service to warm up
    echo "Waiting 5 seconds for service to warm up..."
    sleep 5

    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" 2>/dev/null || echo "000")

    if [ "$HTTP_STATUS" = "200" ]; then
        ok "Health check passed (HTTP $HTTP_STATUS)"
    else
        warn "Health check returned HTTP $HTTP_STATUS — the service may still be starting."
        warn "Wait 30 seconds and try: curl ${SERVICE_URL}/health"
    fi

    echo ""
    echo "========================================================"
    ok "Deployment complete!"
    echo "========================================================"
    echo ""
    echo "  Service URL:   $SERVICE_URL"
    echo "  Swagger UI:    ${SERVICE_URL}/docs"
    echo ""
    echo "  Sample query:"
    echo "    curl -X POST ${SERVICE_URL}/assist \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"query\": \"Windeln bei dm\"}'"
    echo ""
    echo "  To roll back or delete:"
    echo "    gcloud run services delete $SERVICE_NAME --region=$REGION --project=$PROJECT_ID"
    echo ""
    echo "  To delete all resources (full teardown):"
    echo "    See scripts/deploy.README.md § Teardown"
    echo ""
fi

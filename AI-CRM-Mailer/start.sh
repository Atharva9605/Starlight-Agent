#!/bin/bash
# ============================================================
#  Starlight AI-CRM Mailer — Local launcher (Linux / macOS)
# ============================================================
set -e
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║     Starlight AI-CRM Mailer  — Startup      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 1. Check .env
if [ ! -f ".env" ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill in your keys."
  exit 1
fi

# 2. Check for API key placeholder
if grep -q "PASTE_YOUR_KEY_HERE" .env; then
  echo "WARNING: AZURE_OPENAI_API_KEY in .env is still a placeholder."
  echo "         Edit .env and set your real Azure OpenAI key before ingesting or generating."
  echo ""
fi

# 3. Create static image directory (for local blob backend)
mkdir -p static/page_images

# 4. Start static file server in background (for product image URLs in emails)
STATIC_PORT=8000
if lsof -ti:$STATIC_PORT > /dev/null 2>&1; then
  echo "  [INFO] Port $STATIC_PORT already in use — static server assumed running."
else
  echo "  [START] Static image server → http://localhost:$STATIC_PORT"
  python3 -m http.server $STATIC_PORT --directory static &
  STATIC_PID=$!
  echo "  [PID $STATIC_PID] Static server started."
fi

echo ""
echo "  [START] Streamlit app → http://localhost:8501"
echo ""

# 5. Start Streamlit
streamlit run app.py --server.port 8501 --server.headless true

# Cleanup on exit
if [ ! -z "$STATIC_PID" ]; then
  kill $STATIC_PID 2>/dev/null || true
fi

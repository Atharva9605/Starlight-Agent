"""
Setup checker for Starlight AI-CRM Mailer.
Run this BEFORE ingest_catalogues.py to verify everything is configured.

Usage:
    cd AI-CRM-Mailer
    python check_setup.py
"""
import os
import sys
from pathlib import Path

here = Path(__file__).parent
DIVIDER = "─" * 55

print(f"\n{'═'*55}")
print("  Starlight AI-CRM Mailer — Setup Check")
print(f"{'═'*55}\n")

# ---------------------------------------------------------------------------
# 1. .env file
# ---------------------------------------------------------------------------
print("1. .env file")
env_path = here / ".env"
if env_path.exists():
    print(f"   ✓  Found: {env_path}")
else:
    print(f"   ✗  NOT FOUND: {env_path}")
    print( "      Create it — see .env.example for the template.")
    print( "      Minimum required content:")
    print( "        AZURE_OPENAI_ENDPOINT=https://openai-04.openai.azure.com/")
    print( "        AZURE_OPENAI_API_KEY=<your key>")
    sys.exit(1)

# Load .env
from dotenv import load_dotenv
load_dotenv(dotenv_path=env_path)
print()

# ---------------------------------------------------------------------------
# 2. Required env variables
# ---------------------------------------------------------------------------
print("2. Required environment variables")

required = {
    "AZURE_OPENAI_ENDPOINT":    "https://openai-04.openai.azure.com/",
    "AZURE_OPENAI_API_KEY":     "<your key>",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
}
optional = {
    "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    "BLOB_BACKEND":             "local",
    "STATIC_SERVER_URL":        "http://localhost:8000",
}

all_ok = True
for var, example in required.items():
    val = os.getenv(var, "")
    if not val or "PASTE_YOUR" in val:
        print(f"   ✗  {var}  ← MISSING or placeholder  (example: {example})")
        all_ok = False
    else:
        # Mask the API key for display
        display = val if "KEY" not in var else val[:8] + "..." + val[-4:]
        print(f"   ✓  {var} = {display}")

for var, default in optional.items():
    val = os.getenv(var, default)
    print(f"   ~  {var} = {val}  (optional, default={default})")

if not all_ok:
    print(f"\n   Fix the missing values in: {env_path}\n")
    sys.exit(1)

# Endpoint must start with https://
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
if not endpoint.startswith("https://"):
    print(f"\n   ✗  AZURE_OPENAI_ENDPOINT must start with 'https://'")
    print(f"      Current: '{endpoint}'")
    sys.exit(1)
print()

# ---------------------------------------------------------------------------
# 3. Python packages
# ---------------------------------------------------------------------------
print("3. Required Python packages")

packages = {
    "openai":        "pip install openai",
    "chromadb":      "pip install chromadb",
    "fitz":          "pip install pymupdf",
    "PIL":           "pip install pillow",
    "dotenv":        "pip install python-dotenv",
    "jinja2":        "pip install jinja2",
    "streamlit":     "pip install streamlit",
    "pandas":        "pip install pandas",
    "openpyxl":      "pip install openpyxl",
    "requests":      "pip install requests",
    "bs4":           "pip install beautifulsoup4",
    "dateutil":      "pip install python-dateutil",
}

missing_pkgs = []
for pkg, install_cmd in packages.items():
    try:
        __import__(pkg)
        print(f"   ✓  {pkg}")
    except ImportError:
        print(f"   ✗  {pkg}  ← run: {install_cmd}")
        missing_pkgs.append(install_cmd)

if missing_pkgs:
    print(f"\n   Install all at once:")
    print(f"   pip install openai chromadb pymupdf pillow python-dotenv "
          f"jinja2 streamlit pandas openpyxl beautifulsoup4 python-dateutil")
    sys.exit(1)
print()

# ---------------------------------------------------------------------------
# 4. PDF files
# ---------------------------------------------------------------------------
print("4. Catalogue PDFs")
pdf_paths = [
    here.parent / "STARLIGHT KITCHEN & FURNITURE 2024-2025 (3).pdf",
    here.parent / "Starlight_Linear_Catalogue.PDF",
]
pdfs_ok = True
for p in pdf_paths:
    if p.exists():
        size_mb = p.stat().st_size / 1_048_576
        print(f"   ✓  {p.name}  ({size_mb:.1f} MB)")
    else:
        print(f"   ✗  NOT FOUND: {p}")
        pdfs_ok = False

if not pdfs_ok:
    print("      Make sure the PDFs are in the repo root (one level above AI-CRM-Mailer/).")
print()

# ---------------------------------------------------------------------------
# 5. Quick Azure OpenAI connectivity test
# ---------------------------------------------------------------------------
print("5. Azure OpenAI connection test")
try:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=endpoint.rstrip("/") + "/",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or "gpt-4o"
    )
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        max_tokens=5,
        temperature=0,
    )
    reply = resp.choices[0].message.content.strip()
    print(f"   ✓  Connected!  Model '{deployment}' replied: {reply!r}")
except Exception as exc:
    print(f"   ✗  Connection FAILED: {exc}")
    print( "      Check AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and")
    print( "      AZURE_OPENAI_DEPLOYMENT_NAME in your .env file.")
    sys.exit(1)
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"{'═'*55}")
print("  All checks passed!")
print(f"{'═'*55}")
print()
print("  Next steps:")
print("  1.  python ingest_catalogues.py   ← index the catalogues (once)")
print("  2.  start.bat                     ← launch the app")
print()

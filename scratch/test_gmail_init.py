import logging
import sys
import os

# Add the current directory to sys.path so we can import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from send_eml_gsuite import get_gmail_service

logging.basicConfig(level=logging.INFO)

print("Attempting to initialize Gmail service...")
svc = get_gmail_service()
if svc:
    print("✅ Success: Gmail service created.")
else:
    print("❌ Failure: Gmail service could not be created.")

import json
try:
    with open('service_account.json', 'r') as f:
        content = f.read()
        print(f"File length: {len(content)}")
        print(f"Char at 1434: {content[1434]!r}")
        print(f"Context: {content[1430:1440]!r}")
        json.loads(content)
        print("JSON is valid")
except Exception as e:
    print(f"Error: {e}")

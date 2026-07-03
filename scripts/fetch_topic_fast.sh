#!/bin/bash
# Fast fetch of a single Kaggle NeuroGolf discussion topic via agent-browser.
# Usage: fetch_topic_fast.sh <topic_id> [scrolls]
set -e
TOPIC_ID="$1"
SCROLLS="${2:-3}"
OUT="/home/z/my-project/data/topics/topic_${TOPIC_ID}.txt"

agent-browser open "https://www.kaggle.com/competitions/neurogolf-2026/discussion/${TOPIC_ID}" > /dev/null 2>&1 || true
# Brief wait for React render
sleep 4
# Scroll a few times to lazy-load comments
for i in $(seq 1 "$SCROLLS"); do
  agent-browser eval "window.scrollTo(0, document.body.scrollHeight)" > /dev/null 2>&1 || true
  sleep 1
done

# Extract body innerText and save (JSON-encoded string)
agent-browser eval "document.body.innerText" > "${OUT}.raw" 2>&1 || true
python3 - "$OUT" <<'PY'
import json, sys
fp = sys.argv[1] + '.raw'
with open(fp) as f:
    raw = f.read()
s = raw.strip()
if s.startswith('"') and s.endswith('"'):
    try:
        text = json.loads(s)
    except Exception:
        text = raw
else:
    text = raw
with open(sys.argv[1], 'w') as f:
    f.write(text)
print(f"Saved {len(text)} chars to {sys.argv[1]}")
PY
rm -f "${OUT}.raw"

#!/bin/bash
# Re-fetch topic with extended scrolling for lazy-loaded comments.
# Usage: fetch_topic_deep.sh <topic_id>
set -e
TOPIC_ID="$1"
OUT="/home/z/my-project/data/topics/topic_${TOPIC_ID}.txt"

agent-browser open "https://www.kaggle.com/competitions/neurogolf-2026/discussion/${TOPIC_ID}" > /dev/null 2>&1 || true
sleep 5
# Aggressive scroll - many comments are lazy-loaded
for i in $(seq 1 12); do
  agent-browser eval "window.scrollTo(0, document.body.scrollHeight)" > /dev/null 2>&1 || true
  sleep 0.7
done

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

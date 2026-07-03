#!/bin/bash
# Fetch a single Kaggle NeuroGolf discussion topic via agent-browser.
# Usage: fetch_topic.sh <topic_id>
set -e
TOPIC_ID="$1"
OUT="/home/z/my-project/data/topics/topic_${TOPIC_ID}.txt"

agent-browser open "https://www.kaggle.com/competitions/neurogolf-2026/discussion/${TOPIC_ID}" > /dev/null 2>&1
agent-browser wait --load networkidle > /dev/null 2>&1
# Wait a moment for React to render comments
sleep 3
# Scroll to bottom a few times to lazy-load comments
for i in 1 2 3 4 5; do
  agent-browser eval "window.scrollTo(0, document.body.scrollHeight)" > /dev/null 2>&1
  sleep 1
done
# Also try scrolling the comments container
agent-browser eval "const c = document.querySelector('[class*=CommentList],[class*=comment]'); if(c){c.scrollTop = c.scrollHeight;}" > /dev/null 2>&1
sleep 2

# Extract body innerText and save
agent-browser eval "document.body.innerText" > "${OUT}.raw" 2>&1
# Clean: the eval output is a JSON-quoted string with \n escapes
python3 - "$OUT" <<'PY'
import json, sys, re
fp = sys.argv[1] + '.raw'
with open(fp) as f:
    raw = f.read()
# Try to parse as JSON (eval returns JSON-quoted string)
try:
    # The CLI may wrap output in quotes; strip leading/trailing whitespace
    s = raw.strip()
    if s.startswith('"') and s.endswith('"'):
        text = json.loads(s)
    else:
        text = raw
except Exception:
    text = raw
with open(sys.argv[1], 'w') as f:
    f.write(text)
print(f"Saved {len(text)} chars to {sys.argv[1]}")
PY
rm -f "${OUT}.raw"

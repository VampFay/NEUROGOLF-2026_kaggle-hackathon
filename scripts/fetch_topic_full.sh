#!/bin/bash
# Re-fetch topic with comment extraction via direct DOM querying.
# Usage: fetch_topic_full.sh <topic_id>
set -e
TOPIC_ID="$1"
OUT="/home/z/my-project/data/topics/topic_${TOPIC_ID}.txt"

agent-browser close --all > /dev/null 2>&1 || true
agent-browser open "https://www.kaggle.com/competitions/neurogolf-2026/discussion/${TOPIC_ID}" > /dev/null 2>&1 || true
sleep 5
# Scroll to load lazy comments
for i in $(seq 1 8); do
  agent-browser eval "window.scrollTo(0, document.body.scrollHeight)" > /dev/null 2>&1 || true
  sleep 0.5
done

# Extract: page title + body innerText + all comment containers' text
agent-browser eval "(function(){
  // Get full body text
  const bodyText = document.body.innerText;
  // Also gather comment containers (Kaggle uses data-testid on options-menu-button)
  const commentBtns = document.querySelectorAll('[data-testid=options-menu-button]');
  const comments = [];
  commentBtns.forEach(btn => {
    // Walk up to find comment container
    let el = btn.parentElement;
    for (let i=0; i<5; i++) {
      if (!el) break;
      if (el.innerText && el.innerText.length > 20 && el.innerText.length < 5000) {
        comments.push(el.innerText);
        break;
      }
      el = el.parentElement;
    }
  });
  return JSON.stringify({bodyText: bodyText, comments: comments});
})()" > "${OUT}.raw" 2>&1 || true

python3 - "$OUT" <<'PY'
import json, sys
fp = sys.argv[1] + '.raw'
with open(fp) as f:
    raw = f.read().strip()
# The eval output is a JSON-encoded string of a JSON object — i.e. double-encoded
try:
    obj_str = json.loads(raw) if raw.startswith('"') else raw
    obj = json.loads(obj_str)
    bodyText = obj.get('bodyText', '')
    comments = obj.get('comments', [])
except Exception as e:
    print(f"Parse error: {e}", file=sys.stderr)
    bodyText = raw
    comments = []

out = []
out.append('=== BODY TEXT ===')
out.append(bodyText)
if comments:
    out.append('')
    out.append('=== COMMENTS (extracted via DOM) ===')
    for i, c in enumerate(comments, 1):
        out.append(f'--- Comment {i} ---')
        out.append(c)
text = '\n'.join(out)
with open(sys.argv[1], 'w') as f:
    f.write(text)
print(f"Saved {len(text)} chars (body {len(bodyText)}, comments {len(comments)}) to {sys.argv[1]}")
PY
rm -f "${OUT}.raw"
